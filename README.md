# Blood Test Interpreter Agent

Your blood test shows 47 values and you understand 3. Without this, you'd Google each value separately, land on scary medical sites, and still be confused by what "reference range" means *for you*.

This agent reads your full blood panel and returns a plain-English report — personalized to your age, sex, conditions, and medications — with the most important values flagged and a list of questions to bring to your doctor.

> **Disclaimer:** Educational purposes only. Not medical advice. Always consult a healthcare provider.

---

## How It Works

The agent is built with [LangGraph](https://github.com/langchain-ai/langgraph) and powered by Llama 3.3 70B via Groq. It processes your results in three stages:

```
START
  │
  ▼
categorize_values
  Groups your test names by body system
  (Complete Blood Count, Liver Function, Lipid Panel, etc.)
  │
  ▼
explain_values
  For every value: what it measures, what your result means
  for YOU specifically, what can affect it, whether to act.
  Reasons across the full panel with patient-specific context.
  │
  ▼
generate_summary
  Writes a plain-English big picture + a list of questions
  to bring to your next doctor appointment.
  │
  ▼
END → formatted report printed to terminal
```

Each node returns only the state keys it updates — clean separation, easy to extend.

---

## Setup

**Requirements:** Python 3.11+, a [Groq API key](https://console.groq.com) (free).

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd langgraph-project

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env
# Edit .env and paste your GROQ_API_KEY
```

---

## Usage

### Streamlit UI (recommended)

```bash
streamlit run app.py
```

Opens in your browser. Upload a PDF or JSON report, fill in your details in the sidebar, and click **Analyse Blood Test**.

### CLI — with a JSON file

```bash
python main.py examples/sample_blood_test.json
```

The agent asks a few quick questions about you (age, sex, conditions, medications) and returns a full report.

### CLI — interactive entry

```bash
python main.py
```

Enter values one at a time:
```
  > Hemoglobin, 13.2, g/dL, 13.5-17.5
  > Ferritin, 8, ng/mL, 20-500
  > done
```

### JSON input format

```json
[
  {
    "name": "Hemoglobin",
    "value": "13.2",
    "unit": "g/dL",
    "reference_range": "13.5-17.5"
  }
]
```

See [examples/sample_blood_test.json](examples/sample_blood_test.json) for a 20-value panel with realistic data.

---

## Example Output

```
==================================================================
  YOUR BLOOD TEST — IN PLAIN ENGLISH
==================================================================

THE BIG PICTURE
----------------------------------------
Overall your results look reasonably healthy, but there are a
few patterns worth discussing with your doctor. Your iron stores
are low — which likely explains fatigue if you've been feeling
tired — and your LDL cholesterol is above the recommended target.
Everything else, including your kidney function, liver, thyroid,
and blood sugar, is in good shape.

[!] DISCUSS WITH YOUR DOCTOR  (2 values)
----------------------------------------

  Ferritin
    What it measures : Iron stored in your body, available for making red blood cells
    Your result      : 8 ng/mL — significantly below the 20-500 range; your iron stores are depleted
    Note             : Low ferritin is one of the most common causes of fatigue in your age group...

  LDL
    What it measures : "Bad" cholesterol carried through your bloodstream
    Your result      : 138 mg/dL — above the under-100 target for most adults
    Note             : Not an emergency, but a consistent LDL this high raises long-term heart risk...

[~] WORTH MONITORING  (2 values)
...

[OK] LOOKING GOOD  (16 values)
...

QUESTIONS TO BRING TO YOUR DOCTOR
----------------------------------------
  1. My ferritin is 8 — should I start an iron supplement, or is there a reason you'd want to investigate the cause first?
  2. Ask whether my LDL of 138 requires dietary changes or medication given my other risk factors.
  ...
```

---

## Project Structure

```
langgraph-project/
├── agent/
│   ├── __init__.py
│   ├── state.py        # BloodTestState TypedDict
│   ├── nodes.py        # Node factory functions (closure pattern)
│   ├── graph.py        # StateGraph wiring
│   └── pdf_parser.py   # PDF text extraction + LLM-powered parsing
├── examples/
│   └── sample_blood_test.json
├── app.py              # Streamlit UI
├── main.py             # CLI entry point + report formatter
├── requirements.txt
├── .env.example
└── .gitignore
```

### Key design decisions

- **Closure-based nodes** — each node factory takes a `Groq` client and returns a plain function. No global state, easily testable.
- **Cross-value reasoning** — the explain node receives the full panel at once, so the model can catch patterns across values (e.g. low MCV + low ferritin + low hemoglobin = iron deficiency, not three separate issues).
- **Streaming state accumulation** — `graph.stream()` is used so progress is printed as each node completes; state updates are merged into a local dict so the final report requires no second `invoke()` call.
- **Patient context** — age, sex, conditions, and medications are injected into the explain and summary prompts, so reference ranges are interpreted for the actual person, not a generic adult.

---

## Tech Stack

| Library | Role |
|---|---|
| `groq` | Llama 3.3 70B inference via Groq API |
| `langgraph` | Multi-node agent graph orchestration |
| `streamlit` | Browser UI with file upload and live progress |
| `pdfplumber` | PDF text extraction |
| `python-dotenv` | API key loading from `.env` |
