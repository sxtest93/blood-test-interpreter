import json
from groq import Groq

from .state import BloodTestState

MODEL = "llama-3.1-8b-instant"


def _extract_json(text: str) -> str:
    """Extract the first complete JSON object or array from LLM response."""
    text = text.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            if part.startswith(("json", "\n")):
                text = part[part.index("\n") + 1:] if "\n" in part else part
                break
        else:
            text = parts[1] if len(parts) > 1 else text
    text = text.strip()
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth, in_str, escape = 0, False, False
        for i, ch in enumerate(text[start:], start):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_str:
                escape = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text


def create_categorize_node(client: Groq):
    def categorize_values(state: BloodTestState) -> dict:
        prompt = f"""You are a medical laboratory expert. Categorize these blood test values by body system.

Blood test results:
{json.dumps(state["raw_results"], indent=2)}

Group test names into categories such as: Complete Blood Count, Metabolic Panel, Liver Function, Kidney Function, Lipid Panel, Thyroid, Vitamins & Minerals, Blood Sugar, Other.

Output ONLY valid JSON — no prose, no markdown fences:
{{"Category Name": ["Test Name 1", "Test Name 2"], ...}}"""
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
        )
        text = response.choices[0].message.content
        categorized = json.loads(_extract_json(text))
        return {"categorized_values": categorized}

    return categorize_values


def create_explain_node(client: Groq):
    def explain_values(state: BloodTestState) -> dict:
        p = state["patient_context"]
        patient_desc = f"{p.get('age', 'unknown age')} year old {p.get('sex', 'person')}"
        if p.get("conditions"):
            patient_desc += f", known conditions: {', '.join(p['conditions'])}"
        if p.get("medications"):
            patient_desc += f", current medications: {', '.join(p['medications'])}"
        if p.get("symptoms"):
            patient_desc += f", current concerns: {', '.join(p['symptoms'])}"

        prompt = f"""You are a compassionate doctor explaining blood test results to a patient in plain English.

Patient: {patient_desc}

Blood test results:
{json.dumps(state["raw_results"], indent=2)}

For EVERY value in the list, provide an explanation. Be warm, clear, and specific to this patient's context.
Explain what reference ranges mean for THEIR situation — not just the generic lab range.
Never use unexplained medical jargon. Never be alarmist.

Output ONLY a valid JSON array — no prose, no markdown fences:
[
  {{
    "name": "exact test name from input",
    "what_it_measures": "one plain-English sentence",
    "your_result": "their value compared to range, what it means for them personally",
    "what_affects_it": "lifestyle, diet, or conditions that commonly influence this",
    "attention_level": "none | watch | discuss",
    "attention_reason": "why they should or should not worry — always provide a sentence even for normal values"
  }}
]

attention_level guide:
- none: result is within or acceptably close to range, no action needed
- watch: borderline or trending; worth monitoring at next test
- discuss: meaningfully outside range or clinically significant given their context"""
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
        )
        text = response.choices[0].message.content
        explanations = json.loads(_extract_json(text))
        risk_flags = [e for e in explanations if e.get("attention_level") in ("watch", "discuss")]
        return {"explanations": explanations, "risk_flags": risk_flags}

    return explain_values


def create_summary_node(client: Groq):
    def generate_summary(state: BloodTestState) -> dict:
        p = state["patient_context"]
        patient_desc = f"{p.get('age', 'unknown')} year old {p.get('sex', 'person')}"

        discuss_count = sum(1 for e in state["explanations"] if e.get("attention_level") == "discuss")
        watch_count = sum(1 for e in state["explanations"] if e.get("attention_level") == "watch")
        normal_count = sum(1 for e in state["explanations"] if e.get("attention_level") == "none")

        prompt = f"""You are a compassionate doctor giving a patient the "big picture" from their blood test.

Patient: {patient_desc}
Results breakdown: {normal_count} values look good, {watch_count} to monitor, {discuss_count} to discuss with doctor.

Full analysis:
{json.dumps(state["explanations"], indent=2)}

Write:
1. overall_summary: 3-4 warm, plain-English sentences. What does this test say about their health overall?
   Acknowledge the good, be honest about concerns without being frightening.
   Help them feel informed and empowered, not anxious.

2. action_items: 4-6 specific, concrete questions to bring to their next doctor appointment.
   Make them actionable (e.g. "Ask whether your borderline iron levels could explain your fatigue").

Output ONLY valid JSON — no prose, no markdown fences:
{{"overall_summary": "...", "action_items": ["question 1", "question 2", ...]}}"""
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
        )
        text = response.choices[0].message.content
        result = json.loads(_extract_json(text))
        return {
            "overall_summary": result["overall_summary"],
            "action_items": result["action_items"],
        }

    return generate_summary
