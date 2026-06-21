import os
import json
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
import anthropic

from agent.graph import create_graph
from agent.pdf_parser import extract_text_from_pdf, parse_pdf_to_blood_test

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Blood Test Interpreter",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded",
)

NODE_LABELS = {
    "categorize_values": "Organizing values by body system",
    "explain_values":    "Explaining each value in plain English",
    "generate_summary":  "Writing your personal summary",
}

ATTENTION_COLOR = {
    "discuss": "🔴",
    "watch":   "🟡",
    "none":    "🟢",
}

# ── Sidebar — patient context ─────────────────────────────────────────────────

with st.sidebar:
    st.header("About You")
    st.caption("This personalizes the interpretation to your situation.")

    age  = st.text_input("Age", placeholder="e.g. 34")
    sex  = st.selectbox("Biological sex", ["", "Male", "Female", "Other / prefer not to say"])
    conditions = st.text_area(
        "Known health conditions",
        placeholder="Type 2 diabetes\nHypertension",
        height=90,
    )
    medications = st.text_area(
        "Current medications",
        placeholder="Metformin 500mg\nAtorvastatin",
        height=90,
    )
    symptoms = st.text_area(
        "Current symptoms or concerns",
        placeholder="Fatigue\nShortness of breath",
        height=90,
    )

    st.divider()
    st.header("API Key")
    api_key_input = st.text_input(
        "Anthropic API key",
        value=os.getenv("ANTHROPIC_API_KEY", ""),
        type="password",
        help="Stored only in this session. Not sent anywhere except Anthropic.",
    )


def get_patient_context() -> dict:
    return {
        "age":        age.strip() or "unknown",
        "sex":        sex or "unknown",
        "conditions": [c.strip() for c in conditions.splitlines() if c.strip()],
        "medications":[m.strip() for m in medications.splitlines() if m.strip()],
        "symptoms":   [s.strip() for s in symptoms.splitlines()    if s.strip()],
    }


# ── Main page ─────────────────────────────────────────────────────────────────

st.title("🩸 Blood Test Interpreter")
st.caption("Your blood test shows 47 values. You understand 3. Let's fix that.")

st.warning(
    "**Educational purposes only — not medical advice.** "
    "Always consult your doctor or healthcare provider for medical decisions.",
    icon="⚠️",
)

# ── Input tabs ────────────────────────────────────────────────────────────────

upload_tab, manual_tab = st.tabs(["📄 Upload Report (PDF or JSON)", "✏️ Enter Manually"])

raw_results = None

with upload_tab:
    uploaded = st.file_uploader(
        "Upload your blood test report",
        type=["pdf", "json"],
        help="PDF: most lab printouts work. JSON: see examples/sample_blood_test.json for the format.",
    )

    if uploaded:
        if uploaded.name.endswith(".json"):
            try:
                raw_results = json.load(uploaded)
                st.success(f"Loaded {len(raw_results)} values from JSON.")
            except json.JSONDecodeError as e:
                st.error(f"Could not parse JSON: {e}")

        elif uploaded.name.endswith(".pdf"):
            if not api_key_input:
                st.info("Add your Anthropic API key in the sidebar to parse the PDF.")
            else:
                with st.spinner("Reading PDF..."):
                    try:
                        pdf_text = extract_text_from_pdf(BytesIO(uploaded.read()))
                        client   = anthropic.Anthropic(api_key=api_key_input)
                        raw_results = parse_pdf_to_blood_test(pdf_text, client)
                        st.success(f"Extracted {len(raw_results)} values from PDF.")
                    except ValueError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"PDF parsing failed: {e}")

        if raw_results:
            with st.expander("Preview extracted values"):
                st.dataframe(raw_results, use_container_width=True)

with manual_tab:
    st.caption("Add your values one row at a time. Reference range is optional.")
    manual_data = st.data_editor(
        [{"name": "", "value": "", "unit": "", "reference_range": ""}],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "name":            st.column_config.TextColumn("Test Name", width="medium"),
            "value":           st.column_config.TextColumn("Value",     width="small"),
            "unit":            st.column_config.TextColumn("Unit",      width="small"),
            "reference_range": st.column_config.TextColumn("Reference Range", width="medium"),
        },
    )
    filled = [r for r in manual_data if r.get("name") and r.get("value")]
    if filled:
        raw_results = filled

# ── Analyse button ─────────────────────────────────────────────────────────────

st.divider()
run_disabled = not raw_results or not api_key_input
run_btn = st.button(
    "Analyse Blood Test",
    type="primary",
    disabled=run_disabled,
    help="Add your API key and blood test values to enable." if run_disabled else "",
)

if run_disabled and not api_key_input:
    st.caption("Add your Anthropic API key in the sidebar to run the analysis.")
elif run_disabled and not raw_results:
    st.caption("Upload a report or enter values manually above.")

# ── Run the agent ─────────────────────────────────────────────────────────────

if run_btn and raw_results and api_key_input:
    patient_context = get_patient_context()
    graph = create_graph(api_key_input)

    initial_state = {
        "raw_results":        raw_results,
        "patient_context":    patient_context,
        "categorized_values": {},
        "explanations":       [],
        "risk_flags":         [],
        "overall_summary":    "",
        "action_items":       [],
    }

    final_state = dict(initial_state)

    with st.status("Analyzing your blood test...", expanded=True) as status:
        for chunk in graph.stream(initial_state):
            for node_name, node_output in chunk.items():
                st.write(f"✓ {NODE_LABELS.get(node_name, node_name)}")
                final_state.update(node_output)
        status.update(label="Analysis complete!", state="complete", expanded=False)

    st.session_state["results"] = final_state

# ── Display results ────────────────────────────────────────────────────────────

if "results" in st.session_state:
    r = st.session_state["results"]

    st.divider()
    st.header("Your Results")

    # Big picture
    st.subheader("The Big Picture")
    st.info(r["overall_summary"], icon="💡")

    # Counts
    discuss = [e for e in r["explanations"] if e.get("attention_level") == "discuss"]
    watch   = [e for e in r["explanations"] if e.get("attention_level") == "watch"]
    normal  = [e for e in r["explanations"] if e.get("attention_level") == "none"]

    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 Discuss with doctor", len(discuss))
    col2.metric("🟡 Worth monitoring",    len(watch))
    col3.metric("🟢 Looking good",        len(normal))

    st.divider()

    # Values that need attention
    if discuss:
        st.subheader("🔴 Discuss With Your Doctor")
        for e in discuss:
            with st.expander(f"**{e['name']}** — {e['your_result'][:60]}"):
                st.markdown(f"**What it measures:** {e['what_it_measures']}")
                st.markdown(f"**Your result:** {e['your_result']}")
                st.markdown(f"**What can affect it:** {e['what_affects_it']}")
                st.error(e["attention_reason"])

    if watch:
        st.subheader("🟡 Worth Monitoring")
        for e in watch:
            with st.expander(f"**{e['name']}** — {e['your_result'][:60]}"):
                st.markdown(f"**What it measures:** {e['what_it_measures']}")
                st.markdown(f"**Your result:** {e['your_result']}")
                st.markdown(f"**What can affect it:** {e['what_affects_it']}")
                st.warning(e["attention_reason"])

    if normal:
        st.subheader("🟢 Looking Good")
        with st.expander(f"All {len(normal)} values in normal range"):
            for e in normal:
                st.markdown(f"**{e['name']}** — {e['your_result']}")
                st.caption(e["attention_reason"])

    # Doctor questions
    st.divider()
    st.subheader("📋 Questions to Bring to Your Doctor")
    for i, item in enumerate(r["action_items"], 1):
        st.markdown(f"**{i}.** {item}")

    # Download report as JSON
    st.divider()
    st.download_button(
        "Download full report (JSON)",
        data=json.dumps(r, indent=2),
        file_name="blood_test_report.json",
        mime="application/json",
    )
