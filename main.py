import os
import sys
import json
from dotenv import load_dotenv

from agent.graph import create_graph

DISCLAIMER = """
╔══════════════════════════════════════════════════════════════════╗
║  IMPORTANT: Educational purposes only. Not medical advice.       ║
║  Always consult your doctor or healthcare provider for any       ║
║  decisions about your health.                                     ║
╚══════════════════════════════════════════════════════════════════╝
"""

NODE_LABELS = {
    "categorize_values": "Organizing your values by body system...",
    "explain_values":    "Explaining each value in plain English...",
    "generate_summary":  "Writing your personal summary...",
}


def format_report(state: dict) -> str:
    lines = []
    lines.append("\n" + "=" * 66)
    lines.append("  YOUR BLOOD TEST — IN PLAIN ENGLISH")
    lines.append("=" * 66)

    lines.append("\nTHE BIG PICTURE")
    lines.append("-" * 40)
    lines.append(state["overall_summary"])

    discuss = [e for e in state["explanations"] if e.get("attention_level") == "discuss"]
    watch   = [e for e in state["explanations"] if e.get("attention_level") == "watch"]
    normal  = [e for e in state["explanations"] if e.get("attention_level") == "none"]

    if discuss:
        lines.append(f"\n[!] DISCUSS WITH YOUR DOCTOR  ({len(discuss)} value{'s' if len(discuss) != 1 else ''})")
        lines.append("-" * 40)
        for e in discuss:
            lines.append(f"\n  {e['name']}")
            lines.append(f"    What it measures : {e['what_it_measures']}")
            lines.append(f"    Your result      : {e['your_result']}")
            lines.append(f"    Note             : {e['attention_reason']}")

    if watch:
        lines.append(f"\n[~] WORTH MONITORING  ({len(watch)} value{'s' if len(watch) != 1 else ''})")
        lines.append("-" * 40)
        for e in watch:
            lines.append(f"\n  {e['name']}")
            lines.append(f"    What it measures : {e['what_it_measures']}")
            lines.append(f"    Your result      : {e['your_result']}")
            lines.append(f"    Note             : {e['attention_reason']}")

    if normal:
        lines.append(f"\n[OK] LOOKING GOOD  ({len(normal)} value{'s' if len(normal) != 1 else ''})")
        lines.append("-" * 40)
        names = ", ".join(e["name"] for e in normal)
        lines.append(f"  {names}")

    lines.append("\nQUESTIONS TO BRING TO YOUR DOCTOR")
    lines.append("-" * 40)
    for i, item in enumerate(state["action_items"], 1):
        lines.append(f"  {i}. {item}")

    lines.append("\n" + "=" * 66)
    return "\n".join(lines)


def collect_patient_context() -> dict:
    print("\nTell me a bit about yourself (press Enter to skip any field):")
    age              = input("  Age: ").strip()
    sex              = input("  Biological sex (male / female / other): ").strip()
    conditions_raw   = input("  Known health conditions (comma-separated): ").strip()
    medications_raw  = input("  Current medications (comma-separated): ").strip()
    symptoms_raw     = input("  Current symptoms or concerns (comma-separated): ").strip()

    return {
        "age":        age or "unknown",
        "sex":        sex or "unknown",
        "conditions": [c.strip() for c in conditions_raw.split(",")  if c.strip()],
        "medications":[m.strip() for m in medications_raw.split(",") if m.strip()],
        "symptoms":   [s.strip() for s in symptoms_raw.split(",")    if s.strip()],
    }


def load_blood_test_from_file(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def collect_blood_test_interactively() -> list[dict]:
    print("\nEnter your blood test values one at a time.")
    print("Format:  TEST NAME, VALUE, UNIT, REFERENCE RANGE")
    print("Example: Hemoglobin, 13.2, g/dL, 13.5-17.5")
    print("Type 'done' when finished.\n")

    results = []
    while True:
        line = input("  > ").strip()
        if line.lower() in ("done", ""):
            if results:
                break
            print("  Please enter at least one value.")
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            print("  Need at least: NAME, VALUE, UNIT")
            continue

        results.append({
            "name":            parts[0],
            "value":           parts[1],
            "unit":            parts[2],
            "reference_range": parts[3] if len(parts) > 3 else "not provided",
        })
        print(f"  Added: {parts[0]}")

    return results


def main():
    load_dotenv()
    api_key = os.getenv("ANTHROPIC_API_KEY")

    if not api_key:
        print("Error: ANTHROPIC_API_KEY is not set. Add it to your .env file.")
        sys.exit(1)

    print(DISCLAIMER)
    print("BLOOD TEST INTERPRETER")
    print("Turning lab values into plain English.\n")

    if len(sys.argv) > 1:
        print(f"Loading blood test from: {sys.argv[1]}")
        raw_results = load_blood_test_from_file(sys.argv[1])
    else:
        raw_results = collect_blood_test_interactively()

    patient_context = collect_patient_context()

    print(f"\nAnalyzing {len(raw_results)} blood test values...\n")

    graph = create_graph(api_key)
    initial_state = {
        "raw_results":       raw_results,
        "patient_context":   patient_context,
        "categorized_values": {},
        "explanations":      [],
        "risk_flags":        [],
        "overall_summary":   "",
        "action_items":      [],
    }

    # Stream execution — accumulate state updates as each node completes
    final_state = dict(initial_state)
    for chunk in graph.stream(initial_state):
        for node_name, node_output in chunk.items():
            print(f"  {NODE_LABELS.get(node_name, node_name)}")
            final_state.update(node_output)

    print(format_report(final_state))


if __name__ == "__main__":
    main()
