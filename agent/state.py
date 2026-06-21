from typing import TypedDict


class BloodTestState(TypedDict):
    raw_results: list[dict]       # [{"name": "Hemoglobin", "value": "13.2", "unit": "g/dL", "reference_range": "13.5-17.5"}]
    patient_context: dict         # age, sex, known conditions, medications, symptoms
    categorized_values: dict      # grouped by body system: {"Liver Function": ["ALT", "AST"], ...}
    explanations: list[dict]      # per-value plain-English breakdowns
    risk_flags: list[dict]        # values that warrant attention
    overall_summary: str          # big-picture plain-English assessment
    action_items: list[str]       # specific questions to bring to the doctor
