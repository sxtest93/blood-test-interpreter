import json
from io import BytesIO

import pdfplumber
import anthropic


def _clean_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text[text.index("\n") + 1:]
    if text.endswith("```"):
        text = text[: text.rindex("```")]
    return text.strip()


def extract_text_from_pdf(file) -> str:
    """Accept a file path string or a file-like object (e.g. BytesIO from Streamlit)."""
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    text = "\n\n".join(p for p in pages if p.strip())
    if not text:
        raise ValueError(
            "No text could be extracted from this PDF. "
            "Scanned (image-only) PDFs are not supported."
        )
    return text


def parse_pdf_to_blood_test(pdf_text: str, client: anthropic.Anthropic) -> list[dict]:
    """Use Claude to extract structured blood test values from raw PDF text."""
    response = client.messages.create(
        model="claude-opus-4-8",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": f"""Extract every blood test value from this lab report. Do not skip any measurable result.

Lab report text:
{pdf_text}

Output ONLY a valid JSON array — no prose, no markdown fences.
For each test value found:
  name            : test name exactly as shown in the report
  value           : numeric result as a string
  unit            : unit of measurement (e.g. mg/dL, g/dL, K/uL, %)
  reference_range : the normal range shown in the report (e.g. "13.5-17.5"), or "not provided"

[{{"name": "...", "value": "...", "unit": "...", "reference_range": "..."}}]""",
            }
        ],
    )
    raw = response.content[0].text
    return json.loads(_clean_json(raw))
