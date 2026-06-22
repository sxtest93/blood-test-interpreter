import json
from io import BytesIO

import pdfplumber
from groq import Groq

MODEL = "llama-3.3-70b-versatile"


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


def parse_pdf_to_blood_test(pdf_text: str, client: Groq) -> list[dict]:
    """Use Groq to extract structured blood test values from raw PDF text."""
    prompt = f"""Extract every blood test value from this lab report. Do not skip any measurable result.

Lab report text:
{pdf_text}

Output ONLY a valid JSON array — no prose, no markdown fences.
For each test value found:
  name            : test name exactly as shown in the report
  value           : numeric result as a string
  unit            : unit of measurement (e.g. mg/dL, g/dL, K/uL, %)
  reference_range : the normal range shown in the report (e.g. "13.5-17.5"), or "not provided"

[{{"name": "...", "value": "...", "unit": "...", "reference_range": "..."}}]"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=4096,
    )
    return json.loads(_extract_json(response.choices[0].message.content))
