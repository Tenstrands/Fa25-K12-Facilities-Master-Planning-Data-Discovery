#!/usr/bin/env python3
import os, json, argparse, re, textwrap
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from pypdf import PdfReader
import pandas as pd

# --- OpenAI SDK (Responses API) ---
from openai import OpenAI  # pip install openai

# ---------- Helpers ----------
def read_pdf_text(pdf_path: Path, max_pages: int | None = None) -> str:
    reader = PdfReader(str(pdf_path))
    pages = reader.pages[: (max_pages or len(reader.pages))]
    text = []
    for p in pages:
        try:
            text.append(p.extract_text() or "")
        except Exception:
            text.append("")
    return "\n\n".join(text)

def chunk_text(s: str, target_tokens: int = 1200) -> list[str]:
    # naive chunking by words so we don’t blow context; adjust if needed
    words = s.split()
    chunk_size = target_tokens * 4  # ~4 chars/token heuristic
    chunks, buf = [], []
    size = 0
    for w in words:
        buf.append(w)
        size += len(w) + 1
        if size >= chunk_size:
            chunks.append(" ".join(buf))
            buf, size = [], 0
    if buf:
        chunks.append(" ".join(buf))
    return chunks

SYSTEM_PROMPT = """You are an analyst scoring K–12 Facility Master Plans using the provided rubric.
Return STRICT JSON only. No prose. Follow the schema exactly.
Scoring categories and allowed ranges:
- Facility Inventory & Condition Assessment: 1–4
- Enrollment & Capacity Planning: 1–4
- Educational Alignment: 1–4
- Outdoor Spaces & Greening: 1–4
- Climate Risk & Mitigation: 0–4
- Energy Efficiency & Resilience: 1–4
- Financial Plan & Capital Strategy: 1–4
For each category, include a concise 1–2 sentence justification with page/section citations if possible.
"""

USER_PROMPT_TEMPLATE = """Rubric (verbatim):
---
{rubric}
---

Plan content (chunked). Combine evidence across chunks. If a category is not covered, score at the appropriate lower tier and say why.

Return JSON with this exact shape:

{{
  "district": "{district}",
  "scores": {{
    "Facility Inventory & Condition Assessment": {{"score": int, "justification": str}},
    "Enrollment & Capacity Planning": {{"score": int, "justification": str}},
    "Educational Alignment": {{"score": int, "justification": str}},
    "Outdoor Spaces & Greening": {{"score": int, "justification": str}},
    "Climate Risk & Mitigation": {{"score": int, "justification": str}},
    "Energy Efficiency & Resilience": {{"score": int, "justification": str}},
    "Financial Plan & Capital Strategy": {{"score": int, "justification": str}}
  }}
}}

Plan chunks:
{chunks}
"""

def call_model(client, model, system, user):
    resp = client.chat.completions.create(
        model=model,
        temperature=0,  # deterministic
        messages=[
            {"role": "system", "content": system + "\nReturn ONLY valid JSON. No prose, no markdown."},
            {"role": "user", "content": user}
        ]
    )
    return resp.choices[0].message.content or ""

import re, json

def parse_possible_json(s: str):
    s = (s or "").strip()
    if not s:
        raise ValueError("Model returned empty output")

    # Strip common wrappers
    s = re.sub(r"^```json\s*|\s*```$", "", s, flags=re.S).strip()

    # If other text leaked, grab the first top-level JSON object
    if not s.startswith("{"):
        m = re.search(r"\{.*\}\s*\Z", s, flags=re.S)
        if m:
            s = m.group(0)

    return json.loads(s)


def to_rows(obj: dict) -> list[dict]:
    row = {"district": obj["district"]}
    for k, v in obj["scores"].items():
        row[k] = v["score"]
        row[f"{k} — why"] = v["justification"]
    return [row]

# ---------- CLI ----------
def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="Score school district masterplan PDFs against a rubric.")
    parser.add_argument("--rubric", required=True, help="Path to rubric file (txt or md).")
    parser.add_argument("--pdfs", nargs="+", required=True, help="One or more PDF paths.")
    parser.add_argument("--district-names", nargs="*", help="Optional names aligned to --pdfs order.")
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--max-pages", type=int, default=None, help="For quick tests, limit pages per PDF.")
    parser.add_argument("--out", default="scores")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set (put it in .env)")

    client = OpenAI(api_key=api_key)
    rubric_text = Path(args.rubric).read_text(encoding="utf-8")

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []

    for i, pdf in enumerate(tqdm(args.pdfs, desc="Scoring PDFs")):
        # Pick district name
        district = (
            args.district_names[i]
            if args.district_names and i < len(args.district_names)
            else Path(pdf).stem.replace("_", " ").title()
        )

        # Read and chunk the plan text (you can keep --max-pages small while testing)
        text = read_pdf_text(Path(pdf), max_pages=args.max_pages)
        chunks = chunk_text(text, target_tokens=1200)
        chunk_block = "\n\n".join([f"--- CHUNK {j+1} ---\n{c}" for j, c in enumerate(chunks)])

        # Build the user prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            rubric=rubric_text,
            district=district,
            chunks=chunk_block
        )

        # ---- call the model
        raw = call_model(client, args.model, SYSTEM_PROMPT, user_prompt)

        # Always save the raw reply so you can inspect when parsing fails
        raw_path = out_dir / f"{district}_raw.txt"
        raw_path.write_text(raw if raw else "<EMPTY_RESPONSE>", encoding="utf-8")

        # Hardened JSON parse (your parse_possible_json should be the safer version)
        try:
            obj = parse_possible_json(raw)
        except Exception as e:
            # Leave the saved raw file in place and surface a helpful error
            print(f"\n[ERROR] Could not parse JSON for {district}. "
                  f"See raw response at: {raw_path}\nDetails: {e}\n")
            raise

        # Write per-district JSON and accumulate rows
        (out_dir / f"{district}.json").write_text(
            json.dumps(obj, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        all_rows.extend(to_rows(obj))


    # aggregate CSV + master JSON
    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "scores.csv", index=False)
    Path(out_dir / "scores.json").write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done. Wrote {out_dir / 'scores.csv'} and per-district JSON files.")
    
if __name__ == "__main__":
    main()
