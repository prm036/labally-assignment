"""
Stage 2: Hierarchical transcription with model ensemble.

  2a. Text/special symbols — Qwen2.5-VL + TrOCR ensemble with token voting
  2b. Tables — structured cell-by-cell extraction
  2c. Mathematical formulas — Pix2Tex LaTeX OCR (optional)

The ensemble strategy:
  - Run both Qwen and TrOCR on each text crop
  - If they agree (edit-distance ≤ 2 chars on short tokens), use the agreed text
  - Where they disagree, flag uncertain tokens; prefer Qwen for chemistry abbreviations
    (it understands context), TrOCR for numeric values (it's more precise)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .utils import crop_image_region, ensure_json_response
from .preprocess import upscale_region, adaptive_binarize
from .prompts import (
    REGION_TRANSCRIPTION_PROMPT, REGION_TRANSCRIPTION_SCHEMA,
    TABLE_TRANSCRIPTION_PROMPT, TABLE_TRANSCRIPTION_SCHEMA,
)
from .extraction_rules import normalize_scientific_text


# ---------------------------------------------------------------------------
# Placeholder / quality filters
# ---------------------------------------------------------------------------

BAD_PLACEHOLDER_PHRASES = {
    "transcribed line", "same line with obvious symbols normalized",
    "step description", "exact line or phrase", "as written",
    "short description", "visible content", "text", "normalized_text",
    "exact handwritten content", "same content with symbols corrected",
    "col1_name", "col2_name", "value",
}


def is_bad_placeholder_line(text: Optional[str]) -> bool:
    if text is None:
        return True
    t = str(text).strip().lower()
    if not t:
        return True
    if t in BAD_PLACEHOLDER_PHRASES:
        return True
    if "same line with obvious" in t:
        return True
    if "same content with" in t:
        return True
    return False


def description_looks_like_content(description: Optional[str]) -> bool:
    if description is None:
        return False
    d = str(description).strip()
    if is_bad_placeholder_line(d):
        return False
    useful_patterns = [
        r"\d", r"°C", r"Li", r"Ag/AgCl", r"Goal", r"Electrolyte",
        r"Deposition", r"Apply", r"mA", r"ppm", r"crown", r"glovebox",
        r"temperature", r"XRD", r"Film", r"TFSI", r"glyme", r"EtOH",
        r"diglyme", r"LiTFSI",
    ]
    return any(re.search(p, d, flags=re.IGNORECASE) for p in useful_patterns)


# ---------------------------------------------------------------------------
# Ensemble merge
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance (no weights)."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    dp = list(range(lb + 1))
    for i in range(1, la + 1):
        new_dp = [i]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            new_dp.append(min(new_dp[j - 1] + 1, dp[j] + 1, dp[j - 1] + cost))
        dp = new_dp
    return dp[lb]


def _merge_two_transcriptions(qwen_text: str,
                               trocr_text: str,
                               prefer_qwen_pattern: str = r"[A-Z][a-z]{1,}[A-Z0-9]") -> Tuple[str, List[str]]:
    """
    Merge Qwen and TrOCR outputs at the token level.

    Rules:
      1. Split both outputs into whitespace tokens.
      2. For each aligned token pair:
         - If identical → accept.
         - If edit distance ≤ 2 and short → accept the Qwen token.
         - If they disagree on a numeric/unit-like token → prefer TrOCR.
         - If they disagree on a chemistry abbreviation → prefer Qwen.
         - Otherwise → flag as uncertain (keep Qwen token).

    Returns (merged_text, uncertain_tokens).
    """
    q_tokens = qwen_text.split()
    t_tokens = trocr_text.split()

    # Align with simple zip (pad shorter)
    max_len = max(len(q_tokens), len(t_tokens))
    q_tokens += [""] * (max_len - len(q_tokens))
    t_tokens += [""] * (max_len - len(t_tokens))

    merged: List[str] = []
    uncertain: List[str] = []

    num_unit_pattern = re.compile(r"^\d+\.?\d*\s*[a-zA-Z°μ%/]+$")
    chem_abbr_pattern = re.compile(prefer_qwen_pattern)

    for qt, tt in zip(q_tokens, t_tokens):
        if not qt and not tt:
            continue
        if qt == tt:
            merged.append(qt)
        elif not tt:
            merged.append(qt)
        elif not qt:
            merged.append(tt)
        else:
            dist = _edit_distance(qt, tt)
            if dist <= 2:
                # Close enough — prefer Qwen (better chemical context)
                merged.append(qt)
            elif num_unit_pattern.match(tt):
                # TrOCR is more reliable for numeric values
                merged.append(tt)
            elif chem_abbr_pattern.search(qt):
                # Qwen better for chemistry abbreviations
                merged.append(qt)
            else:
                # Genuine disagreement
                merged.append(qt)
                uncertain.append(f"{qt}|{tt}")

    return " ".join(merged), uncertain


def transcribe_with_ensemble(qwen,
                              trocr,
                              crop_path: str,
                              qwen_prompt: str,
                              region_type: str = "text",
                              allow_repair: bool = True) -> Dict[str, Any]:
    """
    Transcribe a region crop using Qwen (primary) + TrOCR (secondary).
    Returns a parsed transcription dict with ensemble-merged lines.
    """
    # Qwen pass
    qwen_schema = REGION_TRANSCRIPTION_SCHEMA
    qwen_raw = qwen.ask_image(crop_path, qwen_prompt, max_new_tokens=1024)
    qwen_parsed = ensure_json_response(qwen, qwen_raw, schema_hint=qwen_schema, allow_repair=allow_repair)

    if trocr is None:
        qwen_parsed["_source"] = "qwen_only"
        return qwen_parsed

    # TrOCR pass — use binarized crop for better contrast
    try:
        img_bgr = cv2.imread(crop_path)
        if img_bgr is not None:
            bin_img = adaptive_binarize(img_bgr)
            # Convert binary to 3-channel for TrOCR (expects RGB)
            bin_rgb = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2RGB)
            trocr_text = trocr.transcribe_pil(Image.fromarray(bin_rgb))
        else:
            trocr_text = trocr.transcribe(crop_path)
    except Exception as e:
        trocr_text = ""

    if not trocr_text.strip():
        qwen_parsed["_source"] = "qwen_only_trocr_empty"
        return qwen_parsed

    # Merge each qwen line with the trocr output
    qwen_lines = qwen_parsed.get("lines", [])
    if not qwen_lines:
        # If Qwen gave nothing, use TrOCR as a single line
        return {
            "lines": [{
                "text": trocr_text,
                "normalized_text": normalize_scientific_text(trocr_text),
                "contains_chemistry": bool(re.search(r"Li|Ag|H₂O|mA|M\b|ppm|TFSI|glyme|crown", trocr_text, re.I)),
                "uncertain_tokens": [],
                "confidence": 0.7,
            }],
            "notes": ["qwen_empty_trocr_fallback"],
            "_source": "trocr_fallback",
        }

    merged_lines = []
    for line in qwen_lines:
        qwen_text = line.get("normalized_text") or line.get("text") or ""
        merged_text, uncertain = _merge_two_transcriptions(qwen_text, trocr_text)
        merged_text = normalize_scientific_text(merged_text)
        merged_lines.append({
            "text": line.get("text", ""),
            "normalized_text": merged_text,
            "contains_chemistry": line.get("contains_chemistry"),
            "uncertain_tokens": list(set(line.get("uncertain_tokens", []) + uncertain)),
            "confidence": line.get("confidence", 0.75),
        })

    return {
        "lines": merged_lines,
        "notes": qwen_parsed.get("notes", []) + ["ensemble_qwen_trocr"],
        "_source": "ensemble",
    }


# ---------------------------------------------------------------------------
# Fallback from region description
# ---------------------------------------------------------------------------

def make_fallback_line_from_region_description(region: Dict[str, Any],
                                                crop_path: str) -> Optional[Dict[str, Any]]:
    desc = region.get("description")
    if not description_looks_like_content(desc):
        return None

    norm = normalize_scientific_text(desc)
    return {
        "line_id": f"{region.get('region_id')}_desc",
        "region_id": region.get("region_id"),
        "region_type": region.get("type"),
        "text": desc,
        "normalized_text": norm,
        "contains_chemistry": bool(re.search(
            r"Li|Ag/AgCl|H₂O|mA|M\b|ppm|crown|EtOH|glyme|TFSI|LiTFSI",
            norm, flags=re.IGNORECASE
        )),
        "uncertain_tokens": [],
        "confidence": min(float(region.get("confidence", 0.6) or 0.6), 0.75),
        "crop_path": crop_path,
        "source": "layout_description_fallback",
    }


# ---------------------------------------------------------------------------
# Pix2Tex formula extraction (optional)
# ---------------------------------------------------------------------------

def extract_latex_from_formula_region(crop_path: str) -> Optional[str]:
    """
    Try to extract LaTeX from a formula/equation region using pix2tex.
    Returns LaTeX string or None if unavailable or failed.
    """
    try:
        from pix2tex.cli import LatexOCR
    except ImportError:
        return None

    try:
        model = LatexOCR()
        img = Image.open(crop_path)
        latex = model(img)
        return latex
    except Exception as e:
        return None


# ---------------------------------------------------------------------------
# Main region transcription orchestrator
# ---------------------------------------------------------------------------

def transcribe_regions_with_vlm(qwen,
                                 image_path_vlm: str,
                                 layout_json: Dict[str, Any],
                                 output_dir: str,
                                 allow_repair: bool = True,
                                 trocr=None) -> Dict[str, Any]:
    """
    Transcribe all layout regions.
    Uses ensemble (Qwen + TrOCR) if trocr is not None.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_lines: List[Dict[str, Any]] = []
    tables: List[Dict[str, Any]] = []
    formulas: List[Dict[str, Any]] = []
    raw_region_outputs: List[Dict[str, Any]] = []
    crop_records: List[Dict[str, Any]] = []

    regions = layout_json.get("regions", [])

    for idx, region in enumerate(regions, start=1):
        region_id = region.get("region_id") or f"r{idx}"
        region_type = region.get("type", "unknown")
        bbox = region.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        safe_rid = str(region_id).replace("/", "_")
        safe_rtype = str(region_type).replace("/", "_")
        crop_path = output_dir / f"{idx:03d}_{safe_rid}_{safe_rtype}.png"
        crop_image_region(image_path_vlm, bbox, crop_path, padding=10)

        crop_records.append({
            "region_id": region_id,
            "region_type": region_type,
            "bbox": bbox,
            "crop_path": str(crop_path),
            "description": region.get("description", ""),
        })

        print(f"  Transcribing {region_id} ({region_type})...")

        # ---- Table regions ----
        if region_type == "table":
            raw = qwen.ask_image(str(crop_path), TABLE_TRANSCRIPTION_PROMPT, max_new_tokens=1024)
            parsed = ensure_json_response(qwen, raw, schema_hint=TABLE_TRANSCRIPTION_SCHEMA, allow_repair=allow_repair)
            raw_region_outputs.append({
                "region_id": region_id, "region_type": region_type,
                "bbox": bbox, "raw_response": raw, "parsed": parsed,
            })
            table = parsed.get("table") if isinstance(parsed, dict) else None
            if table:
                table["region_id"] = region_id
                table["crop_path"] = str(crop_path)
                tables.append(table)
            continue

        # ---- Formula/equation regions ----
        if region_type == "formula":
            latex = extract_latex_from_formula_region(str(crop_path))
            if latex:
                formulas.append({
                    "region_id": region_id,
                    "latex": latex,
                    "crop_path": str(crop_path),
                    "source": "pix2tex",
                })
                # Also add as a text line for the transcript
                all_lines.append({
                    "line_id": f"{region_id}_l1",
                    "region_id": region_id,
                    "region_type": "formula",
                    "text": latex,
                    "normalized_text": normalize_scientific_text(latex),
                    "contains_chemistry": True,
                    "uncertain_tokens": [],
                    "confidence": 0.75,
                    "crop_path": str(crop_path),
                    "source": "pix2tex",
                })
            # Also run Qwen on formula regions as text
            # (fall through to text processing below)

        # ---- Text/heading/observation/formula (Qwen ± TrOCR) ----
        if region_type not in {"structure_drawing", "molecular_structure", "reaction_scheme"}:
            parsed = transcribe_with_ensemble(
                qwen, trocr, str(crop_path),
                qwen_prompt=REGION_TRANSCRIPTION_PROMPT,
                region_type=region_type,
                allow_repair=allow_repair,
            )

            raw_region_outputs.append({
                "region_id": region_id,
                "region_type": region_type,
                "bbox": bbox,
                "description": region.get("description", ""),
                "crop_path": str(crop_path),
                "parsed": parsed,
            })

            good_lines_for_region: List[Dict[str, Any]] = []

            if not parsed.get("parse_error"):
                for i, line in enumerate(parsed.get("lines", [])):
                    text = line.get("text") or ""
                    norm = line.get("normalized_text") or text
                    norm = normalize_scientific_text(norm)

                    if is_bad_placeholder_line(text) or is_bad_placeholder_line(norm):
                        continue

                    good_lines_for_region.append({
                        "line_id": f"{region_id}_l{i+1}",
                        "region_id": region_id,
                        "region_type": region_type,
                        "text": text,
                        "normalized_text": norm,
                        "contains_chemistry": line.get("contains_chemistry"),
                        "uncertain_tokens": line.get("uncertain_tokens", []),
                        "confidence": line.get("confidence"),
                        "crop_path": str(crop_path),
                        "source": parsed.get("_source", "qwen"),
                    })

            if not good_lines_for_region:
                fallback = make_fallback_line_from_region_description(region, str(crop_path))
                if fallback:
                    good_lines_for_region.append(fallback)

            all_lines.extend(good_lines_for_region)

    return {
        "lines": all_lines,
        "tables": tables,
        "formulas": formulas,
        "raw_region_outputs": raw_region_outputs,
        "crop_records": crop_records,
    }


# ---------------------------------------------------------------------------
# Build merged transcript
# ---------------------------------------------------------------------------

def build_transcript_blocks(lines: List[Dict[str, Any]]) -> Tuple[str, str]:
    """
    Build two representations from transcribed lines:
      1. Plain merged text (for NER/chemistry extraction)
      2. Line-ID-annotated text (for evidence-cited experiment reasoning)
    """
    plain_parts: List[str] = []
    line_id_parts: List[str] = []

    for i, line in enumerate(lines, start=1):
        lid = line.get("line_id") or f"line_{i}"
        text = line.get("normalized_text") or line.get("text") or ""
        text = normalize_scientific_text(text)

        if is_bad_placeholder_line(text):
            continue

        plain_parts.append(text)
        line_id_parts.append(f"{lid}: {text}")

    return "\n".join(plain_parts), "\n".join(line_id_parts)
