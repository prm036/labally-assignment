#!/usr/bin/env python3
"""
tests/validate_output.py

Validates the JSON output of the V4 lab notebook parser against known
ground-truth content from the example page (Page 57).

Usage:
    python tests/validate_output.py lab_parser_outputs_v4/Example_lab_notebook_page_v4_full.json
    python tests/validate_output.py path/to/output.json --verbose

Exit code 0 = all checks pass; 1 = failures found.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Ground truth for Page 57
# ---------------------------------------------------------------------------

# Level 1 — text phrases that MUST appear in the merged transcript
REQUIRED_TEXT_PHRASES = [
    "Page 57",
    "Li electrodeposition",
    "glyme",
    "240604",
    "Goal",
    "Screen electrolyte",
    "stable",
    "plating",
    "June 4",
    "LiTFSI",
    "20 mL",
    "12-crown-4",
    "glovebox",
    "glassy",
    "RDE",
    "Ag/AgCl",
    "Li foil",
    "Deposition",
    "240604-B",
    "Film",
    "XRD",
]

# Level 2 — special symbols and notations
REQUIRED_SYMBOLS = [
    "°C",      # temperature
    "→",       # reaction arrow (or "->" acceptable)
    "2θ",      # XRD angle — must be present (not "2theta")
    "mA/cm²",  # current density
    "Li⁺",     # cation notation
    "H₂O",     # water formula subscript
]

# Level 2 — critical quantities (value + unit pairs)
REQUIRED_QUANTITIES = [
    # (value, unit)
    (1.0,   "M"),      # 1M LiTFSI
    (20.0,  "mL"),     # 20 mL total
    (0.5,   "mA/cm²"), # current density (or mA cm-2)
    (0.3,   "cm²"),    # electrode area
    (90.0,  "min"),    # deposition time
    (1600.0,"rpm"),    # rotation speed
    (22.4,  "°C"),     # initial temperature
    (30.0,  "°C"),     # target temperature (approx)
]

# Level 2 — XRD peak positions that must be extracted
REQUIRED_XRD_VALUES = [
    "2.1",  # XRD peak at 2θ = 2.1°
    "4.7",  # shoulder at 2θ = 4.7°
]

# Level 3 — chemicals that must appear in chemistry entities
REQUIRED_CHEMICALS = [
    "LiTFSI",
    "Ag/AgCl",
    "12-crown-4",
    "glyme",
    "EtOH",
    "Li",
]

# Level 3 — forbidden hallucinated chemicals (should NOT appear)
FORBIDDEN_HALLUCINATIONS = [
    "NaOH", "KOH", "H2SO4", "HCl", "NaCl",
    "CuSO4", "FeCl3", "silver nitrate", "AgNO3",
]

# Level 4 — experiment fields
REQUIRED_EXPERIMENT_KEYS = [
    "goal", "materials", "conditions", "procedure", "results",
]

# Applied potential
REQUIRED_POTENTIAL = "-0.45"  # V vs Ag/AgCl


# ---------------------------------------------------------------------------
# Checker helpers
# ---------------------------------------------------------------------------

CheckResult = Tuple[bool, str]


def check_level1_text(data: dict, verbose: bool) -> List[CheckResult]:
    transcript = data.get("merged_transcript", "") or ""
    results = []
    for phrase in REQUIRED_TEXT_PHRASES:
        found = phrase.lower() in transcript.lower()
        msg = f"L1-text | '{phrase}' {'FOUND' if found else 'MISSING'} in merged_transcript"
        results.append((found, msg))
    return results


def check_level2_symbols(data: dict, verbose: bool) -> List[CheckResult]:
    transcript = data.get("merged_transcript", "") or ""
    results = []

    for sym in REQUIRED_SYMBOLS:
        # Allow loose equivalents for arrows
        if sym == "→":
            found = "→" in transcript or "->" in transcript
        else:
            found = sym in transcript
        msg = f"L2-symbol | '{sym}' {'FOUND' if found else 'MISSING'}"
        results.append((found, msg))

    # XRD values
    for val in REQUIRED_XRD_VALUES:
        found = val in transcript
        msg = f"L2-xrd | 2θ value '{val}°' {'FOUND' if found else 'MISSING'} in transcript"
        results.append((found, msg))

    # Quantities
    quantities = data.get("deterministic_quantities", []) or []
    for (value, unit) in REQUIRED_QUANTITIES:
        matched = any(
            q.get("unit", "").replace("cm2", "cm²").replace("cm^2", "cm²") == unit and
            abs((q.get("value") or -9999) - value) < 0.1
            for q in quantities
        )
        msg = f"L2-quantity | {value} {unit} {'FOUND' if matched else 'MISSING'} in extracted quantities"
        results.append((matched, msg))

    return results


def check_level3_chemistry(data: dict, verbose: bool) -> List[CheckResult]:
    chem = data.get("vlm_chemistry", {}) or {}
    entities = chem.get("chemical_entities", []) or []
    ner = chem.get("ner_entities", []) or []
    structures = data.get("structures", {}).get("structures", []) or []

    # Build a flat list of all chemical text
    all_chem_text = " ".join([
        str(e.get("raw_text", "")) + " " + str(e.get("normalized_name", ""))
        for e in entities + ner
    ]).lower()
    # Also include transcript
    all_chem_text += " " + (data.get("merged_transcript", "") or "").lower()

    results = []

    for chem_name in REQUIRED_CHEMICALS:
        found = chem_name.lower() in all_chem_text
        msg = f"L3-chem | '{chem_name}' {'FOUND' if found else 'MISSING'} in chemistry entities"
        results.append((found, msg))

    for bad in FORBIDDEN_HALLUCINATIONS:
        found_hall = bad.lower() in all_chem_text
        msg = f"L3-hallucination | '{bad}' {'BAD - present in output' if found_hall else 'OK - not present'}"
        results.append((not found_hall, msg))

    # At least one valid SMILES structure
    valid_smiles = [s for s in structures if s.get("is_valid_smiles") or s.get("smiles")]
    msg = f"L3-structure | Valid SMILES found: {len(valid_smiles)} structures"
    results.append((len(valid_smiles) > 0, msg))

    # Applied potential check in transcript or conditions
    transcript = (data.get("merged_transcript", "") or "").lower()
    potential_found = REQUIRED_POTENTIAL in transcript
    if not potential_found:
        exp = data.get("experiment_summary", {}) or {}
        conditions = exp.get("conditions", {}).get("conditions", []) or []
        for c in conditions:
            if REQUIRED_POTENTIAL in str(c.get("value", "")):
                potential_found = True
                break
    msg = f"L3-condition | Applied potential {REQUIRED_POTENTIAL} V vs Ag/AgCl {'FOUND' if potential_found else 'MISSING'}"
    results.append((potential_found, msg))

    return results


def check_level4_experiment(data: dict, verbose: bool) -> List[CheckResult]:
    exp = data.get("experiment_summary", {}) or {}
    results = []

    for key in REQUIRED_EXPERIMENT_KEYS:
        # Both V4 structured (dict of keys) and V3 (nested under "experiment") accepted
        v4_val = exp.get(key)
        v3_val = (exp.get("experiment") or {}).get(key)
        found = bool(v4_val or v3_val)
        msg = f"L4-experiment | Field '{key}' {'PRESENT' if found else 'MISSING'} in experiment_summary"
        results.append((found, msg))

    # Goal should mention electrolyte screening
    goal_text = ""
    if exp.get("goal"):
        g = exp["goal"]
        goal_text = str(g.get("goal", "")) if isinstance(g, dict) else str(g)
    elif (exp.get("experiment") or {}).get("goal"):
        goal_text = str(exp["experiment"]["goal"])

    goal_keywords = ["screen", "electrolyte", "Li", "plating", "stable"]
    goal_ok = any(kw.lower() in goal_text.lower() for kw in goal_keywords)
    msg = f"L4-experiment | Goal contains experiment keywords: {'YES' if goal_ok else 'NO'} (goal: {goal_text[:100]!r})"
    results.append((goal_ok, msg))

    # Film observation mentioned in results
    obs_text = ""
    for key in ("results", "observations"):
        val = exp.get(key)
        if isinstance(val, dict):
            obs = val.get("observations", []) + val.get("results", [])
            obs_text += " ".join(str(o.get("text", "")) for o in obs)
        elif isinstance(val, list):
            obs_text += " ".join(str(o) for o in val)

    film_obs = "film" in obs_text.lower() or "grey" in obs_text.lower() or "dull" in obs_text.lower()
    msg = f"L4-experiment | Visual observation (film appearance) {'FOUND' if film_obs else 'MISSING'}"
    results.append((film_obs, msg))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate V4 lab notebook parser output")
    parser.add_argument("output_json", help="Path to parser output JSON file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print all results (not just failures)")
    args = parser.parse_args()

    path = Path(args.output_json)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    all_results: List[CheckResult] = []
    checks = [
        ("Level 1 — Plain Text",         check_level1_text(data, args.verbose)),
        ("Level 2 — Special Symbols",    check_level2_symbols(data, args.verbose)),
        ("Level 3 — Chemistry",          check_level3_chemistry(data, args.verbose)),
        ("Level 4 — Experiment",         check_level4_experiment(data, args.verbose)),
    ]

    total_pass = total_fail = 0
    for level_name, results in checks:
        passed = sum(1 for ok, _ in results if ok)
        failed = len(results) - passed
        total_pass += passed
        total_fail += failed
        print(f"\n{'─'*55}")
        print(f"{level_name}: {passed}/{len(results)} passed")
        print(f"{'─'*55}")
        for ok, msg in results:
            if not ok or args.verbose:
                status = "✓" if ok else "✗"
                print(f"  {status} {msg}")
        all_results.extend(results)

    print(f"\n{'='*55}")
    print(f"TOTAL: {total_pass}/{total_pass + total_fail} checks passed")
    print(f"{'='*55}")

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    main()
