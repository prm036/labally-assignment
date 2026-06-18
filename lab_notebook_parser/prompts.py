"""
Prompt templates and builders for the V4 lab notebook parser.

Design principles:
  - Every prompt is narrow: one task, one schema
  - Provide explicit symbol guidance (the page's own shorthand)
  - Chain-of-thought experiment reasoning is split into 6 sequential steps
  - Use concrete few-shot examples from the chemistry domain
"""

# ---------------------------------------------------------------------------
# Layout prompt (fallback / hybrid mode)
# ---------------------------------------------------------------------------

LAYOUT_SCHEMA = """\
{
  "regions": [
    {
      "region_id": "r1",
      "type": "heading | text | table | structure_drawing | formula | observation | unknown",
      "bbox": [x1, y1, x2, y2],
      "description": "short description of visible content",
      "confidence": 0.0
    }
  ],
  "notes": []
}"""

LAYOUT_PROMPT = f"""\
You are parsing a scanned handwritten chemistry lab notebook page.

Identify meaningful visual regions on the page.

Return only valid JSON in this schema:

{LAYOUT_SCHEMA}

Rules:
- Use pixel coordinates relative to the provided image.
- Every region_id must be unique: r1, r2, r3, ...
- Prefer larger full-width regions over tiny left-margin strips.
- For each handwritten row, bbox should span the full writing area (not just the margin date).
- Separate molecular drawings, calculations, reaction schemes, and tables when visible.
- Ignore page borders, notebook ruling lines, and dark corner markers.
- Do not invent content. Do not copy schema placeholders.
- Types: "structure_drawing" for ring/bond diagrams; "formula" for equations like Q=I·t
"""

# ---------------------------------------------------------------------------
# Region transcription
# ---------------------------------------------------------------------------

REGION_TRANSCRIPTION_SCHEMA = """\
{
  "lines": [
    {
      "text": "exact handwritten content",
      "normalized_text": "same content with symbols corrected",
      "contains_chemistry": true,
      "uncertain_tokens": ["token1"],
      "confidence": 0.85
    }
  ],
  "notes": []
}"""

REGION_TRANSCRIPTION_PROMPT = f"""\
You are reading one cropped region from a handwritten chemistry lab notebook.

Transcribe ONLY the visible handwritten content. Preserve it exactly.

Return only valid JSON:

{REGION_TRANSCRIPTION_SCHEMA}

Symbol rules — correct these automatically in normalized_text:
  - "22.4C" or "22.4 deg C" → "22.4 °C"
  - "-0.45V" → "-0.45 V"
  - "H2O", "H20" → "H₂O"
  - "Li+" → "Li⁺"
  - "Ag/AgCI" (capital I) → "Ag/AgCl"
  - "mA/cm2", "mA/cm^2" → "mA/cm²"
  - "uL" or "ul" → "μL"
  - "2theta", "2Θ", "2Theta" → "2θ"
  - "->" → "→"
  - "1.5E-4" stays as "1.5E-4" (scientific notation preserved)
  - Preserve: LiTFSI, 12-crown-4, Ag/AgCl, DME, DOL, diglyme, EtOH, TFSI

Critical rules:
  - Do NOT summarise or paraphrase.
  - Do NOT copy schema placeholder text ("transcribed line", "exact line or phrase", etc.).
  - If a token is unclear, add it to uncertain_tokens; still transcribe your best guess.
  - If the crop is blank or unreadable, return: {{"lines": [], "notes": ["unreadable"]}}
  - Do NOT infer formulas from names (e.g. do not write C2F6LiNO4S2 from "LiTFSI").
"""

# ---------------------------------------------------------------------------
# Table transcription
# ---------------------------------------------------------------------------

TABLE_TRANSCRIPTION_SCHEMA = """\
{
  "table": {
    "title": null,
    "columns": ["col1_name", "col2_name"],
    "rows": [
      {"col1_name": "value", "col2_name": "value"}
    ],
    "uncertain_cells": [{"row": 0, "col": "col1_name", "issue": "illegible"}],
    "units": {"col1_name": "°C", "col2_name": "min"},
    "confidence": 0.9
  }
}"""

TABLE_TRANSCRIPTION_PROMPT = f"""\
You are reading one cropped table from a handwritten chemistry lab notebook.

Transcribe the table into structured JSON.

Return only valid JSON:

{TABLE_TRANSCRIPTION_SCHEMA}

Rules:
  - Identify column headers from the first row.
  - Preserve units exactly (°C, min, h, mA, etc.).
  - Put each unit in the "units" dict keyed by column name.
  - Use null for unreadable cells.
  - Do not invent or interpolate missing cells.
  - If uncertain about a cell value, add it to uncertain_cells.
  - Apply the same symbol normalisation as in region transcription (e.g. "30.1C" → "30.1 °C").
"""

# ---------------------------------------------------------------------------
# Chemical structure prompt (for Qwen, used when MolScribe/DECIMER is unavailable)
# ---------------------------------------------------------------------------

STRUCTURE_SCHEMA = """\
{
  "structures": [
    {
      "structure_id": "s1",
      "description": "textual description of the drawing",
      "smiles": null,
      "molecular_formula": null,
      "nearby_labels": ["label text next to or below the structure"],
      "structure_type": "crown_ether | salt | anion | cation | unknown",
      "notes": []
    }
  ]
}"""

STRUCTURE_PROMPT = f"""\
You are analysing a cropped molecular drawing from a handwritten chemistry lab notebook.

Describe what you see and identify the structure if you can.

Return only valid JSON:

{STRUCTURE_SCHEMA}

Rules:
  - Describe the drawing accurately: ring size, bond types, heteroatoms, charge markers.
  - If you can confidently identify a SMILES, provide it — otherwise use null.
  - Do NOT invent SMILES for ambiguous sketches.
  - Capture all nearby text labels (compound names, charge symbols, bracket notations).
  - Example: A ring with 4 oxygens and 4 CH₂ groups = 12-crown-4 = SMILES "C1COCCOCCOCCO1"
  - Example: A Li+ ion inside a crown ether bracket = [Li(12-crown-4)]+
"""

# ---------------------------------------------------------------------------
# Schemas for experiment reasoning steps
# ---------------------------------------------------------------------------

GOAL_SCHEMA = """\
{
  "goal": "exact text of the goal statement",
  "project": "project name or code",
  "date": "date string",
  "evidence_line_ids": ["L1", "L2"]
}"""

MATERIALS_SCHEMA = """\
{
  "materials": [
    {
      "name": "chemical name as written",
      "role": "solute | solvent | electrode | additive | product | reference | other",
      "concentration": "e.g. 1M",
      "volume_or_mass": "e.g. 20 mL",
      "ratio": "e.g. 4:1 v/v",
      "evidence_line_ids": ["L3"]
    }
  ]
}"""

CONDITIONS_SCHEMA = """\
{
  "conditions": [
    {
      "parameter": "e.g. potential",
      "value": "e.g. -0.45 V",
      "reference": "e.g. vs Ag/AgCl",
      "evidence_line_ids": ["L10"]
    }
  ]
}"""

PROCEDURE_SCHEMA = """\
{
  "procedure_steps": [
    {
      "step_number": 1,
      "description": "exact text of step",
      "evidence_line_ids": ["L5", "L6"]
    }
  ]
}"""

RESULTS_SCHEMA = """\
{
  "observations": [
    {
      "text": "exact observed text",
      "measurement_type": "visual | XRD | electrochemical | temperature | other",
      "value": "numeric value if any",
      "unit": "unit if any",
      "evidence_line_ids": ["L20"]
    }
  ],
  "results": [
    {
      "text": "summary of result",
      "evidence_line_ids": ["L21"]
    }
  ]
}"""

CONCLUSION_SCHEMA = """\
{
  "conclusion": {
    "text": null,
    "confidence": 0.0,
    "evidence_line_ids": [],
    "is_inferred": false,
    "removal_reason": null
  },
  "uncertainties": [
    "any ambiguous or unclear items"
  ]
}"""

# ---------------------------------------------------------------------------
# Prompt builders — one per reasoning step
# ---------------------------------------------------------------------------

_SYMBOL_HINT = """\
Symbol key for this notebook page:
  → = reaction arrow
  ⇌ = equilibrium
  °C = degrees Celsius  (may be written as "C" or "deg C")
  μL = microlitre  (may be written as "uL")
  mA/cm² = current density  (may be written as "mA/cm2" or "mA/cm^2")
  2θ = XRD diffraction angle  (may be written as "2theta" or "2Θ")
  Li⁺ = lithium cation
  Ag/AgCl = silver/silver-chloride reference electrode
  RDE = rotating disk electrode
  LiTFSI = lithium bis(trifluoromethanesulfonyl)imide
  12-crown-4 = macrocyclic ether that coordinates Li⁺
"""


def build_goal_prompt(transcript_with_line_ids: str) -> str:
    return f"""\
You are extracting the experimental goal from a handwritten chemistry lab notebook.

{_SYMBOL_HINT}

Return only valid JSON:

{GOAL_SCHEMA}

Critical rules:
  - Quote the goal verbatim from the transcript.
  - Include the project code (e.g. "240604-B") if present.
  - Include the date as written.
  - Cite the exact line IDs as evidence.
  - If no explicit goal is stated, set "goal" to null.

Transcript with line IDs:
{transcript_with_line_ids}
"""


def build_materials_prompt(transcript_with_line_ids: str) -> str:
    return f"""\
You are extracting the materials list from a handwritten chemistry lab notebook.

{_SYMBOL_HINT}

Return only valid JSON:

{MATERIALS_SCHEMA}

Critical rules:
  - List every chemical, electrode, and solvent mentioned.
  - Preserve concentrations, volumes, and ratios exactly as written.
  - Assign roles: "solute" for dissolved salts, "solvent" for liquids,
    "electrode" for working/counter/reference electrodes, "additive" for crown ethers etc.
  - Do NOT invent missing quantities.
  - Cite evidence line IDs for each material.

Transcript with line IDs:
{transcript_with_line_ids}
"""


def build_conditions_prompt(transcript_with_line_ids: str) -> str:
    return f"""\
You are extracting experimental conditions from a handwritten chemistry lab notebook.

{_SYMBOL_HINT}

Return only valid JSON:

{CONDITIONS_SCHEMA}

Critical rules:
  - Include: applied potential, current density, rotation speed, temperature, time, atmosphere.
  - For potential, always include the reference electrode if stated.
  - Do NOT invent conditions not mentioned in the transcript.
  - Cite evidence line IDs.

Transcript with line IDs:
{transcript_with_line_ids}
"""


def build_procedure_prompt(transcript_with_line_ids: str) -> str:
    return f"""\
You are reconstructing the experimental procedure from a handwritten chemistry lab notebook.

{_SYMBOL_HINT}

Return only valid JSON:

{PROCEDURE_SCHEMA}

Critical rules:
  - Order steps chronologically (by line ID order if unclear).
  - Quote each step description verbatim from the transcript.
  - Each step must have at least one evidence_line_id.
  - Separate "prepare electrolyte", "assemble cell", "run deposition", "run characterisation" etc. into distinct steps.

Transcript with line IDs:
{transcript_with_line_ids}
"""


def build_results_prompt(transcript_with_line_ids: str) -> str:
    return f"""\
You are extracting experimental observations and results from a handwritten chemistry lab notebook.

{_SYMBOL_HINT}

Return only valid JSON:

{RESULTS_SCHEMA}

Critical rules:
  - Include ALL explicit observations: visual, electrochemical, diffraction, thermal.
  - For XRD: capture peak positions (2θ values) and intensities exactly.
  - For temperature data: extract the full time-vs-temperature table if present.
  - Do NOT infer results not explicitly stated.
  - Cite evidence line IDs for each observation.

Transcript with line IDs:
{transcript_with_line_ids}
"""


def build_conclusion_prompt(transcript_with_line_ids: str,
                             observations_summary: str) -> str:
    return f"""\
You are determining the conclusion of an experiment from a handwritten chemistry lab notebook.

{_SYMBOL_HINT}

Observations already extracted:
{observations_summary}

Return only valid JSON:

{CONCLUSION_SCHEMA}

Critical rules:
  - A conclusion is only valid if explicitly stated in the transcript OR directly supported
    by the extracted observations.
  - If the page records data but no explicit conclusion, set "text" to null.
  - If you infer a conclusion from observations, set "is_inferred" to true and
    "confidence" to at most 0.6.
  - If no supporting evidence_line_ids can be cited, set "text" to null and
    populate "removal_reason".

Transcript with line IDs:
{transcript_with_line_ids}
"""


# ---------------------------------------------------------------------------
# Chemistry extraction from merged transcript (used in all backends)
# ---------------------------------------------------------------------------

CHEMISTRY_SCHEMA = """\
{
  "chemical_entities": [
    {
      "raw_text": "as written in transcript",
      "normalized_name": "canonical chemical name",
      "role": "solute | solvent | electrode | additive | product | other",
      "formula": null,
      "formula_source": null,
      "concentration": null,
      "volume_or_mass": null,
      "evidence": "exact phrase from transcript",
      "notes": []
    }
  ],
  "conditions": [],
  "observations": [],
  "ambiguities": []
}"""


def build_chemistry_from_transcript_prompt(transcript: str) -> str:
    return f"""\
You are extracting chemistry information from a handwritten chemistry lab notebook transcription.

{_SYMBOL_HINT}

Return only valid JSON:

{CHEMISTRY_SCHEMA}

Critical rules:
  - Do NOT invent chemicals not mentioned in the transcript.
  - Do NOT infer molecular formulas from chemical names
    (e.g. do not write "C₂F₆LiNO₄S₂" from "LiTFSI" unless the formula is written).
  - formula_source must be "explicit_page_text" (formula appears verbatim) or null.
  - Use "evidence" to quote the exact phrase containing this chemical.
  - Identify the role of each chemical in the experiment.
  - Capture concentrations, volumes, and ratios from the transcript exactly.

Transcript:
{transcript}
"""

# ---------------------------------------------------------------------------
# Legacy compat: full experiment schema (used in qwen_only backend)
# ---------------------------------------------------------------------------

EXPERIMENT_SCHEMA = """\
{
  "experiment": {
    "project": null,
    "date": null,
    "goal": null,
    "experiment_type": null,
    "materials": [],
    "procedure": [],
    "reaction_or_process": {},
    "observations": [],
    "results": [],
    "conclusion": {
      "text": null,
      "evidence_line_ids": [],
      "confidence": 0.0
    }
  },
  "uncertainties": []
}"""


def build_experiment_from_transcript_prompt(transcript_with_line_ids: str) -> str:
    return f"""\
You are interpreting a handwritten chemistry lab notebook transcription.

{_SYMBOL_HINT}

Return only valid JSON:

{EXPERIMENT_SCHEMA}

Critical rules:
  - Do NOT hallucinate missing product, yield, result, or conclusion.
  - A conclusion is allowed only if explicitly supported by one or more line IDs.
  - If the page records observations but not a final decision, set conclusion.text to null.
  - Every procedure step should include evidence_line_ids.
  - Capture all XRD peak positions, temperatures, and visual observations exactly.

Transcript with line IDs:
{transcript_with_line_ids}
"""
