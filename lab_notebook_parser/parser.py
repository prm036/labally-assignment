"""
V4 Lab Notebook Parser — main orchestration.

Three backends selectable via --backend flag:
  qwen_only  : V3-compatible, Qwen only (no TrOCR / MolScribe / DECIMER)
  ensemble   : Qwen + TrOCR text ensemble, still no structure models
  full       : complete V4 pipeline (all models, PubChem, MolScribe/DECIMER)

The parse() method returns a single JSON-serialisable dict with all extracted
information across all four evaluation levels:
  Level 1 — plain text
  Level 2 — special symbols
  Level 3 — chemistry (entities, formulas, structures with SMILES)
  Level 4 — experiment (goal, materials, conditions, procedure, results, conclusion)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, Optional

from .utils import save_json, ensure_json_response, crop_image_region
from .preprocess import preprocess_for_vlm
from .layout_detector import (
    detect_layout_from_ruled_lines,
    merge_opencv_and_vlm_regions,
    draw_layout_regions,
    LayoutRegion,
)
from .transcription import (
    transcribe_regions_with_vlm,
    build_transcript_blocks,
)
from .extraction_rules import (
    normalize_scientific_text,
    extract_quantities,
    extract_explicit_formula_mentions,
    extract_ratios,
    run_chemistry_ner,
    resolve_chemical_with_pubchem,
    ChemEntity,
)
from .prompts import (
    LAYOUT_PROMPT, LAYOUT_SCHEMA,
    STRUCTURE_PROMPT, STRUCTURE_SCHEMA,
    CHEMISTRY_SCHEMA,
    build_chemistry_from_transcript_prompt,
    build_goal_prompt,
    build_materials_prompt,
    build_conditions_prompt,
    build_procedure_prompt,
    build_results_prompt,
    build_conclusion_prompt,
    build_experiment_from_transcript_prompt,
    EXPERIMENT_SCHEMA,
)

import cv2


# ---------------------------------------------------------------------------
# Structure interpretation (Qwen fallback, or MolScribe/DECIMER when available)
# ---------------------------------------------------------------------------

def interpret_structures(qwen,
                         image_path_vlm: str,
                         layout_regions,
                         output_dir: Path,
                         allow_repair: bool = True,
                         structure_recognizer=None) -> Dict[str, Any]:
    """
    For each structure_drawing / reaction_scheme / molecular_structure region:
      1. Crop and (optionally) upscale the region
      2. Run MolScribe/DECIMER if available → validated SMILES
      3. Run Qwen STRUCTURE_PROMPT for textual description + nearby labels
      4. Merge results

    Returns {"structures": [...], "raw_structure_outputs": [...]}
    """
    from .preprocess import upscale_region
    import cv2

    output_dir.mkdir(parents=True, exist_ok=True)
    all_structures = []
    raw_outputs = []

    for idx, region in enumerate(layout_regions, start=1):
        if hasattr(region, 'to_dict'):
            reg_dict = region.to_dict()
        else:
            reg_dict = dict(region)

        rtype = reg_dict.get("type", "unknown")
        if rtype not in {"structure_drawing", "molecular_structure", "reaction_scheme"}:
            continue

        bbox = reg_dict.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        region_id = reg_dict.get("region_id", f"s{idx}")
        safe_rid = str(region_id).replace("/", "_")
        safe_rtype = str(rtype).replace("/", "_")

        # Crop at 2× resolution for structure models
        crop_path = output_dir / f"{idx:03d}_{safe_rid}_{safe_rtype}.png"
        crop_image_region(image_path_vlm, bbox, crop_path, padding=15)

        # Upscale for clearer recognition
        img_bgr = cv2.imread(str(crop_path))
        if img_bgr is not None:
            upscaled = upscale_region(img_bgr, scale=2.0)
            upscaled_path = output_dir / f"{idx:03d}_{safe_rid}_{safe_rtype}_2x.png"
            cv2.imwrite(str(upscaled_path), upscaled)
        else:
            upscaled_path = crop_path

        print(f"  Interpreting structure in {region_id} ({rtype})...")

        # ---- MolScribe / DECIMER pass ----
        structure_result = None
        if structure_recognizer is not None:
            nearby_labels = [reg_dict.get("description", "")]
            sr = structure_recognizer.recognize(
                str(upscaled_path),
                region_id=region_id,
                nearby_labels=nearby_labels,
            )
            structure_result = {
                "backend": sr.backend,
                "smiles": sr.smiles,
                "is_valid_smiles": sr.is_valid_smiles,
                "canonical_smiles": sr.canonical_smiles,
                "molecular_formula": sr.molecular_formula,
                "iupac_name": sr.iupac_name,
                "pubchem_cid": sr.pubchem_cid,
                "confidence": sr.confidence,
                "notes": sr.notes,
            }

        # ---- Qwen description pass ----
        qwen_raw = qwen.ask_image(str(upscaled_path), STRUCTURE_PROMPT, max_new_tokens=1024)
        qwen_parsed = ensure_json_response(qwen, qwen_raw, schema_hint=STRUCTURE_SCHEMA, allow_repair=allow_repair)

        raw_outputs.append({
            "region_id": region_id,
            "region_type": rtype,
            "crop_path": str(crop_path),
            "upscaled_crop_path": str(upscaled_path),
            "structure_model_result": structure_result,
            "qwen_raw_response": qwen_raw,
            "qwen_parsed": qwen_parsed,
        })

        if qwen_parsed.get("parse_error"):
            continue

        for i, s in enumerate(qwen_parsed.get("structures", [])):
            s = dict(s)
            s["structure_id"] = s.get("structure_id") or f"{region_id}_s{i+1}"
            s["region_id"] = region_id
            s["region_type"] = rtype
            s["crop_path"] = str(crop_path)
            # Merge in structure model result
            if structure_result and structure_result.get("is_valid_smiles"):
                if not s.get("smiles"):
                    s["smiles"] = structure_result["smiles"]
                s["canonical_smiles"] = structure_result.get("canonical_smiles")
                s["molecular_formula"] = structure_result.get("molecular_formula")
                s["iupac_name"] = structure_result.get("iupac_name")
                s["pubchem_cid"] = structure_result.get("pubchem_cid")
                s["structure_model_backend"] = structure_result.get("backend")
                s["structure_model_confidence"] = structure_result.get("confidence")
            all_structures.append(s)

        # If Qwen returned no structures but model did, add a minimal record
        if not qwen_parsed.get("structures") and structure_result and structure_result.get("is_valid_smiles"):
            all_structures.append({
                "structure_id": f"{region_id}_s1",
                "region_id": region_id,
                "region_type": rtype,
                "crop_path": str(crop_path),
                "description": reg_dict.get("description", ""),
                "smiles": structure_result["smiles"],
                "canonical_smiles": structure_result.get("canonical_smiles"),
                "molecular_formula": structure_result.get("molecular_formula"),
                "iupac_name": structure_result.get("iupac_name"),
                "pubchem_cid": structure_result.get("pubchem_cid"),
                "structure_model_backend": structure_result.get("backend"),
                "structure_model_confidence": structure_result.get("confidence"),
            })

    return {"structures": all_structures, "raw_structure_outputs": raw_outputs}


# ---------------------------------------------------------------------------
# Chemistry sanitisation
# ---------------------------------------------------------------------------

def sanitize_chemistry(chemistry_json: Dict[str, Any],
                       explicit_formulas,
                       enable_pubchem: bool) -> Dict[str, Any]:
    chemistry = dict(chemistry_json)
    entities = chemistry.get("chemical_entities", [])
    explicit_formula_texts = {f.formula for f in explicit_formulas}
    cleaned = []

    for ent in entities:
        if isinstance(ent, str):
            ent = {"raw_text": ent, "normalized_name": ent}
        elif isinstance(ent, dict):
            ent = dict(ent)
        else:
            continue

        formula = ent.get("formula")
        formula_source = ent.get("formula_source")

        if formula:
            if formula_source == "explicit_page_text" and formula in explicit_formula_texts:
                pass  # Valid — explicitly written on page
            elif formula_source == "structure_model":
                pass  # Valid — from MolScribe/DECIMER with RDKit validation
            elif formula_source == "pubchem_lookup":
                pass  # Valid — externally verified
            else:
                ent.setdefault("notes", [])
                ent["notes"].append("Formula removed: not explicitly supported by page text or verified model.")
                ent["formula"] = None
                ent["formula_source"] = None

        if enable_pubchem:
            name = ent.get("normalized_name") or ent.get("raw_text")
            resolver = resolve_chemical_with_pubchem(name)
            ent["resolver_result"] = resolver
            if resolver.get("resolved") and not ent.get("formula"):
                ent["formula"] = resolver.get("molecular_formula")
                ent["formula_source"] = "pubchem_lookup"
        else:
            ent["resolver_result"] = None

        cleaned.append(ent)

    chemistry["chemical_entities"] = cleaned
    return chemistry


# ---------------------------------------------------------------------------
# Structured experiment reasoning (V4 chain-of-thought)
# ---------------------------------------------------------------------------

def run_structured_experiment_reasoning(qwen,
                                        transcript_with_line_ids: str,
                                        allow_repair: bool = True) -> Dict[str, Any]:
    """
    Run 6 narrow sequential prompts instead of one big experiment prompt.
    Each step is individually parseable and retryable.
    """
    from .prompts import (
        GOAL_SCHEMA, MATERIALS_SCHEMA, CONDITIONS_SCHEMA,
        PROCEDURE_SCHEMA, RESULTS_SCHEMA, CONCLUSION_SCHEMA,
    )

    def _ask(prompt: str, schema: str) -> Dict[str, Any]:
        raw = qwen.ask_text(prompt, max_new_tokens=2048)
        return ensure_json_response(qwen, raw, schema_hint=schema, allow_repair=allow_repair)

    print("    4a: Extracting goal / project / date...")
    goal_json = _ask(build_goal_prompt(transcript_with_line_ids), GOAL_SCHEMA)

    print("    4b: Extracting materials list...")
    materials_json = _ask(build_materials_prompt(transcript_with_line_ids), MATERIALS_SCHEMA)

    print("    4c: Extracting experimental conditions...")
    conditions_json = _ask(build_conditions_prompt(transcript_with_line_ids), CONDITIONS_SCHEMA)

    print("    4d: Reconstructing procedure...")
    procedure_json = _ask(build_procedure_prompt(transcript_with_line_ids), PROCEDURE_SCHEMA)

    print("    4e: Extracting observations and results...")
    results_json = _ask(build_results_prompt(transcript_with_line_ids), RESULTS_SCHEMA)

    # Build a brief observations summary for the conclusion prompt
    obs_list = results_json.get("observations", [])
    obs_summary = "\n".join(
        f"  - {o.get('text', '')} (lines: {o.get('evidence_line_ids', [])})"
        for o in obs_list
    ) if obs_list else "  (no observations extracted yet)"

    print("    4f: Evaluating conclusion...")
    conclusion_json = _ask(
        build_conclusion_prompt(transcript_with_line_ids, obs_summary),
        CONCLUSION_SCHEMA,
    )

    return {
        "goal": goal_json,
        "materials": materials_json,
        "conditions": conditions_json,
        "procedure": procedure_json,
        "results": results_json,
        "conclusion": conclusion_json,
    }


# ---------------------------------------------------------------------------
# Main parser class
# ---------------------------------------------------------------------------

class LabNotebookParserV4:
    """
    V4 pipeline:
      Stage 0: Image pre-processing (deskew, contrast, ruled-line detection)
      Stage 1: Layout detection (OpenCV line segmentation + optional VLM hybrid)
      Stage 2: Hierarchical transcription (Qwen ± TrOCR ensemble + Pix2Tex)
      Stage 3: Chemistry extraction (MolScribe/DECIMER + NER + PubChem)
      Stage 4: Structured experiment reasoning (6 narrow prompts)

    Args:
        qwen:                    QwenVLExtractor instance
        output_dir:              directory for all outputs
        backend:                 "qwen_only" | "ensemble" | "full"
        enable_pubchem:          run PubChem name resolution
        allow_json_repair:       retry malformed JSON with repair prompt
    """

    def __init__(self,
                 qwen,
                 output_dir: str = "lab_parser_outputs_v4",
                 backend: str = "full",
                 enable_pubchem: bool = True,
                 allow_json_repair: bool = True):
        self.qwen = qwen
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.enable_pubchem = enable_pubchem
        self.allow_json_repair = allow_json_repair

        # Lazy-load secondary models based on backend
        self._trocr = None
        self._structure_recognizer = None

        if backend in ("ensemble", "full"):
            self._load_trocr()

        if backend == "full":
            self._load_structure_recognizer()

    def _load_trocr(self):
        try:
            from .trocr_wrapper import TrOCRExtractor
            self._trocr = TrOCRExtractor()
        except Exception as e:
            print(f"[Parser] TrOCR not available ({e}). Falling back to Qwen-only transcription.")

    def _load_structure_recognizer(self):
        try:
            from .structure_recognizer import StructureRecognizer
            self._structure_recognizer = StructureRecognizer(
                backend="both",
                use_pubchem=self.enable_pubchem,
            )
        except Exception as e:
            print(f"[Parser] Structure recognizer not available ({e}). Structures will use Qwen description only.")

    # ------------------------------------------------------------------

    def parse(self,
              image_path: str,
              max_vlm_width: int = 1600,
              save_outputs: bool = True,
              visualize_layout: bool = False) -> Dict[str, Any]:

        start_total = time.time()
        image_path_original = str(image_path)

        print(f"\n{'='*60}")
        print(f"V4 Lab Notebook Parser — backend: {self.backend}")
        print(f"Image: {image_path_original}")
        print(f"{'='*60}\n")

        # ----------------------------------------------------------
        # Stage 0: Pre-processing
        # ----------------------------------------------------------
        print("Stage 0/4: Image pre-processing (deskew + contrast + ruled lines)...")
        preproc = preprocess_for_vlm(
            image_path_original,
            output_dir=str(self.output_dir),
            max_width=max_vlm_width,
            apply_deskew=True,
            apply_contrast=True,
        )
        image_path_vlm = preproc["rgb_path"]
        ruled_lines = preproc["ruled_lines"]
        print(f"  Ruled lines detected: {len(ruled_lines)} at y={ruled_lines}")

        # ----------------------------------------------------------
        # Stage 1: Layout detection
        # ----------------------------------------------------------
        print("\nStage 1/4: Layout detection (OpenCV ruled-line segmentation)...")
        img_bgr = cv2.imread(image_path_vlm)
        opencv_regions = detect_layout_from_ruled_lines(
            img_bgr,
            ruled_lines=ruled_lines,
            min_row_height=14,
            padding=6,
        )
        print(f"  OpenCV regions detected: {len(opencv_regions)}")

        # Optional VLM layout pass for structure drawing regions
        vlm_layout_regions = []
        if self.backend == "full":
            print("  Running VLM layout pass for structure drawing regions...")
            raw_vlm_layout = self.qwen.ask_image(
                image_path_vlm, LAYOUT_PROMPT, max_new_tokens=1024
            )
            from .utils import extract_json_from_text
            vlm_layout_parsed = extract_json_from_text(raw_vlm_layout)
            vlm_layout_regions = vlm_layout_parsed.get("regions", [])
            print(f"  VLM layout regions: {len(vlm_layout_regions)}")

        # Merge OpenCV + VLM regions
        layout_regions = merge_opencv_and_vlm_regions(
            opencv_regions,
            vlm_layout_regions,
            img_size=(img_bgr.shape[1], img_bgr.shape[0]),
        )
        print(f"  Final merged regions: {len(layout_regions)}")

        if visualize_layout:
            vis_path = self.output_dir / "layout_debug.png"
            draw_layout_regions(img_bgr, layout_regions, output_path=str(vis_path))
            print(f"  Layout visualisation saved: {vis_path}")

        # Convert to dict for downstream compatibility
        layout_json = {"regions": [r.to_dict() for r in layout_regions]}

        # ----------------------------------------------------------
        # Stage 2: Transcription
        # ----------------------------------------------------------
        print(f"\nStage 2/4: Transcription ({'ensemble' if self._trocr else 'qwen_only'})...")
        transcription = transcribe_regions_with_vlm(
            self.qwen,
            image_path_vlm,
            layout_json,
            output_dir=str(self.output_dir / "region_crops"),
            allow_repair=self.allow_json_repair,
            trocr=self._trocr,
        )
        print(f"  Lines transcribed: {len(transcription.get('lines', []))}")
        print(f"  Tables found: {len(transcription.get('tables', []))}")
        print(f"  Formulas (LaTeX): {len(transcription.get('formulas', []))}")

        # ----------------------------------------------------------
        # Stage 2 (cont): Build merged transcript
        # ----------------------------------------------------------
        print("\nBuilding merged transcript...")
        merged_text, transcript_with_line_ids = build_transcript_blocks(transcription["lines"])
        merged_text = normalize_scientific_text(merged_text)
        print(f"  Transcript length: {len(merged_text)} chars")
        print("  Preview (first 800 chars):")
        print(merged_text[:800])

        # ----------------------------------------------------------
        # Stage 2 (cont): Deterministic extraction
        # ----------------------------------------------------------
        print("\nDeterministic quantity/formula/ratio extraction...")
        quantities = extract_quantities(merged_text, source="v4_transcription")
        explicit_formulas = extract_explicit_formula_mentions(merged_text, source="v4_transcription")
        ratios = extract_ratios(merged_text, source="v4_transcription")
        print(f"  Quantities: {len(quantities)}")
        print(f"  Formula mentions: {len(explicit_formulas)}")
        print(f"  Ratios: {len(ratios)}")

        # ----------------------------------------------------------
        # Stage 3: Chemistry extraction
        # ----------------------------------------------------------
        print("\nStage 3/4: Chemistry extraction...")

        # 3a: Structure recognition
        print("  3a: Chemical structure recognition...")
        structures = interpret_structures(
            self.qwen,
            image_path_vlm,
            layout_regions,
            output_dir=self.output_dir / "structure_crops",
            allow_repair=self.allow_json_repair,
            structure_recognizer=self._structure_recognizer,
        )
        print(f"  Structures found: {len(structures['structures'])}")

        # 3b: Chemistry NER from transcript
        print("  3b: Chemistry NER...")
        ner_entities = run_chemistry_ner(merged_text)
        print(f"  NER entities: {len(ner_entities)}")

        # 3c: VLM chemistry extraction from transcript
        print("  3c: VLM chemistry extraction from transcript...")
        chem_prompt = build_chemistry_from_transcript_prompt(merged_text)
        chem_raw = self.qwen.ask_text(chem_prompt, max_new_tokens=2048)
        chemistry_json = ensure_json_response(
            self.qwen, chem_raw, schema_hint=CHEMISTRY_SCHEMA, allow_repair=self.allow_json_repair
        )
        chemistry_json["_raw_response"] = chem_raw
        chemistry_json = sanitize_chemistry(chemistry_json, explicit_formulas, self.enable_pubchem)

        # Merge NER entities into chemistry output
        ner_ents_dict = [
            {
                "raw_text": e.raw_text,
                "normalized_name": e.normalized_name,
                "role": e.entity_type,
                "formula": e.formula,
                "formula_source": e.formula_source,
                "source": "chemdataextractor",
            }
            for e in ner_entities
        ]
        chemistry_json["ner_entities"] = ner_ents_dict

        # ----------------------------------------------------------
        # Stage 4: Structured experiment reasoning
        # ----------------------------------------------------------
        print("\nStage 4/4: Structured experiment reasoning (6-step chain)...")

        if self.backend == "full":
            experiment_json = run_structured_experiment_reasoning(
                self.qwen,
                transcript_with_line_ids,
                allow_repair=self.allow_json_repair,
            )
        else:
            # Fallback: single large prompt (V3-compatible)
            exp_prompt = build_experiment_from_transcript_prompt(transcript_with_line_ids)
            exp_raw = self.qwen.ask_text(exp_prompt, max_new_tokens=2048)
            experiment_json = ensure_json_response(
                self.qwen, exp_raw, schema_hint=EXPERIMENT_SCHEMA, allow_repair=self.allow_json_repair
            )

        # ----------------------------------------------------------
        # Assemble result
        # ----------------------------------------------------------
        total_time = round(time.time() - start_total, 2)
        print(f"\nTotal time: {total_time}s")

        result = {
            "document_metadata": {
                "source_original": image_path_original,
                "source_vlm_image": image_path_vlm,
                "strategy": f"v4_{self.backend}",
                "processing_status": "completed",
                "backend": self.backend,
                "enable_pubchem": self.enable_pubchem,
                "total_runtime_seconds": total_time,
            },
            "preprocessing": {
                "ruled_lines_y": ruled_lines,
                "orig_size": preproc["orig_size"],
                "scale_factor": preproc["scale_factor"],
            },
            "layout": layout_json,
            "transcription": {
                "lines": transcription.get("lines", []),
                "tables": transcription.get("tables", []),
                "formulas": transcription.get("formulas", []),
                "crop_records": transcription.get("crop_records", []),
            },
            "merged_transcript": merged_text,
            "merged_transcript_with_line_ids": transcript_with_line_ids,
            "deterministic_quantities": [asdict(q) for q in quantities],
            "explicit_formula_mentions": [asdict(f) for f in explicit_formulas],
            "ratios": [asdict(r) for r in ratios],
            "vlm_chemistry": chemistry_json,
            "structures": structures,
            "experiment_summary": experiment_json,
        }

        if save_outputs:
            base = Path(image_path_original).stem.replace(" ", "_")
            out_json = self.output_dir / f"{base}_v4_{self.backend}.json"
            save_json(result, out_json)

            raw_dir = self.output_dir / f"{base}_raw_debug"
            raw_dir.mkdir(exist_ok=True)

            (raw_dir / "region_raw_outputs.json").write_text(
                json.dumps(transcription.get("raw_region_outputs", []), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            (raw_dir / "structure_raw_outputs.json").write_text(
                json.dumps(structures.get("raw_structure_outputs", []), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"\nResults saved: {out_json}")
            print(f"Debug outputs: {raw_dir}")

        return result


# ---------------------------------------------------------------------------
# V3 backward-compat alias
# ---------------------------------------------------------------------------

class VLMFirstLabNotebookParserV3(LabNotebookParserV4):
    """
    Backward-compatible alias.
    Defaults to the 'qwen_only' backend to replicate V3 behaviour.
    """
    def __init__(self, qwen,
                 output_dir: str = "lab_parser_outputs_v3",
                 enable_pubchem_resolution: bool = False,
                 allow_json_repair: bool = True):
        super().__init__(
            qwen=qwen,
            output_dir=output_dir,
            backend="qwen_only",
            enable_pubchem=enable_pubchem_resolution,
            allow_json_repair=allow_json_repair,
        )
