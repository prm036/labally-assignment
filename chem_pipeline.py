from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import yaml
from PIL import Image
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

# ---- rdkit is optional; SMILES validation degrades gracefully without it ----
try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
except Exception:  # pragma: no cover
    _HAS_RDKIT = False


# ============================================================================ #
#  CONFIG
# ============================================================================ #
@dataclass
class Config:
    # --- OCR endpoint (olmOCR on vLLM) -----------------------------------
    ocr_base_url: str = "http://localhost:8000/v1"
    ocr_model: str = "allenai/olmOCR-2-7B-1025"
    ocr_api_key: str = "EMPTY"

    # --- LLM endpoint (instruct model for NER + synthesis) ---------------
    # Point this at a second vLLM, or any OpenAI-compatible API.
    llm_base_url: str = "http://localhost:8001/v1"
    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    llm_api_key: str = "EMPTY"

    # --- YOLO detector ----------------------------------------------------
    yolo_weights: str = "runs/chem/yolo11n_chem/weights/best.pt"
    device: str = "0"
    conf: float = 0.25
    imgsz: int = 1024
    # how the classes in YOUR data.yaml map to roles (edit to match your names)
    scheme_classes: tuple[str, ...] = ("reaction_scheme", "scheme", "reaction")
    structure_classes: tuple[str, ...] = ("chemical_structure", "molecule", "structure")

    # --- olmOCR sampling --------------------------------------------------
    longest_dim: int = 1288        # 1288 for olmOCR-2 (1025); 1024 for 0225-preview
    ocr_temperature: float = 0.1
    ocr_max_tokens: int = 2048
    rotate_retry: bool = True

    # --- instruct-LLM sampling -------------------------------------------
    llm_temperature: float = 0.0
    llm_max_tokens: int = 3072
    use_guided_json: bool = True   # vLLM guided decoding -> reliable JSON

    # --- behaviour / io ---------------------------------------------------
    do_ocsr: bool = True          # run structure->SMILES on chemical_structure crops
    run_dir: str = "runs/pipeline"
    max_retries: int = 4


# ============================================================================ #
#  PYDANTIC SCHEMAS
# ============================================================================ #
class Quantity(BaseModel):
    value: Optional[float] = None
    unit: Optional[str] = None
    text: str = Field(..., description="verbatim span as written")


class ChemEntity(BaseModel):
    """One extracted entity from the notebook text."""
    text: str = Field(..., description="exact surface string from the page")
    label: str = Field(..., description=(
        "one of: CHEMICAL, FORMULA, SOLVENT, SALT, ADDITIVE, ELECTRODE, "
        "AMOUNT, CONCENTRATION, TEMPERATURE, TIME, CURRENT_DENSITY, POTENTIAL, "
        "CAPACITY, CURRENT, EQUIPMENT, SAMPLE_ID, MEASUREMENT, OTHER"))
    normalized: Optional[str] = Field(None, description="canonical name/value if obvious")
    smiles: Optional[str] = Field(None, description="SMILES if a CHEMICAL with a clear identity")
    source: str = Field("page", description="page | scheme | structure")


class NERResult(BaseModel):
    entities: list[ChemEntity] = Field(default_factory=list)


class Material(BaseModel):
    name: str
    role: str = Field(..., description="reactant|product|solvent|salt|additive|electrode|reference|gas|other")
    amount: Optional[str] = None
    concentration: Optional[str] = None
    smiles: Optional[str] = None


class ExperimentSummary(BaseModel):
    run_id: Optional[str] = None
    title: Optional[str] = None
    # the four questions the assignment asks
    goal: str = Field(..., description="what the experiment set out to do")
    conditions: dict[str, Any] = Field(
        default_factory=dict,
        description="temperature, atmosphere, current density, cutoff voltage, "
                    "areal capacity, cell format, electrolyte, etc.")
    procedure: list[str] = Field(default_factory=list, description="ordered steps")
    results: list[str] = Field(default_factory=list, description="observations + measured outcomes")
    # supporting structure
    materials: list[Material] = Field(default_factory=list)
    conclusions: Optional[str] = None
    uncertainties: list[str] = Field(
        default_factory=list,
        description="anything illegible/ambiguous the model is NOT sure about")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


# ============================================================================ #
#  LOW-LEVEL CLIENTS  (retry wrapper around OpenAI-compatible chat)
# ============================================================================ #
def _chat(client: OpenAI, *, max_retries: int, **kwargs) -> str:
    last: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:  # transient server / network errors
            last = e
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"chat failed after {max_retries} retries: {last}")


# ============================================================================ #
#  olmOCR helpers  (shared by full-page and crop OCR)
# ============================================================================ #
def encode_image(im: Image.Image, longest: int) -> str:
    im = im.convert("RGB")
    im.thumbnail((longest, longest))
    buf = BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def build_page_prompt() -> str:
    return (
        "Attached is a full page from a handwritten chemistry lab notebook.\n"
        "Transcribe ALL handwritten and printed text in natural reading order as "
        "Markdown. Reproduce tables as Markdown tables. Keep headings, dates, run "
        "IDs, lists and numbered steps.\n"
        "Rules:\n"
        "- Transcribe what is WRITTEN; do not paraphrase or invent. Unreadable text "
        "-> [illegible]. Never substitute a plausible reagent for one you cannot read.\n"
        "- Where a molecule is DRAWN as a structure, output the token [structure] in "
        "place of it; where a full reaction is drawn, output [scheme]. Never transcribe "
        "drawings atom-by-atom or as LaTeX arrays.\n"
        "- Preserve subscripts/superscripts/charges as LaTeX: \\(H_2O\\), \\(Li^+\\), "
        "\\(1\\,mA\\,cm^{-2}\\). Inline \\( \\), block \\[ \\]; no unicode math glyphs.\n"
        "- If there is no readable text at all, output null for natural_text.\n"
        "Return Markdown with YAML front matter: primary_language, is_rotation_valid, "
        "rotation_correction, is_table, is_diagram."
    )


def build_scheme_prompt() -> str:
    return (
        "Attached is a cropped reaction scheme from a handwritten chemistry lab "
        "notebook: reactants, an arrow with conditions, and products.\n"
        "Transcribe what is WRITTEN. Distinguish two cases:\n"
        "- If a species is written as text or a chemical formula (e.g. EtONa, "
        "\\(H_2O\\)), transcribe that text exactly.\n"
        "- If a species is drawn as a structure WITH a written name/label next to it, "
        "transcribe only that written label.\n"
        "- If a species is drawn as a structure with no label, output the single token "
        "[structure] and STOP describing it. Never transcribe individual atoms "
        "(O, C, N...) of a drawing, never render it as LaTeX or an array. "
        "Atom-by-atom output is forbidden.\n"
        "Rules:\n"
        "- Do not hallucinate. Unreadable text -> [illegible].\n"
        "- Preserve subscripts/superscripts/charges as LaTeX: \\(H_2O\\), \\(Li^+\\), "
        "\\(30^\\circ C\\). Inline \\( \\), block \\[ \\]; no unicode math glyphs.\n"
        "- This is a fragment of a larger page; transcribe partial text as-is, do not "
        "complete cut-off words.\n"
        "- If there is no readable text at all, output null for natural_text.\n"
        "Return Markdown with YAML front matter: primary_language, is_rotation_valid, "
        "rotation_correction, is_table, is_diagram."
    )


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Split olmOCR's '---\\n<yaml>\\n---\\n<body>' into (meta, body)."""
    m = re.match(r"\s*---\s*\n(.*?)\n---\s*\n?(.*)", text, re.DOTALL)
    if not m:
        return {}, text.strip()
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, m.group(2).strip()


def _olmocr_call(client: OpenAI, cfg: Config, im: Image.Image, prompt: str) -> str:
    b64 = encode_image(im, cfg.longest_dim)
    return _chat(
        client, max_retries=cfg.max_retries,
        model=cfg.ocr_model,
        temperature=cfg.ocr_temperature,
        max_tokens=cfg.ocr_max_tokens,
        extra_body={
            "repetition_penalty": 1.15,        # kills the token-repetition loop
            "stop": ["\\\\\n\\text{O}"],       # hard stop on the known loop pattern
        },
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]}],
    )


def olmocr_image(client: OpenAI, cfg: Config, path: Path, prompt: str) -> dict:
    """Run olmOCR on one image with one rotation-retry. Works for page or crop."""
    im = Image.open(path)
    raw = _olmocr_call(client, cfg, im, prompt)
    meta, body = parse_front_matter(raw)

    if (cfg.rotate_retry and meta.get("is_rotation_valid") is False
            and meta.get("rotation_correction") in (90, 180, 270)):
        # PIL rotates CCW; model reports CW correction -> negate
        im_rot = im.rotate(-int(meta["rotation_correction"]), expand=True)
        raw = _olmocr_call(client, cfg, im_rot, prompt)
        meta, body = parse_front_matter(raw)

    natural = None if body.strip().lower() == "null" else body
    return {
        "file": path.name,
        "primary_language": meta.get("primary_language"),
        "is_rotation_valid": meta.get("is_rotation_valid"),
        "rotation_correction": meta.get("rotation_correction"),
        "is_table": meta.get("is_table"),
        "is_diagram": meta.get("is_diagram"),
        "natural_text": natural,
        "raw": raw,
    }


# ============================================================================ #
#  JSON / SMILES utilities for the reasoning stages
# ============================================================================ #
def extract_json(text: str) -> Any:
    """Robustly pull a JSON object/array out of an LLM reply."""
    text = text.strip()
    # strip ```json ... ``` fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # fall back to first balanced {...} or [...]
    start = min([i for i in (text.find("{"), text.find("[")) if i != -1], default=-1)
    if start == -1:
        raise ValueError("no JSON found in model output")
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            esc = (c == "\\" and not esc)
            if c == '"' and not esc:
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON in model output")


def validate_smiles(smi: Optional[str]) -> Optional[str]:
    """Return canonical SMILES if valid, else None. No-op pass-through w/o rdkit."""
    if not smi:
        return None
    if not _HAS_RDKIT:
        return smi
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _llm_json(client: OpenAI, cfg: Config, system: str, user: str,
              schema: Optional[dict] = None) -> Any:
    extra: dict = {}
    if cfg.use_guided_json and schema is not None:
        extra["guided_json"] = schema      # vLLM structured output
    raw = _chat(
        client, max_retries=cfg.max_retries,
        model=cfg.llm_model,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
        extra_body=extra or None,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return extract_json(raw)


# ============================================================================ #
#  STAGE 1 — full-page OCR
# ============================================================================ #
def stage1_page_ocr(cfg: Config, page_path: str | Path,
                    ocr_client: Optional[OpenAI] = None) -> dict:
    ocr_client = ocr_client or OpenAI(base_url=cfg.ocr_base_url, api_key=cfg.ocr_api_key)
    return olmocr_image(ocr_client, cfg, Path(page_path), build_page_prompt())


# ============================================================================ #
#  STAGE 2 — YOLO region detection + crops
# ============================================================================ #
def stage2_detect(cfg: Config, page_path: str | Path) -> dict:
    from ultralytics import YOLO
    model = YOLO(cfg.yolo_weights)
    res = model.predict(source=str(page_path), conf=cfg.conf, imgsz=cfg.imgsz,
                        device=cfg.device, save=True, save_crop=True, verbose=False)[0]
    names = res.names
    save_dir = Path(res.save_dir)
    dets = []
    for b in res.boxes:
        cls = names[int(b.cls)]
        dets.append({"cls": cls, "conf": float(b.conf),
                     "xyxy": [round(float(x), 1) for x in b.xyxy[0].tolist()]})
    # ultralytics writes crops to <save_dir>/crops/<class_name>/
    crop_dirs = {c.name: c for c in (save_dir / "crops").glob("*") if c.is_dir()}
    scheme_dirs = [crop_dirs[c] for c in cfg.scheme_classes if c in crop_dirs]
    structure_dirs = [crop_dirs[c] for c in cfg.structure_classes if c in crop_dirs]
    return {
        "save_dir": str(save_dir),
        "detections": dets,
        "scheme_crop_dirs": [str(d) for d in scheme_dirs],
        "structure_crop_dirs": [str(d) for d in structure_dirs],
        "names": names,
    }


def _iter_crops(dirs: list[str]):
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    for d in dirs:
        for p in sorted(Path(d).iterdir()):
            if p.suffix.lower() in exts:
                yield p


# ============================================================================ #
#  STAGE 3a — olmOCR over reaction_scheme crops
# ============================================================================ #
def stage3_scheme_ocr(cfg: Config, scheme_crop_dirs: list[str],
                      ocr_client: Optional[OpenAI] = None) -> list[dict]:
    ocr_client = ocr_client or OpenAI(base_url=cfg.ocr_base_url, api_key=cfg.ocr_api_key)
    out = []
    for p in _iter_crops(scheme_crop_dirs):
        try:
            out.append(olmocr_image(ocr_client, cfg, p, build_scheme_prompt()))
        except Exception as e:
            out.append({"file": p.name, "error": str(e)})
    return out


# ============================================================================ #
#  STAGE 3b — OCSR over chemical_structure crops (optional hook)
# ============================================================================ #
def ocsr_predict(path: Path) -> Optional[str]:
    """Structure image -> SMILES. Plug in your fine-tuned model, DECIMER, or MolScribe.

    Example with DECIMER:
        from DECIMER import predict_SMILES
        return predict_SMILES(str(path))
    Returns None if no OCSR backend is wired up.
    """
    return None


def stage3b_ocsr(cfg: Config, structure_crop_dirs: list[str]) -> list[dict]:
    out = []
    for p in _iter_crops(structure_crop_dirs):
        try:
            smi = validate_smiles(ocsr_predict(p))
            out.append({"file": p.name, "smiles": smi})
        except Exception as e:
            out.append({"file": p.name, "error": str(e)})
    return out


# ============================================================================ #
#  STAGE 4 — chemistry NER
# ============================================================================ #
NER_SYSTEM = (
    "You are a chemistry information-extraction system for handwritten lab "
    "notebooks (battery / electrochemistry domain). Extract entities EXACTLY as "
    "written. Do not invent reagents, values, or units. Only return entities whose "
    "surface string appears in the provided text."
)


def stage4_ner(cfg: Config, text: str,
               source: str = "page",
               llm_client: Optional[OpenAI] = None) -> NERResult:
    llm_client = llm_client or OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)
    user = (
        "Extract all chemistry entities from the notebook text below. Labels: "
        "CHEMICAL, FORMULA, SOLVENT, SALT, ADDITIVE, ELECTRODE, AMOUNT, "
        "CONCENTRATION, TEMPERATURE, TIME, CURRENT_DENSITY, POTENTIAL, CAPACITY, "
        "CURRENT, EQUIPMENT, SAMPLE_ID, MEASUREMENT, OTHER.\n"
        "For a CHEMICAL with a clear, common identity you may add a SMILES; otherwise "
        "leave smiles null. Set `normalized` to a canonical name/value when obvious.\n"
        f'Return JSON: {{"entities":[{{"text","label","normalized","smiles",'
        f'"source":"{source}"}}]}}\n\n'
        f"TEXT:\n{text}"
    )
    schema = NERResult.model_json_schema()
    try:
        data = _llm_json(llm_client, cfg, NER_SYSTEM, user, schema)
        ner = NERResult.model_validate(data)
    except (ValidationError, ValueError) as e:
        return NERResult(entities=[])
    # validate any SMILES the model attached
    for ent in ner.entities:
        ent.smiles = validate_smiles(ent.smiles)
        if not ent.source:
            ent.source = source
    return ner


# ============================================================================ #
#  STAGE 5 — synthesis: goal / conditions / procedure / results
# ============================================================================ #
SUMMARY_SYSTEM = (
    "You are a senior electrochemist reading a digitized handwritten lab-notebook "
    "page. Using ONLY the transcribed text, extracted entities, and reaction-scheme "
    "transcriptions provided, reconstruct what the experiment was. Do not invent "
    "facts. If something is illegible or absent, say so in `uncertainties` and lower "
    "`confidence`. Quote measured values exactly as transcribed."
)


def stage5_summary(cfg: Config,
                   page_text: Optional[str],
                   scheme_texts: list[str],
                   ner: NERResult,
                   smiles: Optional[list[dict]] = None,
                   llm_client: Optional[OpenAI] = None) -> ExperimentSummary:
    llm_client = llm_client or OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)
    schemes_blob = "\n\n".join(f"[scheme {i+1}]\n{t}" for i, t in enumerate(scheme_texts) if t)
    ner_blob = json.dumps([e.model_dump() for e in ner.entities], ensure_ascii=False)
    smiles_blob = json.dumps(smiles or [], ensure_ascii=False)
    user = (
        "Reconstruct the experiment and answer four questions: the GOAL, the "
        "CONDITIONS, the PROCEDURE (ordered steps), and the RESULTS.\n"
        "Return JSON matching this shape: {run_id, title, goal, conditions (object), "
        "procedure (list of strings), results (list of strings), materials "
        "(list of {name, role, amount, concentration, smiles}), conclusions, "
        "uncertainties (list), confidence (0-1 float)}.\n\n"
        f"=== PAGE TRANSCRIPTION ===\n{page_text or '[none]'}\n\n"
        f"=== REACTION SCHEME TRANSCRIPTIONS ===\n{schemes_blob or '[none]'}\n\n"
        f"=== EXTRACTED ENTITIES (NER) ===\n{ner_blob}\n\n"
        f"=== STRUCTURE SMILES (OCSR) ===\n{smiles_blob}\n"
    )
    schema = ExperimentSummary.model_json_schema()
    try:
        data = _llm_json(llm_client, cfg, SUMMARY_SYSTEM, user, schema)
        return ExperimentSummary.model_validate(data)
    except (ValidationError, ValueError) as e:
        return ExperimentSummary(goal=f"[synthesis failed: {e}]", confidence=0.0)


# ============================================================================ #
#  ORCHESTRATOR
# ============================================================================ #
def run_all(cfg: Config, page_path: str | Path) -> dict:
    page_path = Path(page_path)
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ocr_client = OpenAI(base_url=cfg.ocr_base_url, api_key=cfg.ocr_api_key)
    llm_client = OpenAI(base_url=cfg.llm_base_url, api_key=cfg.llm_api_key)

    # 1. full-page OCR
    page = stage1_page_ocr(cfg, page_path, ocr_client)
    (run_dir / "1_page_ocr.json").write_text(json.dumps(page, indent=2, ensure_ascii=False))

    # 2. detection + crops
    det = stage2_detect(cfg, page_path)
    (run_dir / "2_detections.json").write_text(json.dumps(det, indent=2, ensure_ascii=False))

    # 3. scheme OCR
    schemes = stage3_scheme_ocr(cfg, det["scheme_crop_dirs"], ocr_client)
    (run_dir / "3_scheme_ocr.json").write_text(json.dumps(schemes, indent=2, ensure_ascii=False))

    # 3b. optional OCSR
    smiles = stage3b_ocsr(cfg, det["structure_crop_dirs"]) if cfg.do_ocsr else []
    if smiles:
        (run_dir / "3b_ocsr.json").write_text(json.dumps(smiles, indent=2, ensure_ascii=False))

    # 4. NER over page + scheme text
    scheme_texts = [s.get("natural_text") or "" for s in schemes]
    combined = (page.get("natural_text") or "") + "\n\n" + "\n\n".join(scheme_texts)
    ner = stage4_ner(cfg, combined, source="page+scheme", llm_client=llm_client)
    (run_dir / "4_ner.json").write_text(ner.model_dump_json(indent=2))

    # 5. synthesis
    summary = stage5_summary(cfg, page.get("natural_text"), scheme_texts, ner,
                             smiles, llm_client=llm_client)
    (run_dir / "5_summary.json").write_text(summary.model_dump_json(indent=2))

    bundle = {
        "page_path": str(page_path),
        "page_ocr": page,
        "detections": det,
        "scheme_ocr": schemes,
        "ocsr": smiles,
        "ner": ner.model_dump(),
        "summary": summary.model_dump(),
    }
    (run_dir / "run.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=False))
    return bundle


def print_summary(summary: ExperimentSummary) -> None:
    s = summary
    print("=" * 72)
    print(f"RUN: {s.run_id or '-'}   {s.title or ''}".strip())
    print("=" * 72)
    print(f"\nGOAL\n  {s.goal}")
    print("\nCONDITIONS")
    for k, v in s.conditions.items():
        print(f"  {k}: {v}")
    print("\nPROCEDURE")
    for i, step in enumerate(s.procedure, 1):
        print(f"  {i}. {step}")
    print("\nRESULTS")
    for r in s.results:
        print(f"  - {r}")
    if s.materials:
        print("\nMATERIALS")
        for m in s.materials:
            extra = " / ".join(x for x in (m.amount, m.concentration, m.smiles) if x)
            print(f"  - {m.name} ({m.role}){(' — ' + extra) if extra else ''}")
    if s.conclusions:
        print(f"\nCONCLUSIONS\n  {s.conclusions}")
    if s.uncertainties:
        print("\nUNCERTAINTIES")
        for u in s.uncertainties:
            print(f"  - {u}")
    print(f"\nCONFIDENCE: {s.confidence:.2f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Chem lab-notebook understanding pipeline")
    ap.add_argument("page", help="path to the notebook page image")
    ap.add_argument("--weights", default=None, help="override YOLO weights path")
    ap.add_argument("--ocsr", action="store_true", help="run structure->SMILES")
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args()

    cfg = Config()
    if args.weights:
        cfg.yolo_weights = args.weights
    if args.run_dir:
        cfg.run_dir = args.run_dir
    cfg.do_ocsr = args.ocsr

    bundle = run_all(cfg, args.page)
    print_summary(ExperimentSummary.model_validate(bundle["summary"]))
