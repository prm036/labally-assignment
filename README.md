# Lab Notebook Parser — V4

A four-stage pipeline for extracting handwritten chemistry lab notebook pages.
Targets all four evaluation levels: plain text, special symbols, chemistry, and experiment understanding.

## Architecture

```
Stage 0 — Image Pre-processing        preprocess.py
    Deskew (Hough line), CLAHE contrast, ruled-line detection, region upscaling

Stage 1 — Layout Detection            layout_detector.py
    OpenCV ruled-line segmentation → row bands
    Vertical projection profiles → sub-region columns
    Heuristic region type classifier (text / table / structure_drawing / formula)
    Optional VLM hybrid merge for structure drawing regions

Stage 2 — Hierarchical Transcription  transcription.py
    2a. Text/symbols  — Qwen2.5-VL-7B + TrOCR-large ensemble (token-level voting)
    2b. Tables        — structured cell-by-cell extraction → pandas-compatible JSON
    2c. Formulas      — Pix2Tex LaTeX-OCR → sympy-parseable LaTeX

Stage 3 — Chemistry Extraction        structure_recognizer.py + extraction_rules.py
    3a. Structures    — MolScribe (primary) + DECIMER (fallback) → validated SMILES
                        RDKit validation + PubChem CID lookup
    3b. NER           — ChemDataExtractor entity recognition
    3c. VLM chem      — Qwen chemistry extraction from merged transcript

Stage 4 — Experiment Reasoning        parser.py (run_structured_experiment_reasoning)
    6 narrow sequential prompts (vs one large prompt in V3):
      goal / materials / conditions / procedure / results / conclusion
```

## Key improvements over V3

| Issue (V3)                              | Fix (V4)                                         |
|----------------------------------------|--------------------------------------------------|
| Qwen used for bbox layout detection    | OpenCV ruled-line segmentation (deterministic)   |
| No image preprocessing                 | Deskew + CLAHE before any model call             |
| Single model for all regions           | Best-fit model per region type                   |
| Zero SMILES from hand-drawn structures | MolScribe + DECIMER with RDKit validation        |
| 14-element formula validator           | Full 118-element periodic table                  |
| Missing Greek symbols (2θ, λ, Δ, etc.) | Comprehensive normalisation pass                 |
| One-shot experiment reasoning          | 6-step chain-of-thought                          |
| No ratio extraction                    | Dedicated ratio parser (v/v, mol%, w/w)          |
| No NER                                 | ChemDataExtractor entity recognition             |
| Equations via generic transcription    | Pix2Tex LaTeX-OCR                                |

## Install

```bash
# Base environment
pip install -r requirements.txt

# Latest transformers (required for Qwen2.5-VL + TrOCR)
pip install --upgrade --no-deps git+https://github.com/huggingface/transformers
```

## Run

### Full V4 pipeline (recommended)
```bash
python run_parser.py \
  --image Example_lab_notebook_page.jpg \
  --backend full \
  --pubchem \
  --visualize
```

### Ensemble (Qwen + TrOCR, no structure models)
```bash
python run_parser.py --image Example_lab_notebook_page.jpg --backend ensemble
```

### V3-compatible (Qwen only)
```bash
python run_parser.py --image Example_lab_notebook_page.jpg --backend qwen_only
```

### Cluster (SLURM)
```bash
sbatch run_job.slurm
```

## Validate output
```bash
python tests/validate_output.py lab_parser_outputs_v4/Example_lab_notebook_page_v4_full.json --verbose
```

## File structure

```
lab_notebook_parser/
  __init__.py
  qwen_wrapper.py          Qwen2.5-VL model wrapper
  trocr_wrapper.py         TrOCR handwriting OCR wrapper        [NEW]
  preprocess.py            Stage 0: deskew + contrast + lines   [NEW]
  layout_detector.py       Stage 1: OpenCV layout detection     [NEW]
  transcription.py         Stage 2: ensemble transcription      [UPDATED]
  structure_recognizer.py  MolScribe/DECIMER → SMILES           [NEW]
  extraction_rules.py      Deterministic extraction rules       [UPDATED]
  prompts.py               All prompt templates                 [UPDATED]
  parser.py                Main V4 orchestration                [UPDATED]
  utils.py                 Shared utilities
run_parser.py              Entry point
run_job.slurm              SLURM batch job
tests/
  test_deterministic.py
  validate_output.py       Output quality validation            [NEW]
requirements.txt
```

## Design rules

1. **No hardcoded formula corrections** — formulas are accepted only if:
   - explicitly written in the notebook transcription, OR
   - produced by MolScribe/DECIMER with RDKit validation, OR
   - resolved via PubChem lookup (when `--pubchem` is enabled)

2. **No hallucinated conclusions** — a conclusion is emitted only if it has
   supporting `evidence_line_ids` from the transcript.

3. **Symbol fidelity** — all Greek symbols, sub/superscripts, arrows, and units
   are normalised to Unicode (not ASCII approximations) in the final output.

4. **Full periodic table** — the element validator covers all 118 elements
   (not just the 14-element subset in V3 that silently dropped fluorine).
