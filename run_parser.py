"""
run_parser.py — entry point for the V4 lab notebook parser.

Backends:
  --backend qwen_only   : Qwen-only (V3-equivalent, fastest)
  --backend ensemble    : Qwen + TrOCR (better handwriting accuracy)
  --backend full        : All models — full V4 pipeline (recommended)
"""

import argparse
import json

from lab_notebook_parser import QwenVLExtractor, LabNotebookParserV4


def main():
    parser = argparse.ArgumentParser(
        description="V4 Lab Notebook Parser — handwritten chemistry lab page extraction"
    )
    parser.add_argument("--image", required=True,
                        help="Path to lab notebook page image (jpg, png, tiff)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="Qwen2.5-VL model name or local path")
    parser.add_argument("--output-dir", default="lab_parser_outputs_v4",
                        help="Directory for all output files")
    parser.add_argument("--max-width", type=int, default=1600,
                        help="Max image width for VLM (px); image is downscaled if wider")
    parser.add_argument(
        "--backend",
        choices=["qwen_only", "ensemble", "full"],
        default="full",
        help=(
            "qwen_only: Qwen2.5-VL only (fast, V3-compatible)\n"
            "ensemble:  Qwen + TrOCR handwriting model\n"
            "full:      All models — Qwen + TrOCR + MolScribe/DECIMER + NER (recommended)"
        ),
    )
    parser.add_argument("--pubchem", action="store_true",
                        help="Enable PubChem formula/name lookup (requires internet)")
    parser.add_argument("--visualize", action="store_true",
                        help="Save layout visualisation image (layout_debug.png)")
    parser.add_argument("--no-json-repair", action="store_true",
                        help="Disable automatic JSON repair on malformed VLM output")
    args = parser.parse_args()

    qwen = QwenVLExtractor(model_name=args.model, max_new_tokens=1024)

    lab_parser = LabNotebookParserV4(
        qwen=qwen,
        output_dir=args.output_dir,
        backend=args.backend,
        enable_pubchem=args.pubchem,
        allow_json_repair=not args.no_json_repair,
    )

    result = lab_parser.parse(
        args.image,
        max_vlm_width=args.max_width,
        save_outputs=True,
        visualize_layout=args.visualize,
    )

    # Print the structured summary to stdout
    summary = {
        "document_metadata": result.get("document_metadata"),
        "merged_transcript": result.get("merged_transcript"),
        "deterministic_quantities": result.get("deterministic_quantities"),
        "explicit_formula_mentions": result.get("explicit_formula_mentions"),
        "ratios": result.get("ratios"),
        "structures_found": len(result.get("structures", {}).get("structures", [])),
        "experiment_summary": result.get("experiment_summary"),
        "vlm_chemistry": result.get("vlm_chemistry"),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
