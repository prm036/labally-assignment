import os
import re
import json

files_in_order = [
    "lab_notebook_parser/utils.py",
    "lab_notebook_parser/prompts.py",
    "lab_notebook_parser/extraction_rules.py",
    "lab_notebook_parser/qwen_wrapper.py",
    "lab_notebook_parser/trocr_wrapper.py",
    "lab_notebook_parser/structure_recognizer.py",
    "lab_notebook_parser/preprocess.py",
    "lab_notebook_parser/layout.py",
    "lab_notebook_parser/layout_detector.py",
    "lab_notebook_parser/transcription.py",
    "lab_notebook_parser/parser.py"
]

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.split("\n")][:-1] + [text.split("\n")[-1]]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.split("\n")][:-1] + [text.split("\n")[-1]]
    })

add_markdown("# V4 Lab Notebook Parser - Standalone Edition\nThis notebook contains the entire codebase flattened into executable cells in Google Colab.")

add_markdown("## 1. Setup\nUpload your image (e.g., `Example_lab_notebook_page.jpg`) to Google Colab. Then, run this cell to install dependencies.")
add_code("!pip install opencv-python>=4.8 numpy>=1.24 Pillow>=10.0,<12.0 torch torchvision accelerate transformers qwen-vl-utils pix2tex DECIMER>=2.4 chemdataextractor2 rdkit pandas==2.2.2 requests==2.32.4 matplotlib scipy")

add_markdown("## 2. Codebase\nThe following cells contain the classes and functions for the parser, loaded in dependency order.")

pattern = re.compile(r'^[ \t]*from (?:\.|lab_notebook_parser)[^\n]*import[ \t]*(?:\([\s\S]*?\)|[^\n]+)\n?', re.MULTILINE)

for filepath in files_in_order:
    with open(filepath) as f:
        content = f.read()
    
    # Strip local imports
    content = pattern.sub('', content)
    
    header = f"# ==========================================\n# Original File: {filepath}\n# ==========================================\n\n"
    add_code(header + content.strip() + "\n")

add_markdown("## 3. Run Pipeline\nConfigure your parameters and run the extraction.")

run_code = """import json

# Parameters
IMAGE_PATH = 'Example_lab_notebook_page.jpg'
MODEL_NAME = 'Qwen/Qwen2.5-VL-3B-Instruct' # Note: 2B does not exist! Use 3B or 7B.
OUTPUT_DIR = 'lab_parser_outputs_v4'
MAX_WIDTH = 1600
BACKEND = 'full' # Options: 'qwen_only', 'ensemble', 'full'
ENABLE_PUBCHEM = True
VISUALIZE = True
ALLOW_JSON_REPAIR = True

# Initialize extractor
qwen = QwenVLExtractor(model_name=MODEL_NAME, max_new_tokens=1024)

# Initialize parser
lab_parser = LabNotebookParserV4(
    qwen=qwen,
    output_dir=OUTPUT_DIR,
    backend=BACKEND,
    enable_pubchem=ENABLE_PUBCHEM,
    allow_json_repair=ALLOW_JSON_REPAIR,
)

# Run parsing
result = lab_parser.parse(
    IMAGE_PATH,
    max_vlm_width=MAX_WIDTH,
    save_outputs=True,
    visualize_layout=VISUALIZE,
)

# Display summary
summary = {
    'document_metadata': result.get('document_metadata'),
    'merged_transcript': result.get('merged_transcript'),
    'deterministic_quantities': result.get('deterministic_quantities'),
    'explicit_formula_mentions': result.get('explicit_formula_mentions'),
    'ratios': result.get('ratios'),
    'structures_found': len(result.get('structures', {}).get('structures', [])),
    'experiment_summary': result.get('experiment_summary'),
    'vlm_chemistry': result.get('vlm_chemistry'),
}
print(json.dumps(summary, indent=2, ensure_ascii=False))
"""
add_code(run_code)

with open("run_parser_standalone.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
print("run_parser_standalone.ipynb created successfully.")
