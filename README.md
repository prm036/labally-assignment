# Chemistry Lab Notebook Digitization Pipeline

An automated, end-to-end pipeline for extracting, parsing, and summarizing handwritten chemistry lab notebooks, with a focus on battery and electrochemistry domains.

This tool takes an image of a handwritten lab notebook page and processes it through AI models to yield a structured summary of the experiment, including its goals, conditions, procedure, results, and materials used.

## Features

The pipeline is split into five main stages:
1. **Full-Page OCR (`stage1_page_ocr`)**: Transcribes the full notebook page using `olmOCR`, preserving layout elements like tables, steps, and Markdown formatting.
2. **Region Detection (`stage2_detect`)**: Uses a YOLO model to locate and crop specific regions like reaction schemes and chemical structures.
3. **Scheme OCR & OCSR (`stage3`)**: Runs OCR specifically on reaction scheme crops. Optionally(this is still a TODO!), Optical Chemical Structure Recognition (OCSR) can be applied to convert structure crops into SMILES strings.
4. **Chemistry NER (`stage4_ner`)**: Extracts domain-specific entities (e.g., chemicals, solvents, temperatures, current densities, equipment) from the transcribed text using an instruction-tuned LLM.
5. **Experiment Synthesis (`stage5_summary`)**: Consolidates all the extracted information to answer four main questions:
   - What was the goal of the experiment?
   - What were the conditions?
   - What was the step-by-step procedure?
   - What were the measured outcomes and observations?

## Requirements

The pipeline expects API endpoints (OpenAI-compatible) for OCR and Instruct-LLM. The default configuration assumes local `vLLM` instances are running.

- **Python Packages**: `openai`, `pydantic`, `pyyaml`, `Pillow`, `ultralytics`
- **OCR Model**: `allenai/olmOCR-2-7B-1025` (Default: `http://localhost:8000/v1`)
- **LLM Model**: `Qwen/Qwen2.5-7B-Instruct` (Default: `http://localhost:8001/v1`)
- **YOLO Weights**: By default, it looks for weights at `runs/chem/yolo11n_chem/weights/best.pt`.

## Configuration

You can tweak the default endpoints, model names, sampling parameters, and confidence thresholds by modifying the `Config` dataclass located at the top of `chem_pipeline.py`. 

## Usage

You can run the pipeline from the command line by providing a path to a lab notebook page image:

```bash
python chem_pipeline.py path/to/notebook_page.jpg
```

### Command-line Arguments

- `page`: Path to the notebook page image (required).
- `--weights`: Override the path to the YOLO weights.
- `--ocsr`: Enable the OCSR step (converts structure crops to SMILES). Note: You need to plug in your own OCSR backend in `stage3b_ocsr` for this to work.
- `--run-dir`: Set a custom output directory to save intermediate JSON files and crops (default: `runs/pipeline`).

### Example

```bash
python chem_pipeline.py Example_lab_notebook_page.jpg --run-dir runs/test_run
```

The script will print a structured summary to standard output and save all the intermediate model outputs, crops, and metadata into the specified `run-dir`.

## Limitations

- **Spatial Understanding in VLMs**: Qwen-2.5 Vision-Language Models (VLMs) can be fragile when it comes to spatial understanding. They often struggle to correctly interpret complex, non-linear 2D layouts natively, such as the chemical structures found in handwritten lab notes. To mitigate this, our pipeline relies on a dedicated YOLO detection step to localize and crop out these schemes and structures before processing them further.
- **Chemical Reaction Schemes**: Although the YOLO model can detect reaction schemes, the current pipeline does not have a dedicated step to convert these cropped schemes into SMILES strings (a task requiring Optical Chemical Structure Recognition). This limitation means that while the pipeline can extract the text describing the reaction, it cannot yet reconstruct the chemical structures themselves.
