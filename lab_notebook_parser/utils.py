from pathlib import Path
from typing import Dict, Any, List
import json
import cv2
import matplotlib.pyplot as plt
from PIL import Image


def show_image(img, title="Image", figsize=(10, 8), cmap="gray"):
    plt.figure(figsize=figsize)
    if len(img.shape) == 2:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()


def load_image_cv2(image_path: str):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return img


def save_json(data: Dict[str, Any], path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON to: {path}")


def make_resized_copy_for_vlm(image_path: str, max_width: int = 1600, output_dir: str = "lab_parser_outputs_v3") -> str:
    image_path = Path(image_path)
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if w <= max_width:
        out_path = output_dir / f"{image_path.stem}_vlm_input.png"
        img.save(out_path)
        return str(out_path)

    scale = max_width / w
    new_size = (max_width, int(h * scale))
    resized = img.resize(new_size)
    out_path = output_dir / f"{image_path.stem}_vlm_input_{max_width}px.png"
    resized.save(out_path)
    return str(out_path)


def crop_image_region(image_path: str, bbox: List[int], output_path: str, padding: int = 8) -> str:
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(W, x2 + padding)
    y2 = min(H, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid crop bbox after clipping: {bbox}")

    crop = img.crop((x1, y1, x2, y2))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path)
    return str(output_path)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    import re

    if text is None:
        return {"parse_error": True, "raw_response": None}

    raw = str(text).strip()
    cleaned = raw.strip()
    cleaned = re.sub(r"^```json\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    obj_start = cleaned.find("{")
    obj_end = cleaned.rfind("}")

    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        try:
            return json.loads(cleaned[obj_start:obj_end + 1])
        except Exception:
            pass

    return {"parse_error": True, "raw_response": raw}


JSON_REPAIR_PROMPT_TEMPLATE = """
The following output was supposed to be valid JSON, but it may be malformed.

Repair it into valid JSON only. Do not add new information. If an item is incomplete, omit it.

Expected schema:
{schema}

Malformed output:
{raw_output}
"""


def ensure_json_response(model, raw_text: str, schema_hint: str, allow_repair: bool = True) -> Dict[str, Any]:
    parsed = extract_json_from_text(raw_text)

    if not parsed.get("parse_error"):
        return parsed

    if not allow_repair:
        return parsed

    print("JSON parse failed. Asking model to repair output...")

    repair_prompt = JSON_REPAIR_PROMPT_TEMPLATE.format(
        schema=schema_hint,
        raw_output=str(raw_text)[:12000],
    )

    try:
        repaired_raw = model.ask_text(repair_prompt, max_new_tokens=1024)
        repaired = extract_json_from_text(repaired_raw)
        if not repaired.get("parse_error"):
            repaired["_repaired_from_malformed_json"] = True
            return repaired
        return {"parse_error": True, "raw_response": raw_text, "repair_attempt_raw": repaired_raw}
    except Exception as e:
        return {"parse_error": True, "raw_response": raw_text, "repair_error": str(e)}
