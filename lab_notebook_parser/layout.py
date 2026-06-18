from typing import Dict, Any, List
from PIL import Image
import cv2

from .utils import load_image_cv2, show_image


def repair_layout_regions(layout_json: Dict[str, Any], image_path_vlm: str) -> Dict[str, Any]:
    img = Image.open(image_path_vlm).convert("RGB")
    W, H = img.size

    repaired_regions = []
    seen = set()

    for idx, region in enumerate(layout_json.get("regions", []), start=1):
        region = dict(region)
        bbox = region.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        width = x2 - x1
        region["region_id"] = f"r{idx}"

        if width < 0.35 * W:
            x1 = int(0.08 * W)
            x2 = int(0.95 * W)

        pad_y = max(8, int(0.008 * H))
        y1 = max(0, y1 - pad_y)
        y2 = min(H, y2 + pad_y)

        x1 = max(0, x1)
        x2 = min(W, x2)

        if y2 - y1 < 12 or x2 - x1 < 40:
            continue

        repaired_bbox = [x1, y1, x2, y2]
        if tuple(repaired_bbox) in seen:
            continue

        seen.add(tuple(repaired_bbox))
        region["bbox"] = repaired_bbox
        repaired_regions.append(region)

    return {
        "regions": repaired_regions,
        "notes": layout_json.get("notes", []) + [
            "Layout repaired: duplicate IDs fixed; narrow boxes expanded to full writing width."
        ],
        "raw_layout_parse_error": layout_json.get("parse_error", False),
    }


def draw_vlm_regions(image_path: str, layout_regions: List[Dict[str, Any]], figsize=(10, 12), title="VLM Layout Regions"):
    img = load_image_cv2(image_path)
    vis = img.copy()

    for region in layout_regions:
        bbox = region.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        label = f"{region.get('region_id', '?')}: {region.get('type', '?')}"

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(vis, label, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    show_image(vis, title, figsize=figsize)
