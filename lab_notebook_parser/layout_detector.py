"""
Stage 1: Layout Detection — OpenCV ruled-line segmentation
Replaces the fragile Qwen2.5-VL bbox-prediction layout pass.

Strategy:
  1. Use detected ruled lines to define row bands.
  2. Within each band, use projection profiles to find sub-regions (text vs blank).
  3. Classify each region: text | table | structure_drawing | formula | heading.
  4. Optionally merge with a lightweight VLM region pass for difficult regions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

REGION_TYPES = {
    "heading", "text", "table", "table_cell",
    "structure_drawing", "formula", "observation", "unknown"
}


@dataclass
class LayoutRegion:
    region_id: str
    type: str          # one of REGION_TYPES
    bbox: List[int]    # [x1, y1, x2, y2] in scaled image coords
    description: str = ""
    confidence: float = 0.85
    source: str = "opencv"  # "opencv" | "vlm" | "hybrid"
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _horizontal_projection(gray: np.ndarray) -> np.ndarray:
    """Sum of dark pixel counts per row (after inversion)."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary.sum(axis=1).astype(np.float32)


def _vertical_projection(gray: np.ndarray) -> np.ndarray:
    """Sum of dark pixel counts per column."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary.sum(axis=0).astype(np.float32)


def _split_by_gaps(projection: np.ndarray,
                   threshold: float,
                   min_run: int = 5) -> List[Tuple[int, int]]:
    """
    Split a projection profile into non-empty segments.
    Returns list of (start, end) index tuples where projection > threshold.
    """
    active = projection > threshold
    segments = []
    in_seg = False
    start = 0
    for i, a in enumerate(active):
        if a and not in_seg:
            start = i
            in_seg = True
        elif not a and in_seg:
            if i - start >= min_run:
                segments.append((start, i))
            in_seg = False
    if in_seg and len(projection) - start >= min_run:
        segments.append((start, len(projection)))
    return segments


def _classify_region(crop_gray: np.ndarray) -> str:
    """
    Heuristic classifier for a cropped region.

    Rules (applied in order):
      - High aspect ratio (wide, short): likely a single text line
      - Has many isolated small connected components arranged in a grid: table
      - Contains large circular/ring structures with few text pixels: structure_drawing
      - Very dense dark pixels with many thin strokes: formula
      - Default: text
    """
    h, w = crop_gray.shape

    # Too small to classify meaningfully
    if h < 8 or w < 20:
        return "unknown"

    _, binary = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    density = binary.sum() / (255 * h * w)

    # Structure drawing: moderate density, significant circular features
    circles = cv2.HoughCircles(crop_gray, cv2.HOUGH_GRADIENT, dp=1,
                                minDist=10, param1=50, param2=25,
                                minRadius=8, maxRadius=min(h, w) // 3)
    if circles is not None and len(circles[0]) >= 2 and density < 0.25:
        return "structure_drawing"

    # Table: detect vertical lines (column separators)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h // 3)))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)
    v_density = v_lines.sum() / (255 * h * w) if h * w > 0 else 0

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w // 3), 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
    h_density = h_lines.sum() / (255 * h * w) if h * w > 0 else 0

    if v_density > 0.01 and h_density > 0.005:
        return "table"

    # Formula: high density with lots of subscript/superscript vertical variation
    vert_proj = _vertical_projection(crop_gray)
    vert_variation = np.std(vert_proj) / (np.mean(vert_proj) + 1e-6)
    if density > 0.08 and vert_variation > 1.5 and h > 40:
        return "formula"

    return "text"


# ---------------------------------------------------------------------------
# Core layout detection
# ---------------------------------------------------------------------------

def detect_layout_from_ruled_lines(img_bgr: np.ndarray,
                                   ruled_lines: List[int],
                                   margin_left_frac: float = 0.08,
                                   margin_right_frac: float = 0.97,
                                   min_row_height: int = 14,
                                   padding: int = 4) -> List[LayoutRegion]:
    """
    Given an image and detected ruled-line y-coordinates, produce layout regions.

    Each pair of consecutive ruled lines defines a "row band".
    Within each band, horizontal projection profiles find active sub-segments.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Ensure sentinel lines at top and bottom
    ys = sorted(set([0] + ruled_lines + [h]))

    # Content region x-limits
    x1_content = int(margin_left_frac * w)
    x2_content = int(margin_right_frac * w)

    regions: List[LayoutRegion] = []
    rid = 0

    for band_idx in range(len(ys) - 1):
        y_top = ys[band_idx]
        y_bot = ys[band_idx + 1]

        if y_bot - y_top < min_row_height:
            continue

        # Crop the full-width band
        band = gray[y_top:y_bot, x1_content:x2_content]

        # Horizontal projection to find content rows within this band
        horiz_proj = _horizontal_projection(band)
        threshold = max(5.0, float(np.percentile(horiz_proj, 60)))
        row_segs = _split_by_gaps(horiz_proj, threshold, min_run=min_row_height // 2)

        if not row_segs:
            continue

        for row_start, row_end in row_segs:
            abs_y1 = max(0, y_top + row_start - padding)
            abs_y2 = min(h, y_top + row_end + padding)

            if abs_y2 - abs_y1 < min_row_height:
                continue

            # Vertical projection within this row to trim horizontal extent
            row_band = gray[abs_y1:abs_y2, x1_content:x2_content]
            vert_proj = _vertical_projection(row_band)
            v_threshold = max(2.0, float(np.percentile(vert_proj, 50)))
            col_segs = _split_by_gaps(vert_proj, v_threshold, min_run=10)

            if not col_segs:
                col_segs = [(0, x2_content - x1_content)]

            for col_start, col_end in col_segs:
                abs_x1 = max(0, x1_content + col_start - padding)
                abs_x2 = min(w, x1_content + col_end + padding)

                if abs_x2 - abs_x1 < 30:
                    continue

                crop = gray[abs_y1:abs_y2, abs_x1:abs_x2]
                rtype = _classify_region(crop)

                rid += 1
                regions.append(LayoutRegion(
                    region_id=f"r{rid}",
                    type=rtype,
                    bbox=[abs_x1, abs_y1, abs_x2, abs_y2],
                    description="",
                    source="opencv",
                ))

    return regions


def _merge_overlapping_regions(regions: List[LayoutRegion],
                               iou_threshold: float = 0.5) -> List[LayoutRegion]:
    """
    Merge pairs of regions with high IoU (duplicate detection from two sources).
    Prefers VLM-sourced regions over opencv when merging.
    """
    merged: List[LayoutRegion] = []
    used = [False] * len(regions)

    def iou(a: LayoutRegion, b: LayoutRegion) -> float:
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / (area_a + area_b - inter)

    for i, r in enumerate(regions):
        if used[i]:
            continue
        group = [r]
        for j in range(i + 1, len(regions)):
            if not used[j] and iou(r, regions[j]) >= iou_threshold:
                group.append(regions[j])
                used[j] = True
        # Pick the best in the group: prefer vlm type annotation
        best = max(group, key=lambda x: (x.source == "vlm", x.confidence))
        merged.append(best)
        used[i] = True

    return merged


def merge_opencv_and_vlm_regions(opencv_regions: List[LayoutRegion],
                                 vlm_regions: List[Dict[str, Any]],
                                 img_size: Tuple[int, int]) -> List[LayoutRegion]:
    """
    Combine OpenCV regions with VLM regions.
    VLM regions are used to:
      - override the region type if confidence is higher
      - add description text
      - add structure_drawing / reaction_scheme regions that OpenCV may miss
    """
    w, h = img_size
    combined: List[LayoutRegion] = list(opencv_regions)

    for idx, vreg in enumerate(vlm_regions):
        bbox = vreg.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        rtype = vreg.get("type", "unknown")
        if rtype not in REGION_TYPES:
            rtype = "unknown"

        # Only add VLM region if it's for a type that OpenCV typically misses
        if rtype in {"structure_drawing", "reaction_scheme", "molecular_structure", "table"}:
            combined.append(LayoutRegion(
                region_id=f"vlm_{idx+1}",
                type=rtype,
                bbox=[max(0, x1), max(0, y1), min(w, x2), min(h, y2)],
                description=vreg.get("description", ""),
                confidence=float(vreg.get("confidence", 0.7)),
                source="vlm",
            ))

    combined.sort(key=lambda r: r.bbox[1])  # sort top-to-bottom
    merged = _merge_overlapping_regions(combined, iou_threshold=0.4)

    # Re-number
    for i, r in enumerate(merged, start=1):
        r.region_id = f"r{i}"

    return merged


def draw_layout_regions(img_bgr: np.ndarray,
                        regions: List[LayoutRegion],
                        output_path: Optional[str] = None) -> np.ndarray:
    """Draw bounding boxes and labels on a copy of the image."""
    COLOR_MAP = {
        "text": (0, 200, 0),
        "heading": (0, 0, 255),
        "table": (255, 165, 0),
        "table_cell": (200, 200, 0),
        "structure_drawing": (255, 0, 255),
        "formula": (0, 255, 255),
        "observation": (100, 149, 237),
        "unknown": (128, 128, 128),
    }
    vis = img_bgr.copy()
    for r in regions:
        x1, y1, x2, y2 = r.bbox
        color = COLOR_MAP.get(r.type, (128, 128, 128))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{r.region_id}:{r.type[:4]}"
        cv2.putText(vis, label, (x1, max(14, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    if output_path:
        cv2.imwrite(str(output_path), vis)

    return vis
