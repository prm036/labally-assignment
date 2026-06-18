"""
Stage 0: Image Pre-Processing
Deskew, adaptive contrast enhancement, ruled-line detection, and region upscaling.
Applied before any VLM or OCR model call to maximise handwriting readability.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Deskew
# ---------------------------------------------------------------------------

def _skew_angle_hough(gray: np.ndarray) -> float:
    """
    Estimate page skew in degrees using probabilistic Hough line transform.
    Returns angle in degrees (positive = clockwise tilt).
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=100,
                             minLineLength=gray.shape[1] // 4,
                             maxLineGap=20)
    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            # Keep only near-horizontal lines (notebook rules)
            if abs(angle) < 10:
                angles.append(angle)

    if not angles:
        return 0.0

    # Robust median estimate
    return float(np.median(angles))


def deskew_image(img: np.ndarray) -> np.ndarray:
    """
    Rotate img to correct skew.  Returns a new ndarray (BGR or gray).
    Only corrects if |skew| > 0.3 degrees to avoid needless resampling.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    angle = _skew_angle_hough(gray)

    if abs(angle) < 0.3:
        return img

    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                              flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


# ---------------------------------------------------------------------------
# Contrast enhancement
# ---------------------------------------------------------------------------

def enhance_contrast(img: np.ndarray,
                     clip_limit: float = 2.5,
                     tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply CLAHE to the L channel of LAB colour space.
    Preserves colour while boosting local contrast for handwriting.
    """
    if img.ndim == 2:
        # Grayscale
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_ch = clahe.apply(l_ch)
    return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def adaptive_binarize(img: np.ndarray,
                      block_size: int = 31,
                      C: int = 10) -> np.ndarray:
    """
    Sauvola-style adaptive thresholding for thin pencil strokes.
    Returns a binary (0/255) uint8 image suitable for TrOCR.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    binary = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY,
                                   block_size, C)
    return binary


# ---------------------------------------------------------------------------
# Ruled-line detection
# ---------------------------------------------------------------------------

def detect_ruled_lines(img: np.ndarray,
                       min_line_fraction: float = 0.5) -> List[int]:
    """
    Detect horizontal notebook ruling lines.

    Returns a sorted list of y-coordinates (in pixels) of each horizontal rule.
    Only lines that span at least `min_line_fraction` of the image width are kept.

    Strategy:
    1. Convert to grayscale and invert so dark rules become white.
    2. Apply a wide horizontal morphological kernel to isolate horizontal lines.
    3. Find connected components and filter by width.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape

    # Invert + threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological horizontal line extraction
    kernel_len = max(30, int(w * 0.4))
    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horiz_kernel, iterations=2)

    # Find contours of horizontal line segments
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    y_coords = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw >= min_line_fraction * w and bh <= max(5, int(h * 0.01)):
            # Mid-point of the line
            y_coords.append(y + bh // 2)

    return sorted(set(y_coords))


# ---------------------------------------------------------------------------
# Region upscaling
# ---------------------------------------------------------------------------

def upscale_region(crop: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """
    Upscale a cropped region using Lanczos resampling.
    Improves fine-detail legibility for structure drawings and small text.
    """
    if scale <= 1.0:
        return crop
    new_w = int(crop.shape[1] * scale)
    new_h = int(crop.shape[0] * scale)
    return cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def preprocess_for_vlm(image_path: str,
                        output_dir: str = "lab_parser_outputs_v4",
                        max_width: int = 1600,
                        apply_deskew: bool = True,
                        apply_contrast: bool = True) -> dict:
    """
    Full pre-processing pipeline. Returns a dict with:
      - "rgb_path":      path to contrast-enhanced RGB image (for VLM)
      - "binary_path":  path to binarized grayscale image (for TrOCR)
      - "ruled_lines":  list of y-coordinates of horizontal rules
      - "orig_size":    (width, height) of original image
      - "scale_factor": downscale ratio applied (1.0 if no resize)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    orig_h, orig_w = img_bgr.shape[:2]
    scale_factor = 1.0

    # Optionally resize to cap VLM input size while preserving aspect ratio
    if orig_w > max_width:
        scale_factor = max_width / orig_w
        new_w = max_width
        new_h = int(orig_h * scale_factor)
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Deskew
    if apply_deskew:
        img_bgr = deskew_image(img_bgr)

    # Detect ruled lines on (possibly deskewed) image
    ruled_lines = detect_ruled_lines(img_bgr)

    # Contrast enhancement
    if apply_contrast:
        img_rgb_enhanced = enhance_contrast(img_bgr)
    else:
        img_rgb_enhanced = img_bgr

    # Binary version for TrOCR
    binary = adaptive_binarize(img_bgr)

    stem = Path(image_path).stem.replace(" ", "_")
    rgb_path = output_dir / f"{stem}_vlm_input.png"
    bin_path = output_dir / f"{stem}_binary.png"

    # Save as PIL (RGB) for VLM compatibility
    Image.fromarray(cv2.cvtColor(img_rgb_enhanced, cv2.COLOR_BGR2RGB)).save(rgb_path)
    Image.fromarray(binary).save(bin_path)

    return {
        "rgb_path": str(rgb_path),
        "binary_path": str(bin_path),
        "ruled_lines": ruled_lines,
        "orig_size": (orig_w, orig_h),
        "scale_factor": scale_factor,
    }
