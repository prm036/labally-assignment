"""
TrOCR wrapper for handwriting-specific OCR.
Uses microsoft/trocr-large-handwritten, which is purpose-trained on
handwritten text and significantly outperforms generic VLMs on dense
cursive/printed handwriting.

Designed to share the GPU with Qwen2.5-VL; loads in bfloat16 to minimise
VRAM usage (~4 GB on H100).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


_DEFAULT_MODEL = "microsoft/trocr-large-handwritten"


class TrOCRExtractor:
    """
    Thin wrapper around TrOCR for handwritten line OCR.

    Usage:
        trocr = TrOCRExtractor()
        text = trocr.transcribe("/path/to/crop.png")
    """

    def __init__(self,
                 model_name: str = _DEFAULT_MODEL,
                 device: Optional[str] = None):
        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[TrOCR] Loading {model_name} on {device}...")
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.model.eval()
        print("[TrOCR] Model loaded.")

    def transcribe(self, image_path: str, max_new_tokens: int = 256) -> str:
        """
        Transcribe a single crop image (should be a single text line or a
        small block of ~2-3 lines).

        Returns the decoded text string.
        """
        img = Image.open(image_path).convert("RGB")
        return self._transcribe_pil(img, max_new_tokens=max_new_tokens)

    def transcribe_pil(self, img: Image.Image, max_new_tokens: int = 256) -> str:
        return self._transcribe_pil(img, max_new_tokens=max_new_tokens)

    def _transcribe_pil(self, img: Image.Image, max_new_tokens: int = 256) -> str:
        pixel_values = self.processor(
            images=img, return_tensors="pt"
        ).pixel_values.to(self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values,
                max_new_tokens=max_new_tokens,
            )

        text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return text.strip()

    def transcribe_batch(self, image_paths: List[str],
                         max_new_tokens: int = 256,
                         batch_size: int = 8) -> List[str]:
        """
        Batch-transcribe multiple crops for efficiency.
        """
        results = []
        imgs = [Image.open(p).convert("RGB") for p in image_paths]

        for i in range(0, len(imgs), batch_size):
            batch = imgs[i:i + batch_size]
            pixel_values = self.processor(
                images=batch, return_tensors="pt", padding=True
            ).pixel_values.to(self.device, dtype=torch.bfloat16)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_new_tokens=max_new_tokens,
                )

            texts = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            results.extend([t.strip() for t in texts])

        return results
