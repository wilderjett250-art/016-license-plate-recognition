"""OCR backends used after YOLO has cropped a plate."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from plate_analysis import normalize_plate_text


ALLOWED = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"


class EasyOcrBackend:
    name = "easyocr"

    def __init__(self, gpu: bool = False, model_dir: str | None = None) -> None:
        import easyocr

        kwargs = {"gpu": gpu, "verbose": False}
        if model_dir:
            kwargs["model_storage_directory"] = model_dir
        self.reader = easyocr.Reader(["ch_sim", "en"], **kwargs)

    def recognize(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        if crop_bgr is None or crop_bgr.size == 0:
            return "", 0.0
        enlarged = cv2.resize(crop_bgr, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        results = self.reader.readtext(
            enlarged,
            detail=1,
            paragraph=False,
            allowlist=ALLOWED,
            mag_ratio=1.0,
        )
        candidates = []
        for _, raw, confidence in results:
            cleaned = normalize_plate_text(re.sub(r"[^" + ALLOWED + "]", "", raw.upper()))
            if cleaned:
                candidates.append((cleaned, float(confidence)))
        if not candidates:
            return "", 0.0
        return max(candidates, key=lambda item: (len(item[0]), item[1]))


class ModelOcrBackend:
    name = "gru-checkpoint"

    def __init__(self, checkpoint: str, device) -> None:
        import torch
        from torchvision import transforms
        from PIL import Image
        from plate_model import load_gru_checkpoint

        self.torch = torch
        self.Image = Image
        self.transform = transforms.Compose(
            [
                transforms.Resize((64, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        self.device = device
        self.model, self.report = load_gru_checkpoint(checkpoint, device)

    def recognize(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(self.Image.fromarray(rgb)).unsqueeze(0).to(self.device)
        texts, _ = self.model.predict(tensor)
        return (texts[0], 0.35 if texts[0] else 0.0)
