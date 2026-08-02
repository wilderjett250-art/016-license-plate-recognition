"""Unified image -> plate boxes -> crops -> OCR -> analysis pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from plate_analysis import analyze_plate
from ocr_backend import EasyOcrBackend, ModelOcrBackend


class PlateRecognitionPipeline:
    def __init__(
        self,
        detector_path: str,
        output_dir: str,
        ocr_backend: str = "easyocr",
        ocr_model_path: str | None = None,
        easyocr_model_dir: str | None = None,
        device: str | None = None,
    ) -> None:
        self.detector_path = Path(detector_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crop_dir = self.output_dir / "crops"
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.detector = YOLO(str(self.detector_path))
        if ocr_backend == "model":
            if not ocr_model_path:
                raise ValueError("ocr_backend=model 时必须提供 ocr_model_path")
            self.ocr = ModelOcrBackend(ocr_model_path, torch.device(self.device))
        else:
            self.ocr = EasyOcrBackend(
                gpu=self.device.startswith("cuda"), model_dir=easyocr_model_dir
            )

    def process(self, image_path: str | Path) -> dict[str, Any]:
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")

        prediction = self.detector.predict(
            source=image, save=False, verbose=False, device=self.device
        )[0]
        results: list[dict[str, Any]] = []
        boxes = prediction.boxes
        for index, box in enumerate(boxes.xyxy.cpu().numpy().astype(int), start=1):
            height, width = image.shape[:2]
            x1, y1, x2, y2 = box.tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = image[y1:y2, x1:x2]
            crop_path = self.crop_dir / f"{image_path.stem}_plate_{index:02d}.png"
            cv2.imwrite(str(crop_path), crop)
            text, ocr_confidence = self.ocr.recognize(crop)
            detected_type = None
            info = analyze_plate(text, crop, detected_type)
            results.append(
                {
                    "index": index,
                    "box": [x1, y1, x2, y2],
                    "detection_confidence": round(float(boxes.conf[index - 1].cpu()), 4),
                    "ocr_backend": self.ocr.name,
                    "ocr_confidence": round(float(ocr_confidence), 4),
                    "crop": str(crop_path),
                    **info,
                }
            )

        annotated_path = self.output_dir / f"{image_path.stem}_annotated.jpg"
        self._draw_annotations(image, results, annotated_path)
        payload = {
            "source": str(image_path),
            "detector": str(self.detector_path),
            "device": self.device,
            "plates": results,
            "annotated_image": str(annotated_path),
        }
        json_path = self.output_dir / f"{image_path.stem}_result.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    @staticmethod
    def _draw_annotations(image: np.ndarray, results: list[dict[str, Any]], path: Path) -> None:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        canvas = Image.fromarray(rgb)
        draw = ImageDraw.Draw(canvas)
        font_candidates = [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
            r"C:\Windows\Fonts\arial.ttf",
        ]
        font_path = next((p for p in font_candidates if Path(p).exists()), None)
        font = ImageFont.truetype(font_path, 22) if font_path else ImageFont.load_default()
        for item in results:
            x1, y1, x2, y2 = item["box"]
            draw.rectangle((x1, y1, x2, y2), outline=(30, 220, 70), width=3)
            label = f"{item['plate_number']} | {item['plate_color']} | {item['region']}"
            draw.rectangle((x1, max(0, y1 - 30), x1 + max(260, len(label) * 22), y1), fill=(20, 80, 20))
            draw.text((x1 + 6, max(0, y1 - 28)), label, fill=(255, 255, 255), font=font)
        canvas.save(path, quality=95)
