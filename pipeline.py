"""Single-plate image -> crop -> OCR -> analysis pipeline.

The detector may return several candidate boxes, but the downstream OCR and
analysis stages intentionally receive one selected crop only. By default the
highest-confidence candidate is selected; callers can provide a one-based
``plate_index`` to select another candidate after confidence sorting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from ocr_backend import EasyOcrBackend, HyperLpr3Backend, ModelOcrBackend
from plate_analysis import analyze_plate


class PlateRecognitionPipeline:
    def __init__(
        self,
        detector_path: str,
        output_dir: str,
        ocr_backend: str = "easyocr",
        ocr_model_path: str | None = None,
        hyperlpr_model_path: str | None = None,
        easyocr_model_dir: str | None = None,
        device: str | None = None,
        stretch_ocr: bool = False,
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
        elif ocr_backend == "hyperlpr3":
            if not hyperlpr_model_path:
                raise ValueError("ocr_backend=hyperlpr3 时必须提供 hyperlpr_model_path")
            self.ocr = HyperLpr3Backend(hyperlpr_model_path, device=self.device)
        else:
            self.ocr = EasyOcrBackend(
                gpu=self.device.startswith("cuda"),
                model_dir=easyocr_model_dir,
                stretch_to_plate_shape=stretch_ocr,
            )

    def detect(self, image_path: str | Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Detect candidate plates without running OCR.

        Candidates are sorted by detector confidence and assigned a one-based
        ``rank``. This gives both the CLI and GUI a stable way to choose one
        plate before any crop is passed to the OCR model.
        """
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"无法读取图片: {image_path}")

        prediction = self.detector.predict(
            source=image, save=False, verbose=False, device=self.device
        )[0]
        boxes = prediction.boxes
        coordinates = boxes.xyxy.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()
        height, width = image.shape[:2]
        candidates: list[dict[str, Any]] = []
        for raw_index, (box, confidence) in enumerate(
            zip(coordinates, confidences), start=1
        ):
            x1, y1, x2, y2 = box.tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            candidates.append(
                {
                    "raw_index": raw_index,
                    "box": [x1, y1, x2, y2],
                    "confidence": round(float(confidence), 4),
                }
            )
        candidates.sort(key=lambda item: item["confidence"], reverse=True)
        for rank, candidate in enumerate(candidates, start=1):
            candidate["rank"] = rank
        return image, candidates

    def process(
        self, image_path: str | Path, plate_index: int | None = None
    ) -> dict[str, Any]:
        """Select one candidate, then run crop, OCR, and analysis on it only."""
        image_path = Path(image_path)
        image, candidates = self.detect(image_path)
        if plate_index is not None and plate_index < 1:
            raise ValueError("plate_index 必须是从 1 开始的正整数")

        selected: dict[str, Any] | None = None
        if candidates:
            selected_rank = plate_index or 1
            if selected_rank > len(candidates):
                raise ValueError(
                    f"plate_index={selected_rank} 超出检测候选范围（共 {len(candidates)} 块）"
                )
            selected = candidates[selected_rank - 1]

        results: list[dict[str, Any]] = []
        if selected is not None:
            x1, y1, x2, y2 = selected["box"]
            crop = image[y1:y2, x1:x2]
            crop_path = self.crop_dir / f"{image_path.stem}_plate_{selected['rank']:02d}.png"
            cv2.imwrite(str(crop_path), crop)
            text, ocr_confidence = self.ocr.recognize(crop)
            info = analyze_plate(text, crop, detected_type=None)
            results.append(
                {
                    "index": selected["rank"],
                    "box": selected["box"],
                    "detection_confidence": selected["confidence"],
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
            "detected_plate_count": len(candidates),
            "candidates": candidates,
            "selection": (
                {
                    "requested_index": plate_index,
                    "selected_index": selected["rank"],
                    "policy": (
                        "manual" if plate_index is not None else "highest_confidence"
                    ),
                }
                if selected is not None
                else None
            ),
            "plates": results,
            "annotated_image": str(annotated_path),
        }
        json_path = self.output_dir / f"{image_path.stem}_result.json"
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return payload

    @staticmethod
    def _draw_annotations(
        image: np.ndarray, results: list[dict[str, Any]], path: Path
    ) -> None:
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
            draw.rectangle(
                (x1, max(0, y1 - 30), x1 + max(260, len(label) * 22), y1),
                fill=(20, 80, 20),
            )
            draw.text(
                (x1 + 6, max(0, y1 - 28)),
                label,
                fill=(255, 255, 255),
                font=font,
            )
        canvas.save(path, quality=95)
