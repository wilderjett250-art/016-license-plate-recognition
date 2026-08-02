"""OCR backends used after YOLO has cropped a plate."""

from __future__ import annotations

import math
import re
from pathlib import Path

import cv2
import numpy as np

from plate_analysis import normalize_plate_text


ALLOWED = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新ABCDEFGHJKLMNPQRSTUVWXYZ0123456789"


class EasyOcrBackend:
    name = "easyocr"

    def __init__(
        self,
        gpu: bool = False,
        model_dir: str | None = None,
        stretch_to_plate_shape: bool = False,
    ) -> None:
        import easyocr

        # The fixed-shape path is experimental. It is opt-in because a small
        # amount of geometric distortion helps some plates but hurts others.
        self.stretch_to_plate_shape = stretch_to_plate_shape
        self.input_size = (256, 64)  # width, height
        kwargs = {"gpu": gpu, "verbose": False}
        if model_dir:
            kwargs["model_storage_directory"] = model_dir
        self.reader = easyocr.Reader(["ch_sim", "en"], **kwargs)

    def recognize(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        if crop_bgr is None or crop_bgr.size == 0:
            return "", 0.0
        normalized = (
            cv2.resize(crop_bgr, self.input_size, interpolation=cv2.INTER_CUBIC)
            if self.stretch_to_plate_shape
            else crop_bgr
        )
        enlarged = cv2.resize(normalized, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
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


class HyperLpr3Backend:
    """Recognition-only adapter for the HyperLPR3 ONNX plate model.

    The existing YOLO detector remains responsible for locating a plate. This
    backend receives only the selected crop and runs HyperLPR3's CTC sequence
    recognizer, preserving the crop aspect ratio and padding it to 48x160.
    The preprocessing and decoding follow HyperLPR3's Apache-2.0 reference
    implementation without importing its package-level auto-download logic.
    """

    name = "hyperlpr3"

    _TOKENS = (
        "blank", "'", *tuple("0123456789"),
        *tuple("ABCDEFGHJKLMNPQRSTUVWXYZ"),
        "\u4e91", "\u4eac", "\u5180", "\u5409", "\u5b66", "\u5b81",
        "\u5ddd", "\u6302", "\u65b0", "\u664b", "\u6842", "\u6c11",
        "\u6caa", "\u6d25", "\u6d59", "\u6e1d", "\u6e2f", "\u6e58",
        "\u743c", "\u7518", "\u7696", "\u7ca4", "\u822a", "\u82cf",
        "\u8499", "\u85cf", "\u8b66", "\u8c6b", "\u8d35", "\u8d63",
        "\u8fbd", "\u9102", "\u95fd", "\u9655", "\u9752", "\u9c81",
        "\u9ed1", "\u9886", "\u4f7f", "\u6fb3",
    )

    def __init__(self, model_path: str, device: str | None = None) -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"HyperLPR3 OCR 模型不存在: {path}")

        session_options = ort.SessionOptions()
        session_options.log_severity_level = 3
        providers = ["CPUExecutionProvider"]
        if device and str(device).startswith("cuda"):
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(
            str(path), sess_options=session_options, providers=providers
        )
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        self.model_path = path

    @staticmethod
    def _encode(crop_bgr: np.ndarray) -> np.ndarray:
        image_height, image_width = 48, 160
        height, width = crop_bgr.shape[:2]
        ratio = width / float(height)
        resized_width = max(min(int(math.ceil(image_height * ratio)), image_width), 48)
        resized = cv2.resize(crop_bgr, (resized_width, image_height))
        resized = resized.astype("float32")
        resized = (resized.transpose((2, 0, 1)) - 127.5) / 127.5
        padded = np.zeros((3, image_height, image_width), dtype=np.float32)
        padded[:, :, :resized_width] = resized
        return np.expand_dims(padded, axis=0)

    def recognize(self, crop_bgr: np.ndarray) -> tuple[str, float]:
        if crop_bgr is None or crop_bgr.size == 0:
            return "", 0.0
        if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
            raise ValueError("HyperLPR3 OCR 需要三通道 BGR 车牌裁切图")

        output = self.session.run(
            [self.output.name], {self.input.name: self._encode(crop_bgr)}
        )[0]
        indices = np.argmax(output, axis=2)[0]
        probabilities = np.max(output, axis=2)[0]
        chars: list[str] = []
        confidences: list[float] = []
        for index, token_index in enumerate(indices):
            token_index = int(token_index)
            if token_index == 0:
                continue
            if index > 0 and token_index == int(indices[index - 1]):
                continue
            if token_index >= len(self._TOKENS):
                continue
            chars.append(self._TOKENS[token_index])
            confidences.append(float(probabilities[index]))

        text = normalize_plate_text("".join(chars))
        return text, (float(np.mean(confidences)) if confidences else 0.0)


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
