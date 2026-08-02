from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import PlateRecognitionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="中国车牌定位、单牌 OCR 与信息分析程序"
    )
    parser.add_argument("image", help="待识别图片路径")
    parser.add_argument(
        "--output-dir", default="output", help="结果目录，默认 output"
    )
    parser.add_argument(
        "--ocr-backend",
        choices=["easyocr", "model"],
        default="easyocr",
        help="文字识别后端，默认使用 EasyOCR 中文模型",
    )
    parser.add_argument("--device", default=None, help="cuda:0 或 cpu")
    parser.add_argument(
        "--plate-index",
        type=int,
        default=None,
        help="选择第几块候选车牌（按检测置信度排序，从 1 开始；默认选择最高置信度）",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parent
    pipeline = PlateRecognitionPipeline(
        detector_path=root / "models" / "yolov11.pt",
        output_dir=args.output_dir,
        ocr_backend=args.ocr_backend,
        ocr_model_path=root / "models" / "crnn_plate_best_gru.pth",
        easyocr_model_dir=root / "models" / "easyocr",
        device=args.device,
    )
    result = pipeline.process(args.image, plate_index=args.plate_index)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
