from __future__ import annotations

import json
import sys
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pipeline import PlateRecognitionPipeline


class PlateWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("中国车牌单牌识别与信息分析")
        self.resize(1080, 760)
        self.image_label = QLabel("选择一张车辆图片开始识别")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(430)
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.open_button = QPushButton("选择图片并识别一块车牌")
        self.open_button.clicked.connect(self.open_image)
        layout = QVBoxLayout()
        layout.addWidget(self.open_button)
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_box)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        root = Path(__file__).resolve().parent
        self.pipeline = PlateRecognitionPipeline(
            detector_path=root / "models" / "yolov11.pt",
            output_dir=root / "output",
            ocr_backend="easyocr",
            easyocr_model_dir=root / "models" / "easyocr",
        )

    def _choose_plate(self, path: str) -> tuple[int | None, bool]:
        _, candidates = self.pipeline.detect(path)
        if not candidates:
            return None, False
        if len(candidates) == 1:
            return 1, False
        options = [
            f"车牌 {candidate['rank']}（检测置信度 {candidate['confidence']:.4f}）"
            for candidate in candidates
        ]
        choice, accepted = QInputDialog.getItem(
            self,
            "选择车牌区域",
            "检测到多块候选车牌，请选择要进行 OCR 和分析的一块：",
            options,
            0,
            False,
        )
        if not accepted:
            return None, True
        return options.index(choice) + 1, False

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择车辆图片", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not path:
            return
        try:
            selected_index, canceled = self._choose_plate(path)
            if canceled:
                return
            result = self.pipeline.process(path, plate_index=selected_index)
            self.image_label.setPixmap(
                QPixmap(result["annotated_image"]).scaled(
                    self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
            self.result_box.setPlainText(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as exc:
            QMessageBox.critical(self, "识别失败", str(exc))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlateWindow()
    window.show()
    sys.exit(app.exec_())
