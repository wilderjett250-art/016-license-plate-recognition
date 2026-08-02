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
        self.setWindowTitle("统一车牌识别与信息分析")
        self.resize(1080, 760)
        self.image_label = QLabel("选择一张车辆图片开始识别")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(430)
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        self.open_button = QPushButton("选择图片并识别")
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

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择车辆图片", "", "Images (*.jpg *.jpeg *.png *.bmp)"
        )
        if not path:
            return
        try:
            result = self.pipeline.process(path)
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
