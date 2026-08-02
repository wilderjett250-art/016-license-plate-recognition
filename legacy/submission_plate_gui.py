import sys
import os
import cv2
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QFileDialog, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO
from main import PlateTransformer, TYPE_CLASSES, device, preprocess_plate_pil

# ========= 路径兼容打包（PyInstaller 支持） =========
def resource_path(relative_path):
    """获取资源文件路径（兼容 PyInstaller）"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

YOLO_MODEL_PATH = resource_path(os.path.join("models", "yolov11.pt"))
CRNN_MODEL_PATH = resource_path(os.path.join("models", "crnn_plate_type.pth"))
OUTPUT_DIR = resource_path("output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

class PlateApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("中文车牌识别系统")
        self.setGeometry(300, 200, 700, 500)

        self.image_label = QLabel("请选择图片")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.result_label = QLabel("识别结果：")
        self.upload_button = QPushButton("选择车牌图像")
        self.upload_button.clicked.connect(self.load_image)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.result_label)
        self.setLayout(layout)

        self.model = self.load_plate_model()
        self.yolo = YOLO(YOLO_MODEL_PATH)

    def load_plate_model(self):
        model = PlateTransformer().to(device)
        model.load_state_dict(torch.load(CRNN_MODEL_PATH, map_location=device))
        model.eval()
        return model

    def process_image(self, img_path):
        img = cv2.imread(img_path)
        filename = os.path.basename(img_path)
        results = self.yolo.predict(source=img, save=False, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)

        plate_type = "未检测到车牌"
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            plate_crop = img[y1:y2, x1:x2]
            if plate_crop.size == 0:
                continue
            input_tensor = preprocess_plate_pil(plate_crop)
            with torch.no_grad():
                type_logits = self.model(input_tensor)
                pred_type = torch.argmax(type_logits, dim=1).item()
                plate_type = TYPE_CLASSES[pred_type]

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, plate_type, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        out_path = os.path.join(OUTPUT_DIR, f"result_{filename}")
        cv2.imwrite(out_path, img)
        return out_path, plate_type

    def load_image(self):
        img_path, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Images (*.jpg *.png *.jpeg)")
        if img_path:
            out_path, plate_type = self.process_image(img_path)

            # 显示结果图像
            qimg = QImage(out_path)
            pixmap = QPixmap.fromImage(qimg).scaled(600, 300, Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.result_label.setText(f"类型识别：{plate_type}")

            # 自动打开输出目录
            os.startfile(os.path.abspath(OUTPUT_DIR))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PlateApp()
    window.show()
    sys.exit(app.exec_())
