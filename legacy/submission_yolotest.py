import os
import cv2
from tkinter import filedialog, Tk
from ultralytics import YOLO

# ==============================
# 🔧 配置路径（集中配置）
# ==============================
USE_GUI = True  # 是否使用图形界面选图
MODEL_PATH = r"D:\5EDemocache\platemodel\training\plate_detection_yolo115\weights\best.pt"  # 模型路径（可修改）
OUTPUT_DIR = r"D:\5EDemocache\platemodel\test\out\0"              # 输出文件夹路径

# ==============================
# 📂 获取图片路径（支持单张或多张）
# ==============================
def select_images():
    if USE_GUI:
        root = Tk()
        root.withdraw()
        filetypes = [("图像文件", "*.jpg *.jpeg *.png")]
        return filedialog.askopenfilenames(title="选择图片", filetypes=filetypes)
    else:
        return [r"D:\path\to\your\image.jpg"]  # 手动设置路径

# ==============================
# 🧠 加载模型并预测
# ==============================
def run_detection(model_path, image_paths, save_dir):
    model = YOLO(model_path)

    os.makedirs(save_dir, exist_ok=True)

    for img_path in image_paths:
        results = model.predict(source=img_path, save=False, verbose=False)

        img_name = os.path.basename(img_path)
        rendered_img = results[0].plot()
        save_path = os.path.join(save_dir, f"label_{img_name}")
        cv2.imwrite(save_path, rendered_img)
        print(f"✅ 已保存：{save_path}")

    print(f"\n🎉 所有图片处理完毕，保存于：{save_dir}")

# ==============================
# 🚀 主程序入口
# ==============================
def main():
    print("🔍 模型路径:", MODEL_PATH)
    print("📤 输出路径:", OUTPUT_DIR)

    image_paths = select_images()
    if not image_paths:
        print("❌ 未选择任何图片，程序退出")
        return

    run_detection(MODEL_PATH, image_paths, OUTPUT_DIR)

if __name__ == "__main__":
    main()
