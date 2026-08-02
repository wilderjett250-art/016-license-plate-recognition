import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from PIL import Image

# ===================== 配置 =====================
YOLO_MODEL_PATH = os.path.join("models", "yolov11.pt")  # YOLO 模型路径
CRNN_MODEL_PATH = os.path.join("models", "crnn_plate_type.pth")  # 车牌类型识别模型路径
OUTPUT_DIR = "output"  # 结果输出目录
IMAGE_DIR = "images"  # 输入图片目录
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 设备选择

# 车牌类型类别映射
TYPE_CLASSES = ['普通蓝牌', '单层黄牌', '双层黄牌', '黑色车牌', '新能源小型车', '新能源大型车', '拖拉机绿牌', '其他']
idx2type = {i: t for i, t in enumerate(TYPE_CLASSES)}

# ===================== 模型结构 =====================
class PositionalEncoding(nn.Module):
    """位置编码模块，为Transformer提供位置信息"""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class PlateTransformer(nn.Module):
    """车牌类型识别模型，结合ResNet和Transformer相关结构"""
    def __init__(self, num_chars=69, num_types=8, d_model=256, nhead=8):
        super().__init__()
        self.backbone = models.resnet34(weights=None)  # ResNet34 骨干网络
        self.backbone.fc = nn.Identity()  # 移除原全连接层
        self.conv_proj = nn.Sequential(nn.Conv2d(512, d_model, 1), nn.ReLU())  # 特征投影
        self.pos_enc = PositionalEncoding(d_model)  # 位置编码
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)  # Transformer编码器
        self.decoder = nn.GRU(d_model, d_model, num_layers=2, batch_first=True)  # GRU解码器（这里可根据实际需求调整）
        self.char_out = nn.Linear(d_model, num_chars)  # 字符识别输出（当前主要用类型识别，可按需完善）
        self.type_head = nn.Sequential(  # 车牌类型识别头
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_types)
        )
        self.char_embedding = nn.Embedding(num_embeddings=num_chars, embedding_dim=d_model, padding_idx=2)  # 字符嵌入

    def forward(self, x, tgt=None):
        """前向传播，主要用于获取车牌类型预测结果"""
        feat = self.backbone.conv1(x)
        feat = self.backbone.bn1(feat)
        feat = self.backbone.relu(feat)
        feat = self.backbone.maxpool(feat)
        feat = self.backbone.layer1(feat)
        feat = self.backbone.layer2(feat)
        feat = self.backbone.layer3(feat)
        feat = self.backbone.layer4(feat)
        type_logits = self.type_head(feat)  # 获取类型预测逻辑
        return type_logits

# ===================== 图像预处理（统一化） =====================
transform = transforms.Compose([
    transforms.Resize((64, 256)),  # 调整尺寸
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize([0.5] * 3, [0.5] * 3)  # 归一化
])

def preprocess_plate_pil(img_bgr):
    """将OpenCV的BGR图像转换为模型可处理的张量"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # BGR转RGB
    pil_img = Image.fromarray(img_rgb)  # 转换为PIL Image
    img_tensor = transform(pil_img).unsqueeze(0)  # 转换为张量并添加batch维度
    return img_tensor.to(device)  # 移动到指定设备

# ===================== 主流程 =====================
def run_pipeline():
    """执行整个车牌识别流程：检测、分类、标注"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)  # 创建输出目录
    # 初始化车牌类型识别模型
    model = PlateTransformer().to(device)
    model.load_state_dict(torch.load(CRNN_MODEL_PATH, map_location=device))
    model.eval()  # 设为评估模式

    # 初始化YOLO目标检测模型
    yolo_model = YOLO(YOLO_MODEL_PATH)

    # 遍历图像目录处理每张图
    for img_name in os.listdir(IMAGE_DIR):
        # 过滤非图片文件
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = cv2.imread(img_path)  # 读取图像
        # 使用YOLO检测车牌
        results = yolo_model.predict(source=img, save=False, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy().astype(int)  # 获取检测框坐标

        # 处理每个检测到的车牌
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # 裁剪车牌区域
            plate_crop = img[y1:y2, x1:x2]
            if plate_crop.size == 0:  # 裁剪区域为空则跳过
                continue

            # 预处理并预测车牌类型
            input_tensor = preprocess_plate_pil(plate_crop)
            with torch.no_grad():  # 推理阶段不需要计算梯度
                type_logits = model(input_tensor)
                pred_type = torch.argmax(type_logits, dim=1).item()  # 获取类型预测结果
                plate_type = idx2type.get(pred_type, '未知')  # 映射为类型名称

            # 在原图上标注车牌类型
            print(f"📷 {img_name} - Plate {i+1}: 类型 = {plate_type}")
            # 绘制检测框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制类型文本，位置在检测框上方
            cv2.putText(img, plate_type, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 尝试移除车牌上方的绿色问号（假设是固定位置或可简单处理的干扰，
            # 若实际场景复杂，建议在预处理时用图像修复等更完善方法）
            # 这里简单示例：如果检测到问号区域（假设在车牌上方固定区域），用周围像素填充
            # 以下为示例代码，需根据实际问号位置调整，若没有实际问号可删除这部分
            # 假设问号在车牌上方一定区域，这里简单将该区域用黑色或周围颜色覆盖
            # 比如：y1 - 30 到 y1，x1 到 x2 区域（根据实际调整）
            # 可根据实际情况精细调整坐标或采用更智能的修复方法
            # 以下为示例，若不需要可删除
            # repair_y1 = max(0, y1 - 30)
            # repair_y2 = y1
            # if repair_y2 > repair_y1:
            #     img[repair_y1:repair_y2, x1:x2] = 0  # 用黑色覆盖，可改为更合理的填充

        # 保存标注后的图像
        out_path = os.path.join(OUTPUT_DIR, f"result_{img_name}")
        cv2.imwrite(out_path, img)

    print("🎉 全部完成，结果保存在 output/ 目录。")

# ===================== 程序入口 =====================
if __name__ == "__main__":
    run_pipeline()