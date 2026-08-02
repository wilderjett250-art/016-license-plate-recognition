from ultralytics import YOLO
from datetime import datetime


def train_yolov8_detector(
        data_config=r"D:\5EDemocache\platemodel\training\ccpd.yaml",  # 数据集配置文件路径（需手动创建）
        model_size="yolov8m.pt",  # 可选: yolov8n.pt/yolov8m.pt等
        epochs=220,
        batch=16,
        imgsz=640,
        device=0,  # 0表示使用GPU，-1表示CPU
        project="path3/detect",
        name="ccpd_plate_detection"
):
    """训练YOLOv8车牌检测模型（直接使用已配置的ccpd.yaml）"""
    # 加载预训练模型
    model = YOLO(model_size)

    # 打印训练配置
    print(f"[训练配置]")
    print(f"模型版本: {model_size}")
    print(f"数据集配置: {data_config}")
    print(f"训练轮次: {epochs}, 批次大小: {batch}")
    print(f"输入尺寸: {imgsz}x{imgsz}, 设备: {'GPU' if device >= 0 else 'CPU'}")

    # 开始训练，移除`hyp`字典，改用官方支持的参数
    results = model.train(
        data=data_config,  # 直接指定配置文件路径
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        exist_ok=True,  # 覆盖已存在的训练目录
        pretrained=True,  # 启用预训练权重
        save_period=10,  # 每10轮保存一次检查点
        verbose=True,  # 打印详细训练日志
        amp=True,  # 开启混合精度加速（FP16计算）
        half=True,  # 模型参数采用FP16加载（需GPU支持，4060可兼容）
        cache="disk",  # 缓存到磁盘（可选"ram"或"disk"，默认False）
        workers=8,  # 根据CPU核心数设置合理并行数
        # 以下为原`hyp`中的参数，改为直接传递官方支持的参数名
        lr0=0.01,       # 初始学习率
        lrf=0.0001,     # 最终学习率（余弦退火终点）
        cos_lr=True,    # 启用余弦退火学习率调度
        warmup_epochs=2, # 热身轮次
    )

    # 返回最佳模型路径
    best_model_path = f"{project}/{name}/weights/best.pt"
    print(f"\n[训练完成] 最佳模型保存至: {best_model_path}")
    return best_model_path


def validate_model(model_path, data_config=r"D:\5EDemocache\platemodel\training\ccpd.yaml", save_path="validation_results.txt"):
    """验证模型性能"""
    model = YOLO(model_path)
    print("\n[开始验证]")
    results = model.val(
        data=data_config,  # 使用验证集配置
        imgsz=640,
        device=0,
        verbose=True
    )

    # 提取关键指标
    map50 = results.box.map50  # mAP@0.5
    map50_95 = results.box.map  # mAP@0.5:0.95
    print(f"[验证结果]")
    print(f"mAP@0.5: {map50:.4f}")
    print(f"mAP@0.5:0.95: {map50_95:.4f}")

    # 将验证结果保存到文件
    with open(save_path, 'w') as f:
        f.write(f"[验证结果]\n")
        f.write(f"mAP@0.5: {map50:.4f}\n")
        f.write(f"mAP@0.5:0.95: {map50_95:.4f}\n")

    return results


def main():
    # 手动指定已存在的ccpd.yaml路径（需与训练脚本在同一目录或写绝对路径）
    ccpd_config_path = r"D:\5EDemocache\platemodel\training/ccpd.yaml"

    # 指定训练结果保存路径
    train_project_path = "path3/to/save/train_results"
    train_name = "ccpd_plate_detection1"

    # 训练模型
    best_model = train_yolov8_detector(
        data_config=ccpd_config_path,
        model_size="yolov8m.pt",  # 示例使用v8m，可按需换v8s等
        epochs=220,
        batch=16,  # 若显存不足可降低
        project=train_project_path,
        name=train_name
    )

    # 指定验证结果保存路径
    validation_save_path = f"{train_project_path}/{train_name}/validation_results.txt"

    # 验证模型（可选）
    validate_model(best_model, ccpd_config_path, validation_save_path)


if __name__ == "__main__":
    main()