# 016 · 中国车牌识别与信息分析

这是一个面向中国机动车车牌的图片识别项目，针对中文省份简称、字母和数字
组成的中国车牌进行定位、裁切、文字识别与信息分析：

```text
车辆图片 → YOLO 定位车牌 → 裁切车牌 → 中文文字识别 → 颜色/类型/地区分析 → 图片与 JSON 结果
```

## 当前功能

- 使用 YOLO 车牌检测模型定位一张或多张车牌。
- 自动裁切每块车牌，并保存到 `output/crops/`。
- 使用 EasyOCR 中文识别后端读取中国车牌文字。
- 支持中国车牌省份简称、英文字母和数字组成的字符集。
- 根据中国车牌图像颜色分析蓝色、黄色、绿色、黑色等颜色。
- 根据中国车牌首字符解析省份/地区，例如“浙”对应浙江、“粤”对应广东。
- 根据颜色、长宽比和中国车牌分类规则推断普通蓝牌、黄牌、新能源牌等基本类型。
- 生成标注图和结构化 JSON，便于 GUI、接口或后续业务继续使用。
- 提供 PyQt5 桌面界面和命令行入口。

“车辆类型”字段是根据中国车牌类型推断的车辆类别，不等同于对车辆品牌、车型或车身外观进行识别。

## 运行

建议使用 Python 3.10/3.11，并在项目根目录安装依赖：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

第一次使用 EasyOCR 时，会在 `models/easyocr/` 下载中文识别模型。

命令行识别一张图片：

```powershell
python main.py samples\111_yello.jpg
```

结果包括：

```text
output\111_yello_annotated.jpg
output\111_yello_result.json
output\crops\111_yello_plate_01.png
```

启动桌面界面：

```powershell
python gui.py
```

## 代码结构

```text
main.py                 命令行入口
gui.py                  PyQt5 图形界面
pipeline.py             定位、裁切、识别、分析和输出编排
plate_analysis.py       颜色、地区、车牌类型分析
ocr_backend.py          EasyOCR 与原项目模型识别后端
plate_model.py          与原提交包权重严格匹配的 GRU 模型结构
models/                 YOLO 和原项目模型权重
training/               后续训练版 newmodel.py 及实验脚本
legacy/                 原提交包的 main.py、plate_gui.py、yolotest.py
samples/                交付包中的测试图片
```

## 关于两个识别模型

提交包中的 `crnn_plate_type_submission.pth` 和后续实验得到的
`crnn_plate_best_gru.pth` 都能严格加载到 `plate_model.py` 的 GRU 结构；
后续 `training/newmodel.py` 改成了 Transformer 解码器，不能直接加载这两个
GRU 权重。训练脚本和运行版已经分开保存，避免混用权重后产生无法解释的结果。

默认运行使用 EasyOCR 完成文字识别；`--ocr-backend model` 可用于验证原项目
GRU 权重的兼容接口：

```powershell
python main.py samples\111_yello.jpg --ocr-backend model --device cpu
```

## 输出说明

每块车牌的 JSON 结果包含：

- `plate_number`：识别出的车牌号码。
- `plate_color`：从裁切图像颜色估计的车牌颜色。
- `plate_type`：根据颜色和尺寸推断的车牌类型。
- `region`：根据车牌首字符解析出的省份/地区。
- `vehicle_kind`：基于车牌类型推断的车辆类别。
- `detection_confidence`、`ocr_confidence`：检测与文字识别的置信度。

## 版权与数据

项目中的 YOLO、PyTorch 权重来自原始课程设计包和后续本地训练目录；训练脚本
保留为可复现实验资料。使用第三方 EasyOCR 时，请同时遵守其依赖和模型的许可
条款。请不要把真实车辆图片、个人信息或生产密钥提交到公开仓库。
