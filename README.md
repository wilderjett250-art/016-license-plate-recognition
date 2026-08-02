# 016 · 中国车牌单牌识别与信息分析

这是一个面向中国机动车车牌的图像识别项目。项目将车牌检测、单块车牌裁切、中文字符识别和车牌信息分析串成一条清晰的处理链，输出可复用的标注图与 JSON 结果。

```text
车辆图片
  → YOLO 检测候选车牌
  → 按置信度排序并选择一块车牌
  → 只裁切被选中的车牌
  → OCR 只读取这张裁切图
  → 分析车牌号码、颜色、类型和地区
  → 输出标注图与结构化 JSON
```

## 核心设计

检测模型负责“找到车牌”，文字模型负责“读取已经裁好的车牌”。即使一张车辆图片中检测到多块候选区域，后续流程也只会选择其中一块：

- 默认选择检测置信度最高的车牌。
- 命令行可以用 `--plate-index` 手动选择按置信度排序后的第几块车牌。
- 桌面界面检测到多块候选区域时，会先弹出选择框。
- OCR、颜色分析、类型分析、地区解析和最终标注都只针对这一个选中区域。

这样可以避免把多块车牌混在同一次识别结果中，也方便后续接入需要“单个目标确认”的业务流程。

## 当前功能

- 使用 YOLO 车牌检测模型定位中国车牌候选区域。
- 支持自动选择最高置信度车牌或手动选择候选序号。
- 只裁切选中的一块车牌，并保存到 `output/crops/`。
- OCR 默认保留原始车牌比例；可通过 `--stretch-ocr` 开启 `256×64` 固定矩形拉伸实验。
- 使用 EasyOCR 中文识别后端读取车牌字符。
- 保留原项目 GRU OCR 权重的兼容调用入口，便于后续对比实验。
- 根据裁切图像分析蓝色、黄色、绿色、黑色等车牌颜色。
- 根据首字符解析省份或地区，例如“浙”对应浙江、“粤”对应广东。
- 根据颜色、长宽比和中国车牌规则推断普通蓝牌、黄牌、新能源牌等基本类型。
- 生成只标注选中车牌的结果图和结构化 JSON。
- 提供 PyQt5 桌面界面和命令行入口。

> `vehicle_kind` 表示根据车牌规则推断出的车辆类别，不等同于车辆品牌、具体车型或车身外观识别。

## 环境安装

建议使用 Python 3.10 或 3.11，在项目根目录执行：

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

第一次使用 EasyOCR 时，会在 `models/easyocr/` 下载中文识别模型。YOLO 和 OCR 权重已经通过 Git LFS 管理。

## 命令行使用

默认识别检测置信度最高的一块车牌：

```powershell
python main.py samples\111_yello.jpg
```

手动选择按检测置信度排序后的第二块候选车牌：

```powershell
python main.py samples\111_yello.jpg --plate-index 2
```

指定 OCR 后端和设备：

```powershell
python main.py samples\111_yello.jpg --ocr-backend model --device cpu
```

启用固定矩形拉伸实验：

```powershell
python main.py samples\111_yello.jpg --stretch-ocr
```

固定拉伸对不同车牌图像的影响可能不同，因此默认关闭，建议在自己的测试集上对比后再决定是否作为默认预处理。

结果示例：

```text
output\111_yello_annotated.jpg
output\111_yello_result.json
output\crops\111_yello_plate_01.png
```

JSON 中的 `detected_plate_count` 和 `candidates` 记录检测阶段发现的候选区域；`selection` 记录最终选择策略；`plates` 始终只包含选中车牌的 OCR 和分析结果。

## 桌面界面

```powershell
python gui.py
```

选择车辆图片后，界面会先完成车牌检测。检测到多块候选车牌时，在选择框中确认一块，再运行 OCR、颜色、类型和地区分析。

## 代码结构

```text
main.py                 命令行入口与车牌序号参数
gui.py                  PyQt5 图形界面与单牌选择框
pipeline.py             检测、排序、单牌裁切、OCR、分析和输出编排
plate_analysis.py       颜色、地区、车牌类型分析
ocr_backend.py          EasyOCR 与原项目模型识别后端
plate_model.py          与原提交包权重匹配的 GRU 模型结构
models/                 YOLO 与 OCR 模型权重（Git LFS）
training/               后续训练脚本与实验材料
legacy/                 原始提交包中的运行文件
samples/                交付包中的测试图片
```

## 模型关系与兼容说明

当前运行链路使用 YOLO 检测模型加 EasyOCR 中文识别后端。`--ocr-backend model` 提供原项目 GRU 权重的兼容验证入口，权重结构由 `plate_model.py` 单独维护。

训练目录中的后续实验脚本保留为研究材料。运行版模型和训练实验分开管理，便于复现当前结果并继续迭代模型，而不会把不同结构的权重混在同一条推理链路中。

## 输出字段

每次运行的 JSON 主要包含：

- `plate_number`：识别出的中国车牌号码。
- `plate_color`：从选中车牌裁切图估计出的车牌颜色。
- `plate_type`：根据颜色和尺寸规则推断出的车牌类型。
- `region`：根据车牌首字符解析出的省份或地区。
- `vehicle_kind`：根据车牌类型推断出的车辆类别。
- `detection_confidence`、`ocr_confidence`：检测与文字识别置信度。
- `selection`：默认最高置信度或手动序号选择记录。

## English overview

This project recognizes Chinese license plates from vehicle images. YOLO detects candidate plates, one candidate is selected by confidence or by user-provided rank, and only that cropped plate is passed to OCR and downstream analysis. The output includes plate text, color, plate type, region, an annotated image, and structured JSON. A PyQt5 desktop UI and a command-line interface are included.

## Copyright and data use

The detector and OCR weights come from the original course project and subsequent local experiments. Please follow the licenses of PyTorch, Ultralytics, EasyOCR and any other dependency used by your deployment. Do not commit real vehicle images, personal information or production credentials to a public repository.
