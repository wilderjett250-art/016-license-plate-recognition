"""Deterministic analysis of the plate text and visual appearance."""

from __future__ import annotations

import re
from typing import Dict

import cv2
import numpy as np


PROVINCES: Dict[str, str] = {
    "京": "北京",
    "沪": "上海",
    "津": "天津",
    "渝": "重庆",
    "冀": "河北",
    "晋": "山西",
    "蒙": "内蒙古",
    "辽": "辽宁",
    "吉": "吉林",
    "黑": "黑龙江",
    "苏": "江苏",
    "浙": "浙江",
    "皖": "安徽",
    "闽": "福建",
    "赣": "江西",
    "鲁": "山东",
    "豫": "河南",
    "鄂": "湖北",
    "湘": "湖南",
    "粤": "广东",
    "桂": "广西",
    "琼": "海南",
    "川": "四川",
    "贵": "贵州",
    "云": "云南",
    "藏": "西藏",
    "陕": "陕西",
    "甘": "甘肃",
    "青": "青海",
    "宁": "宁夏",
    "新": "新疆",
}


def normalize_plate_text(text: str) -> str:
    text = re.sub(r"[^京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新A-Z0-9]", "", text.upper())
    return text[:8]


def classify_color(crop_bgr: np.ndarray) -> tuple[str, float]:
    if crop_bgr is None or crop_bgr.size == 0:
        return "未知", 0.0
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = [x.reshape(-1) for x in cv2.split(hsv)]
    valid = s > 45
    if int(valid.sum()) == 0:
        return "未知", 0.2
    blue = ((h >= 90) & (h <= 135) & valid).mean()
    green = ((h >= 35) & (h <= 95) & valid).mean()
    yellow = ((h >= 15) & (h < 38) & valid).mean()
    black = (v < 75).mean()
    scores = {"蓝色": float(blue), "绿色": float(green), "黄色": float(yellow), "黑色": float(black)}
    color, score = max(scores.items(), key=lambda item: item[1])
    if score < 0.08:
        return "未知", round(score, 3)
    return color, round(min(score * 2.2, 1.0), 3)


def analyze_plate(
    text: str,
    crop_bgr: np.ndarray,
    detected_type: str | None = None,
) -> dict[str, object]:
    normalized = normalize_plate_text(text)
    color, color_confidence = classify_color(crop_bgr)
    province = PROVINCES.get(normalized[:1], "未知地区") if normalized else "未知地区"
    ratio = crop_bgr.shape[1] / max(crop_bgr.shape[0], 1) if crop_bgr is not None else 0

    if detected_type and detected_type != "其他":
        plate_type = detected_type
    elif color == "蓝色":
        plate_type = "普通蓝牌"
    elif color == "黄色":
        plate_type = "双层黄牌" if ratio < 3.0 else "单层黄牌"
    elif color == "绿色":
        plate_type = "新能源大型车" if ratio >= 4.8 else "新能源小型车"
    elif color == "黑色":
        plate_type = "黑色车牌"
    else:
        plate_type = "其他"

    if plate_type in {"双层黄牌", "新能源大型车", "拖拉机绿牌"}:
        vehicle_kind = "大型/特殊车辆（根据车牌类型推断）"
    elif plate_type in {"普通蓝牌", "新能源小型车"}:
        vehicle_kind = "小型车辆（根据车牌类型推断）"
    elif plate_type == "单层黄牌":
        vehicle_kind = "中型或大型车辆（根据车牌类型推断）"
    else:
        vehicle_kind = "未知车辆类型"

    return {
        "plate_number": normalized or "未可靠识别",
        "plate_color": color,
        "plate_type": plate_type,
        "region": province,
        "vehicle_kind": vehicle_kind,
        "color_confidence": color_confidence,
        "analysis_note": "地区来自车牌首字符；车辆类型依据车牌类型推断，不等同于车型识别。",
    }
