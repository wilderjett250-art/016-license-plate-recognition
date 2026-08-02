"""Compatible plate recognition models used by the unified runtime.

The GRU variant matches the checkpoint shipped in the original GUI package
(`crnn_plate_type.pth`) and the later `crnn_plate_best.pth`.  The training
workspace contains a newer Transformer-decoder model; its training source is
kept under ``training/`` and is deliberately not silently mixed with the GRU
checkpoint.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


CHARS = ["<sos>", "<eos>", "<pad>", "<unk>"] + list(
    "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
) + list("ABCDEFGHJKLMNPQRSTUVWXYZ") + list("0123456789")
PLATE_TYPES = [
    "普通蓝牌",
    "单层黄牌",
    "双层黄牌",
    "黑色车牌",
    "新能源小型车",
    "新能源大型车",
    "拖拉机绿牌",
    "其他",
]
SOS_IDX = CHARS.index("<sos>")
EOS_IDX = CHARS.index("<eos>")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class PlateTransformerGRU(nn.Module):
    """The architecture that strictly matches the submitted GRU checkpoint."""

    def __init__(
        self,
        num_chars: int = len(CHARS),
        num_types: int = len(PLATE_TYPES),
        d_model: int = 256,
        nhead: int = 8,
    ) -> None:
        super().__init__()
        self.backbone = models.resnet34(weights=None)
        self.backbone.fc = nn.Identity()
        self.conv_proj = nn.Sequential(nn.Conv2d(512, d_model, 1), nn.ReLU())
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.decoder = nn.GRU(d_model, d_model, num_layers=2, batch_first=True)
        self.char_out = nn.Linear(d_model, num_chars)
        self.type_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, num_types)
        )
        self.char_embedding = nn.Embedding(
            num_embeddings=num_chars, embedding_dim=d_model, padding_idx=2
        )

    def encode_image(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone.conv1(x)
        feat = self.backbone.bn1(feat)
        feat = self.backbone.relu(feat)
        feat = self.backbone.maxpool(feat)
        feat = self.backbone.layer1(feat)
        feat = self.backbone.layer2(feat)
        feat = self.backbone.layer3(feat)
        feat = self.backbone.layer4(feat)
        type_logits = self.type_head(feat)
        sequence = self.conv_proj(feat).flatten(2).permute(0, 2, 1)
        encoded = self.encoder(self.pos_enc(sequence))
        return encoded, type_logits

    def forward(
        self, x: torch.Tensor, tgt: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor | None, torch.Tensor]:
        encoded, type_logits = self.encode_image(x)
        if tgt is None:
            return None, type_logits
        output, _ = self.decoder(self.char_embedding(tgt), None)
        return self.char_out(output), type_logits

    @torch.inference_mode()
    def predict(self, x: torch.Tensor, max_len: int = 10) -> Tuple[list[str], list[str]]:
        encoded, type_logits = self.encode_image(x)
        # The old GRU checkpoint does not expose an image-to-hidden projection.
        # Use the encoded visual context as its initial state, preserving the
        # exact parameter interface while making the runtime deterministic.
        hidden = encoded.mean(dim=1).unsqueeze(0).repeat(2, 1, 1).contiguous()
        tokens = torch.full(
            (x.size(0), 1), SOS_IDX, dtype=torch.long, device=x.device
        )
        for _ in range(max_len):
            output, hidden = self.decoder(self.char_embedding(tokens[:, -1:]), hidden)
            next_token = self.char_out(output[:, -1]).argmax(dim=1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            if bool((next_token == EOS_IDX).all()):
                break

        texts: list[str] = []
        for row in tokens.tolist():
            text = []
            for token in row[1:]:
                if token == EOS_IDX:
                    break
                if 3 < token < len(CHARS):
                    text.append(CHARS[token])
            texts.append("".join(text))
        type_ids = type_logits.argmax(dim=1).tolist()
        types = [PLATE_TYPES[i] if 0 <= i < len(PLATE_TYPES) else "其他" for i in type_ids]
        return texts, types


def load_gru_checkpoint(
    path: str, device: torch.device
) -> Tuple[PlateTransformerGRU, Dict[str, object]]:
    """Load only an exact GRU checkpoint and return a compatibility report."""

    model = PlateTransformerGRU().to(device)
    state = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"模型文件不是 state_dict: {path}")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, {
        "architecture": "PlateTransformerGRU",
        "checkpoint": path,
        "strict_load": True,
        "state_keys": len(state),
    }
