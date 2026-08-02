# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
import time
import json
from torch.nn.utils.rnn import pad_sequence

# 中文车牌字符集
CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
         'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# 常见车牌类型
PLATE_TYPES = ['普通蓝牌', '单层黄牌', '双层黄牌', '黑色车牌', '新能源小型车', '新能源大型车', '拖拉机绿牌', '其他']

class LabelTransformer:
    def __init__(self, chars=CHARS):
        self.chars = chars
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for i, c in enumerate(chars)}
        self.blank_idx = len(chars)

    def encode(self, s):
        return [self.char2idx[c] for c in s if c in self.char2idx]

    def decode(self, idxs, remove_repeats=True):
        chars = []
        prev = None
        for i in idxs:
            if i == self.blank_idx:
                continue
            if remove_repeats and i == prev:
                continue
            chars.append(self.idx2char[i])
            prev = i
        return ''.join(chars)

class PlateTypeTransformer:
    def __init__(self, types=PLATE_TYPES):
        self.types = types
        self.type2idx = {t: i for i, t in enumerate(types)}
        self.idx2type = {i: t for i, t in enumerate(types)}

    def encode(self, tname):
        return self.type2idx.get(tname, self.type2idx['其他'])

    def decode(self, tid):
        return self.idx2type.get(int(tid), '其他')

class PlateDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []

        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    img_path, plate, ptype = parts[0], parts[1], parts[2]
                elif len(parts) == 2:
                    img_path, plate, ptype = parts[0], parts[1], '普通蓝牌'
                else:
                    continue
                self.data.append((img_path, plate, ptype))

        self.label_transformer = LabelTransformer()
        self.type_transformer = PlateTypeTransformer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rel_path, plate, ptype = self.data[idx]
        img_path = os.path.join(self.img_dir, rel_path)

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (100, 32))

        if self.transform:
            image = self.transform(image)

        label = self.label_transformer.encode(plate)
        label_length = len(label)
        type_label = self.type_transformer.encode(ptype)

        return image, torch.tensor(label), torch.tensor(label_length), torch.tensor(type_label)

class CRNN(nn.Module):
    def __init__(self, num_chars=len(CHARS) + 1, num_types=len(PLATE_TYPES)):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d((2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, None)),
            nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.rnn = nn.GRU(256, 256, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(512, num_chars)
        self.type_classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(512, num_types)
        )

    def forward(self, x):
        x = self.cnn(x)  # B x 256 x 1 x W
        x = x.squeeze(2).permute(0, 2, 1)  # B x T x C
        rnn_out, _ = self.rnn(x)
        log_probs = F.log_softmax(self.classifier(rnn_out), dim=2)
        type_logits = self.type_classifier(rnn_out.permute(0, 2, 1))
        return log_probs, type_logits

def custom_collate(batch):
    images, labels, label_lens, type_labels = zip(*batch)
    images = torch.stack(images, dim=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=len(CHARS))
    label_lens = torch.stack(label_lens)
    type_labels = torch.stack(type_labels)
    return images, labels, label_lens, type_labels

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_dir = 'CBLPRD-330k_v1'  # 根目录
    train_file = 'CBLPRD-330k_v1/train.txt'
    val_file = 'CBLPRD-330k_v1/val.txt'

    transform = transforms.Compose([
        transforms.Resize((32, 100)),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.RandomRotation(2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    full_train_set = PlateDataset(img_dir, train_file, transform)
    train_set = Subset(full_train_set, list(range(10000)))

    full_val_set = PlateDataset(img_dir, val_file, transform)
    val_set = Subset(full_val_set, list(range(2000)))

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4, collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4, collate_fn=custom_collate)

    model = CRNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_ctc = nn.CTCLoss(blank=len(CHARS))
    criterion_type = nn.CrossEntropyLoss()

    for epoch in range(2):  # 仅训练 2 轮
        model.train()
        total_loss, correct_types = 0, 0

        for imgs, labels, label_lens, type_labels in train_loader:
            imgs, labels, label_lens, type_labels = imgs.to(device), labels.to(device), label_lens.to(device), type_labels.to(device)
            log_probs, type_logits = model(imgs)
            log_probs = log_probs.permute(1, 0, 2)
            input_lens = torch.full((imgs.size(0),), log_probs.size(0), dtype=torch.long).to(device)
            loss_ctc = criterion_ctc(log_probs, labels, input_lens, label_lens)
            loss_type = criterion_type(type_logits, type_labels)
            loss = loss_ctc + loss_type

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred_types = torch.argmax(type_logits, dim=1)
            correct_types += (pred_types == type_labels).sum().item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Type Acc: {correct_types/len(train_set):.4f}")

    torch.save(model.state_dict(), 'crnn_plate_type.pth')
    with open('plate_chars.json', 'w', encoding='utf-8') as f:
        json.dump(CHARS, f, ensure_ascii=False)
    with open('plate_types.json', 'w', encoding='utf-8') as f:
        json.dump(PLATE_TYPES, f, ensure_ascii=False)

if __name__ == '__main__':
    train()
