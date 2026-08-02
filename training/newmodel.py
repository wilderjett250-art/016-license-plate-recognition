# -*- coding: utf-8 -*-
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 中文车牌字符集 + 特殊符号
CHARS = ['<sos>', '<eos>', '<pad>', '<unk>'] + [
    '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
    '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
    '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁', '新',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
    'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN = '<pad>', '<unk>', '<sos>', '<eos>'
PAD_IDX = CHARS.index('<pad>')
SOS_IDX = CHARS.index('<sos>')
EOS_IDX = CHARS.index('<eos>')

PLATE_TYPES = ['普通蓝牌', '单层黄牌', '双层黄牌', '黑色车牌', '新能源小型车', '新能源大型车', '拖拉机绿牌', '其他']


class LabelTransformer:
    def __init__(self, chars=CHARS):
        self.chars = chars
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for i, c in enumerate(chars)}

    def encode(self, s):
        return [self.char2idx.get('<sos>')] + [self.char2idx.get(c, self.char2idx['<unk>']) for c in s] + [
            self.char2idx.get('<eos>')]

    def decode(self, idxs):
        # 过滤掉SOS、EOS和PAD token，并转换为字符串
        return ''.join([self.idx2char.get(i, '<unk>') for i in idxs if i > 3 and i < len(self.chars)])

    def decode_batch(self, batch_idxs):
        # 批量解码
        return [self.decode(idxs) for idxs in batch_idxs]


class PlateTypeTransformer:
    def __init__(self, types=PLATE_TYPES):
        self.types = types
        self.type2idx = {t: i for i, t in enumerate(types)}
        self.idx2type = {i: t for i, t in enumerate(types)}

    def encode(self, tname):
        return self.type2idx.get(tname, self.type2idx['其他'])

    def decode(self, tid):
        return self.idx2type.get(int(tid), '其他')

    def decode_batch(self, batch_tid):
        return [self.decode(tid) for tid in batch_tid]


class PlateDataset(Dataset):
    def __init__(self, img_dir, label_file, transform=None, is_train=True):
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train
        self.data = []
        # 过滤不存在的图片
        self.valid_indices = []
        for i, (rel_path, _, _) in enumerate(self.data):
            img_path = os.path.join(self.img_dir, rel_path)
            if os.path.exists(img_path):
                self.valid_indices.append(i)
            else:
                print(f"警告: 图像 {img_path} 不存在，已过滤")

        print(f"原始样本数: {len(self.data)}")
        print(f"有效样本数: {len(self.valid_indices)}")

        # 读取标签文件并打印路径信息（用于调试）
        with open(label_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"加载标签文件: {label_file}")
            print(f"样本数: {len(lines)}")

            # 打印前10条路径，检查格式
            print("前10条路径示例:")
            for line in lines[:10]:
                parts = line.strip().split()
                if len(parts) >= 2:
                    print(f"  {parts[0]}")

            # 处理所有路径
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    rel_path = os.path.normpath(parts[0])  # 规范化相对路径
                    plate = parts[1]
                    ptype = parts[2] if len(parts) >= 3 else '普通蓝牌'
                    self.data.append((rel_path, plate, ptype))

        self.label_transformer = LabelTransformer()
        self.type_transformer = PlateTypeTransformer()

        self._char_freq = {}
        for _, plate, _ in self.data:
            for c in plate:
                self._char_freq[c] = self._char_freq.get(c, 0) + 1



    def analyze_data(self):
        print(f"数据集大小: {len(self.data)}")
        sorted_chars = sorted(self._char_freq.items(), key=lambda x: x[1])
        print("出现频率最低的10个字符:")
        for char, freq in sorted_chars[:10]:
            print(f"  {char}: {freq}次")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # 使用有效索引
        true_idx = self.valid_indices[idx]
        rel_path, plate, ptype = self.data[true_idx]
        img_path = os.path.join(self.img_dir, rel_path)

        with open(img_path, 'rb') as f:
            image = Image.open(f).convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.label_transformer.encode(plate))
        type_label = torch.tensor(self.type_transformer.encode(ptype))

        return image, label, type_label, plate, ptype



def custom_collate(batch):
    """自定义数据收集函数，处理可变长度序列"""
    images, labels, types, plates, ptypes = zip(*batch)
    images = torch.stack(images)
    labels = pad_sequence(labels, batch_first=True, padding_value=PAD_IDX)
    types = torch.tensor(types)

    return images, labels, types, plates, ptypes


class PositionalEncoding(nn.Module):
    """位置编码，为模型提供序列位置信息"""

    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PlateTransformer(nn.Module):
    """车牌识别模型，结合CNN特征提取和Transformer序列处理"""

    def __init__(self, num_chars=len(CHARS), num_types=len(PLATE_TYPES), d_model=256, nhead=8):
        super().__init__()

        # 特征提取骨干网络
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        # 移除最后的全连接层
        self.backbone.fc = nn.Identity()

        # 特征投影层，将CNN特征映射到Transformer的维度
        self.conv_proj = nn.Sequential(
            nn.Conv2d(512, d_model, 1),
            nn.ReLU()
        )

        # 位置编码
        self.pos_enc = PositionalEncoding(d_model)

        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Transformer解码器层（替换原来的GRU）
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)

        # 字符分类头
        self.char_out = nn.Linear(d_model, num_chars)

        # 车牌类型分类头
        self.type_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(), nn.Linear(512, num_types)
        )

        # 字符嵌入层
        self.char_embedding = nn.Embedding(num_embeddings=len(CHARS), embedding_dim=d_model, padding_idx=PAD_IDX)

        # 解码器位置编码
        self.dec_pos_enc = PositionalEncoding(d_model)

        # 保存标签转换器
        self.label_transformer = LabelTransformer()
        self.type_transformer = PlateTypeTransformer()

    def forward(self, x, tgt):
        """前向传播函数"""
        # 提取CNN特征
        feat = self.backbone.conv1(x)
        feat = self.backbone.bn1(feat)
        feat = self.backbone.relu(feat)
        feat = self.backbone.maxpool(feat)
        feat = self.backbone.layer1(feat)
        feat = self.backbone.layer2(feat)
        feat = self.backbone.layer3(feat)
        feat = self.backbone.layer4(feat)

        # 车牌类型分类
        type_logits = self.type_head(feat)

        # 处理用于序列识别的特征
        feat = self.conv_proj(feat)

        # 调整特征图形状为序列形式 [B, HW, C]
        B, C, H, W = feat.shape
        feat = feat.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]

        # 添加位置编码
        feat = self.pos_enc(feat)

        # 通过Transformer编码器处理
        encoded = self.encoder(feat)

        # 解码过程
        tgt_embed = self.char_embedding(tgt)
        tgt_embed = self.dec_pos_enc(tgt_embed)

        # 创建三角掩码，防止解码器看到未来位置
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(x.device)

        # 通过Transformer解码器处理
        output = self.decoder(tgt_embed, encoded, tgt_mask=tgt_mask)

        # 字符分类
        logits = self.char_out(output)

        return logits, type_logits

    @torch.no_grad()
    def predict(self, x, max_len=15):
        """预测函数，用于推理阶段"""
        self.eval()

        # 提取特征
        feat = self.backbone.conv1(x)
        feat = self.backbone.bn1(feat)
        feat = self.backbone.relu(feat)
        feat = self.backbone.maxpool(feat)
        feat = self.backbone.layer1(feat)
        feat = self.backbone.layer2(feat)
        feat = self.backbone.layer3(feat)
        feat = self.backbone.layer4(feat)

        # 车牌类型分类
        type_logits = self.type_head(feat)
        type_pred = type_logits.argmax(1)

        # 处理用于序列识别的特征
        feat = self.conv_proj(feat)
        B, C, H, W = feat.shape
        feat = feat.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        feat = self.pos_enc(feat)
        encoded = self.encoder(feat)

        # 初始化预测序列，以SOS开始
        batch_size = x.size(0)
        tgt = torch.full((batch_size, 1), SOS_IDX, dtype=torch.long, device=x.device)

        # 自回归生成字符序列
        for i in range(max_len):
            tgt_embed = self.char_embedding(tgt)
            tgt_embed = self.dec_pos_enc(tgt_embed)

            # 创建三角掩码
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(x.device)

            # 通过解码器
            output = self.decoder(tgt_embed, encoded, tgt_mask=tgt_mask)
            logits = self.char_out(output)

            # 获取下一个字符
            next_token = logits[:, -1, :].argmax(1).unsqueeze(1)

            # 将预测的字符添加到目标序列
            tgt = torch.cat([tgt, next_token], dim=1)

            # 如果所有序列都预测到了EOS，就提前结束
            if (next_token == EOS_IDX).all():
                break

        # 解码预测结果
        char_preds = []
        for seq in tgt:
            # 找到第一个EOS的位置
            eos_pos = (seq == EOS_IDX).nonzero(as_tuple=True)[0]
            if len(eos_pos) > 0:
                seq = seq[:eos_pos[0]]  # 截断到EOS
            char_preds.append(self.label_transformer.decode(seq.tolist()))

        # 解码类型预测结果
        type_preds = [self.type_transformer.decode(t.item()) for t in type_pred]

        return char_preds, type_preds


# === 训练和评估函数 ===
def train():
    # 数据集根目录（包含 CBLPRD-330k 文件夹和 train.txt）
    base_dir = r'D:\5EDemocache\platemodel\charmodel\CBLPRD-330k_v1'
    img_dir = base_dir  # 关键修改：使用父目录
    train_txt = os.path.join(base_dir, 'train.txt')
    val_txt = os.path.join(base_dir, 'val.txt')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")


    # 创建训练集和验证集的转换
    train_transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 创建数据集
    train_dataset = PlateDataset(img_dir, train_txt, transform=train_transform, is_train=True)
    val_dataset = PlateDataset(img_dir, val_txt, transform=val_transform, is_train=False)

    # 分析训练数据
    train_dataset.analyze_data()

    # 创建数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, collate_fn=custom_collate)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, collate_fn=custom_collate)

    # 创建模型
    model = PlateTransformer().to(device)

    # 优化器和学习率调度器
    optimizer = optim.AdamW(model.parameters(), lr=0.0003)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 损失函数
    ce_loss = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    ce_type = nn.CrossEntropyLoss()

    # 训练参数
    num_epochs = 50
    best_val_loss = float('inf')
    best_val_acc = 0.0

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_type_correct = 0
        train_char_correct = 0
        train_total = 0

        for imgs, labels, types, plates, ptypes in train_dataloader:
            imgs, labels, types = imgs.to(device), labels.to(device), types.to(device)

            # 准备解码器输入和输出
            tgt_input = labels[:, :-1]
            tgt_output = labels[:, 1:]

            # 前向传播
            logits, type_logits = model(imgs, tgt_input)

            # 计算损失
            loss_char = ce_loss(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
            loss_type = ce_type(type_logits, types)

            # 调整两个任务的权重
            if epoch < 5:  # 前5个epoch给类型分类更高权重
                loss = loss_char + 1.0 * loss_type
            else:
                loss = loss_char + 0.3 * loss_type

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计训练指标
            train_loss += loss.item() * imgs.size(0)
            train_type_correct += (type_logits.argmax(1) == types).sum().item()

            # 计算字符准确率（完全匹配）
            pred_chars = logits.argmax(2)
            for i in range(len(pred_chars)):
                pred_seq = pred_chars[i].tolist()
                true_seq = tgt_output[i].tolist()

                # 截断到EOS或PAD
                pred_seq = pred_seq[:pred_seq.index(EOS_IDX)] if EOS_IDX in pred_seq else pred_seq
                true_seq = true_seq[:true_seq.index(EOS_IDX)] if EOS_IDX in true_seq else true_seq

                if pred_seq == true_seq:
                    train_char_correct += 1

            train_total += imgs.size(0)

        # 计算平均训练损失和准确率
        train_loss /= len(train_dataset)
        train_type_acc = train_type_correct / train_total
        train_char_acc = train_char_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_type_correct = 0
        val_char_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, labels, types, plates, ptypes in val_dataloader:
                imgs, labels, types = imgs.to(device), labels.to(device), types.to(device)

                # 准备解码器输入和输出
                tgt_input = labels[:, :-1]
                tgt_output = labels[:, 1:]

                # 前向传播
                logits, type_logits = model(imgs, tgt_input)

                # 计算损失
                loss_char = ce_loss(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
                loss_type = ce_type(type_logits, types)
                loss = loss_char + 0.3 * loss_type

                # 统计验证指标
                val_loss += loss.item() * imgs.size(0)
                val_type_correct += (type_logits.argmax(1) == types).sum().item()

                # 计算字符准确率（完全匹配）
                pred_chars = logits.argmax(2)
                for i in range(len(pred_chars)):
                    pred_seq = pred_chars[i].tolist()
                    true_seq = tgt_output[i].tolist()

                    # 截断到EOS或PAD
                    pred_seq = pred_seq[:pred_seq.index(EOS_IDX)] if EOS_IDX in pred_seq else pred_seq
                    true_seq = true_seq[:true_seq.index(EOS_IDX)] if EOS_IDX in true_seq else true_seq

                    if pred_seq == true_seq:
                        val_char_correct += 1

                val_total += imgs.size(0)

            # 计算平均验证损失和准确率
            val_loss /= len(val_dataset)
            val_type_acc = val_type_correct / val_total
            val_char_acc = val_char_correct / val_total

            # 学习率调整
            scheduler.step(val_loss)

            # 打印训练和验证结果
            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"  训练: Loss={train_loss:.4f}, 类型准确率={train_type_acc:.4f}, 字符准确率={train_char_acc:.4f}")
            print(f"  验证: Loss={val_loss:.4f}, 类型准确率={val_type_acc:.4f}, 字符准确率={val_char_acc:.4f}")

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'crnn_plate_best_loss.pth')
                print(f"  已保存最佳损失模型: crnn_plate_best_loss.pth")

            if val_char_acc > best_val_acc:
                best_val_acc = val_char_acc
                torch.save(model.state_dict(), 'crnn_plate_best_acc.pth')
                print(f"  已保存最佳准确率模型: crnn_plate_best_acc.pth")

            # 每10个epoch保存一次模型
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), f'crnn_plate_epoch{epoch + 1}.pth')
                print(f"  已保存第{epoch + 1}轮模型")

            # 可视化预测结果
            if (epoch + 1) % 5 == 0:
                visualize_predictions(model, val_dataloader, device, epoch + 1)


def visualize_predictions(model, dataloader, device, epoch):
    """可视化模型预测结果"""
    model.eval()

    # 获取一批样本
    imgs, _, _, plates, ptypes = next(iter(dataloader))
    imgs = imgs.to(device)

    # 预测
    with torch.no_grad():
        char_preds, type_preds = model.predict(imgs)

    # 创建可视化
    plt.figure(figsize=(16, 12))
    for i in range(min(16, len(imgs))):
        plt.subplot(4, 4, i + 1)
        img_np = imgs[i].cpu().permute(1, 2, 0).numpy()
        img_np = (img_np * 0.5 + 0.5)  # 反归一化
        plt.imshow(np.clip(img_np, 0, 1))
        plt.title(f"真实: {plates[i]} ({ptypes[i]})\n"
                  f"预测: {char_preds[i]} ({type_preds[i]})", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'predictions_epoch_{epoch}.png')
    plt.close()


# === 测试函数 ===
@torch.no_grad()
def test_model(model_path, img_dir, label_file):
    """测试模型性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建转换
    transform = transforms.Compose([
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # 创建数据集和数据加载器
    dataset = PlateDataset(img_dir, label_file, transform=transform, is_train=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, collate_fn=custom_collate)

    # 创建模型并加载权重
    model = PlateTransformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 评估指标
    total = 0
    type_correct = 0
    char_correct = 0
    char_partial_correct = 0
    total_chars = 0

    # 错误样本收集
    errors = []

    for imgs, labels, types, plates, ptypes in dataloader:
        imgs, labels, types = imgs.to(device), labels.to(device), types.to(device)

        # 预测
        char_preds, type_preds = model.predict(imgs)

        # 计算准确率
        for i in range(len(char_preds)):
            total += 1

            # 类型准确率
            true_type = dataset.type_transformer.decode(types[i].item())
            if type_preds[i] == true_type:
                type_correct += 1

            # 字符准确率（完全匹配）
            if char_preds[i] == plates[i]:
                char_correct += 1

            # 字符准确率（部分匹配）
            correct_chars = 0
            min_len = min(len(char_preds[i]), len(plates[i]))
            for j in range(min_len):
                if char_preds[i][j] == plates[i][j]:
                    correct_chars += 1
            char_partial_correct += correct_chars / max(len(plates[i]), 1)
            total_chars += len(plates[i])

            # 收集错误样本
            if char_preds[i] != plates[i]:
                errors.append((imgs[i].cpu(), plates[i], ptypes[i], char_preds[i], type_preds[i]))

    # 打印评估结果
    print(f"测试结果:")
    print(f"  总样本数: {total}")
    print(f"  类型准确率: {type_correct / total:.4f}")
    print(f"  字符完全匹配准确率: {char_correct / total:.4f}")
    print(f"  字符部分匹配准确率: {char_partial_correct / total_chars:.4f}")

    # 可视化错误样本
    visualize_errors(errors, "error_samples.png")

    return {
        'total_samples': total,
        'type_accuracy': type_correct / total,
        'char_full_accuracy': char_correct / total,
        'char_partial_accuracy': char_partial_correct / total_chars
    }


def visualize_errors(errors, filename):
    """可视化错误样本"""
    if not errors:
        print("没有错误样本需要可视化")
        return

    plt.figure(figsize=(16, min(12, len(errors) * 3)))
    for i, (img, true_char, true_type, pred_char, pred_type) in enumerate(errors[:16]):
        plt.subplot(4, 4, i + 1)
        img_np = img.permute(1, 2, 0).numpy()
        img_np = (img_np * 0.5 + 0.5)  # 反归一化
        plt.imshow(np.clip(img_np, 0, 1))
        plt.title(f"真实: {true_char} ({true_type})\n"
                  f"预测: {pred_char} ({pred_type})", fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    train()
    # 测试模型
    # test_model('crnn_plate_best_acc.pth', 'CBLPRD-330k_v1', 'CBLPRD-330k_v1/test.txt')