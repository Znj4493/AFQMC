"""
MacBERT 训练脚本

本脚本实现了完整的训练流程，包含以下核心功能：
1. 数据加载与预处理（Dataset + DataLoader）
2. 模型训练（含 AMP 自动混合精度）
3. 梯度累积（节省显存）
4. 学习率 Warmup + Linear Decay
5. 类别权重处理不平衡
6. 验证集评估（F1-macro 为主要指标）
7. 最佳模型保存（Checkpoint）
8. 早停机制（Early Stopping）

使用方式:
    python train.py

训练配置:
    所有超参数在 config/config.py 中设置
    - batch_size = 32
    - learning_rate = 2e-5
    - num_epochs = 3
    - use_amp = True (自动混合精度)
    - gradient_accumulation_steps = 2

预期性能（阶段二目标）:
    - Accuracy: 80-83%
    - F1-macro: 76-79%
"""

import sys
import os
# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import warnings
import logging

# 抑制警告信息
warnings.filterwarnings('ignore')
logging.getLogger('transformers').setLevel(logging.ERROR)

# 导入项目模块
from config import config
from model_macbert import create_model, create_tokenizer
from data_loader import load_train_data, split_train_val
from utils import (
    set_seed,
    compute_metrics,
    print_metrics,
    compute_class_weights,
    save_checkpoint,
    load_checkpoint
)
from adversarial import FGM


# ============================================
# 自定义 Dataset 类
# ============================================

class AFQMCDataset(Dataset):
    """
    AFQMC 文本对数据集

    作用:
        将原始文本数据转换为模型可以处理的格式

    工作流程:
        1. 接收文本对列表和标签
        2. 使用 tokenizer 将文本转换为 token IDs
        3. 返回 input_ids, attention_mask, labels

    为什么需要自定义 Dataset？
        PyTorch 的 DataLoader 需要一个 Dataset 对象
        Dataset 定义了如何获取单个样本（__getitem__）
        DataLoader 会自动批量化、打乱、多线程加载
    """

    def __init__(self, texts1, texts2, labels, tokenizer, max_length):
        """
        参数:
            texts1: list of str
                第一个文本列表
            texts2: list of str
                第二个文本列表
            labels: list of int
                标签列表（0 或 1）
            tokenizer: BertTokenizer
                分词器
            max_length: int
                最大序列长度
        """
        self.texts1 = texts1
        self.texts2 = texts2
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """返回数据集大小"""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        获取单个样本

        这个函数会被 DataLoader 反复调用，每次获取一个样本
        DataLoader 会自动将多个样本组合成一个 batch

        返回:
            dict: 包含 input_ids, attention_mask, labels
        """
        # 获取第 idx 个样本的文本和标签
        text1 = str(self.texts1[idx])
        text2 = str(self.texts2[idx])
        label = self.labels[idx]

        # 使用 tokenizer 将文本转换为模型输入
        # 关键点：
        # 1. text, text_pair 会自动拼接为 [CLS] text1 [SEP] text2 [SEP]
        # 2. padding='max_length' 会填充到固定长度（batch 内长度统一）
        # 3. truncation=True 会截断超长文本
        # 4. return_tensors='pt' 返回 PyTorch tensor
        encoding = self.tokenizer(
            text1,
            text2,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            verbose=False  # 抑制警告信息
        )

        # 返回一个字典，包含模型需要的所有输入
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # squeeze 去掉 batch 维度 [1, L] -> [L]
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# ============================================
# DataLoader 创建函数
# ============================================

def create_dataloaders(train_texts1, train_texts2, train_labels,
                       val_texts1, val_texts2, val_labels,
                       tokenizer, config):
    """
    创建训练集和验证集的 DataLoader

    DataLoader 的作用:
        1. 自动批量化（batch）
        2. 自动打乱（shuffle）
        3. 多线程加载（num_workers）
        4. 内存管理

    参数:
        train_texts1, train_texts2, train_labels: 训练集数据
        val_texts1, val_texts2, val_labels: 验证集数据
        tokenizer: 分词器
        config: 配置对象

    返回:
        train_loader, val_loader: 训练集和验证集的 DataLoader
    """

    # 创建训练集 Dataset
    train_dataset = AFQMCDataset(
        texts1=train_texts1,
        texts2=train_texts2,
        labels=train_labels,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )

    # 创建验证集 Dataset
    val_dataset = AFQMCDataset(
        texts1=val_texts1,
        texts2=val_texts2,
        labels=val_labels,
        tokenizer=tokenizer,
        max_length=config.MAX_LENGTH
    )

    # 创建训练集 DataLoader
    # 关键点：
    # - shuffle=True: 每个 epoch 都打乱顺序，防止记住顺序
    # - num_workers=0: Windows 上建议设为 0，避免多进程问题
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,  # 训练集必须打乱！
        num_workers=config.NUM_WORKERS
    )

    # 创建验证集 DataLoader
    # 关键点：
    # - shuffle=False: 验证集不需要打乱，保持顺序一致
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,  # 验证集不打乱
        num_workers=config.NUM_WORKERS
    )

    return train_loader, val_loader


# ============================================
# 训练一个 Epoch
# ============================================

def train_epoch(model, train_loader, optimizer, scheduler, criterion,
                device, scaler=None, accumulation_steps=1, fgm=None):
    """
    训练一个 epoch

    这是训练循环的核心函数，包含：
    1. 遍历训练集的所有 batch
    2. 前向传播（使用 AMP）
    3. 反向传播（使用梯度累积）
    4. 参数更新
    5. 学习率调度

    参数:
        model: 模型
        train_loader: 训练集 DataLoader
        optimizer: 优化器
        scheduler: 学习率调度器
        criterion: 损失函数
        device: 设备（cuda 或 cpu）
        scaler: AMP 梯度缩放器（如果使用混合精度）
        accumulation_steps: 梯度累积步数

    返回:
        avg_loss: 平均损失
    """

    # 设置为训练模式
    # 作用：启用 Dropout 和 BatchNorm 的训练行为
    model.train()

    total_loss = 0
    num_batches = len(train_loader)

    # 使用 tqdm 显示进度条
    # position=0: 固定进度条位置
    # leave=True: epoch 结束后保留进度条
    # ncols=100: 固定进度条宽度，避免换行
    pbar = tqdm(train_loader, desc="Training", position=0, leave=True, ncols=100)

    for step, batch in enumerate(pbar):
        # 将数据移到 GPU
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # ========== 前向传播 ==========
        # 关键点：使用 autocast 启用自动混合精度
        if scaler is not None:
            # 使用 AMP（自动混合精度）
            with torch.cuda.amp.autocast():
                # 前向传播
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                # 梯度累积：将损失除以累积步数
                # 原因：我们要累积多个 batch 的梯度，需要平均
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps
        else:
            # 不使用 AMP（普通 FP32 训练）
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            if accumulation_steps > 1:
                loss = loss / accumulation_steps

        # ========== 反向传播 ==========
        if scaler is not None:
            # 使用 AMP：放大损失后反向传播
            scaler.scale(loss).backward()
        else:
            # 普通反向传播
            loss.backward()

        # 累积总损失（用于计算平均损失）
        total_loss += loss.item() * accumulation_steps

        # ========== 对抗训练（FGM） ==========
        # 如果启用对抗训练，执行以下4个步骤
        if fgm is not None:
            # 步骤1: 在 embedding 层添加对抗扰动
            # 扰动方向: 沿梯度方向（损失增长最快的方向）
            # 扰动公式: r = ε * g / ||g||₂
            fgm.attack()

            # 步骤2: 对抗样本前向传播
            # 使用被扰动的 embedding 重新计算损失
            # 这会制造"最难"的样本，强迫模型学习更鲁棒的特征
            if scaler is not None:
                # 使用 AMP（自动混合精度）
                with torch.cuda.amp.autocast():
                    outputs_adv = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss_adv = outputs_adv.loss
                    # 梯度累积：损失除以累积步数
                    if accumulation_steps > 1:
                        loss_adv = loss_adv / accumulation_steps
            else:
                # 不使用 AMP（普通 FP32 训练）
                outputs_adv = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss_adv = outputs_adv.loss
                if accumulation_steps > 1:
                    loss_adv = loss_adv / accumulation_steps

            # 步骤3: 对抗样本反向传播（梯度累积）
            # 关键：这次反向传播的梯度会累加到步骤B的梯度上
            # 最终参数更新使用的是：正常样本梯度 + 对抗样本梯度
            if scaler is not None:
                scaler.scale(loss_adv).backward()
            else:
                loss_adv.backward()

            # 步骤4: 恢复 embedding 原始值
            # 为什么要恢复？
            # - 扰动只用于对抗训练，不应该永久改变模型参数
            # - 参数更新（optimizer.step）基于原始参数位置
            fgm.restore()
                        

        # ========== 参数更新 ==========
        # 关键点：每 accumulation_steps 步才更新一次参数
        if (step + 1) % accumulation_steps == 0 or (step + 1) == num_batches:
            if scaler is not None:
                # 使用 AMP：先 unscale 梯度，然后裁剪，最后更新
                scaler.unscale_(optimizer)
                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
            else:
                # 普通更新
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
                optimizer.step()

            # 更新学习率
            scheduler.step()

            # 清空梯度，准备下一次累积
            optimizer.zero_grad()

        # 更新进度条
        pbar.set_postfix({'loss': loss.item() * accumulation_steps,
                         'lr': scheduler.get_last_lr()[0]})

    # 计算平均损失
    avg_loss = total_loss / num_batches

    return avg_loss


# ============================================
# 验证函数
# ============================================

def evaluate(model, val_loader, criterion, device):
    """
    在验证集上评估模型

    与训练不同：
    1. 不计算梯度（节省显存和时间）
    2. 不更新参数
    3. 使用 model.eval() 模式（关闭 Dropout）

    参数:
        model: 模型
        val_loader: 验证集 DataLoader
        criterion: 损失函数
        device: 设备

    返回:
        avg_loss: 平均损失
        metrics: 评估指标字典（accuracy, f1_macro, etc.）
        all_preds: 所有预测结果
        all_labels: 所有真实标签
    """

    # 设置为评估模式
    # 作用：关闭 Dropout 和 BatchNorm 的训练行为
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    # 使用 tqdm 显示进度条
    pbar = tqdm(val_loader, desc="Evaluating", position=0, leave=True, ncols=100)

    # 不计算梯度（节省显存和加速）
    with torch.no_grad():
        for batch in pbar:
            # 将数据移到 GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            # 累积损失
            total_loss += loss.item()

            # 获取预测结果
            preds = torch.argmax(logits, dim=-1)

            # 收集所有预测和标签（用于计算指标）
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算平均损失
    avg_loss = total_loss / len(val_loader)

    # 计算评估指标
    metrics = compute_metrics(all_labels, all_preds)

    return avg_loss, metrics, all_preds, all_labels


# ============================================
# 主训练函数
# ============================================

def main():
    """
    主训练流程

    完整流程：
    1. 设置随机种子
    2. 加载数据
    3. 创建模型、优化器、损失函数
    4. 创建学习率调度器
    5. 创建 AMP scaler（如果使用）
    6. 训练循环
        - 训练一个 epoch
        - 验证
        - 保存最佳模型
        - 早停检查
    7. 加载最佳模型，最终评估
    """

    print("=" * 80)
    print("MacBERT 训练脚本")
    print("=" * 80)

    # ========== 1. 设置随机种子（保证可复现） ==========
    print("\n[1/7] 设置随机种子...")
    set_seed(config.RANDOM_SEED)

    # 设置设备
    device = config.DEVICE
    print(f"使用设备: {device}")

    # ========== 2. 加载数据 ==========
    print("\n[2/7] 加载数据...")

    # 加载训练数据
    train_data = load_train_data(config.TRAIN_FILE)
    print(f"训练数据总数: {len(train_data)}")

    # 切分训练集和验证集（使用分层采样）
    train_df, val_df = split_train_val(
        train_data,
        val_ratio=config.VAL_RATIO,
        random_state=config.RANDOM_SEED
    )

    print(f"训练集: {len(train_df)} 样本")
    print(f"验证集: {len(val_df)} 样本")

    # 提取数据
    train_texts1 = train_df['text1'].tolist()
    train_texts2 = train_df['text2'].tolist()
    train_labels = train_df['label'].tolist()

    val_texts1 = val_df['text1'].tolist()
    val_texts2 = val_df['text2'].tolist()
    val_labels = val_df['label'].tolist()

    # 打印数据分布
    train_label_dist = np.bincount(train_labels)
    val_label_dist = np.bincount(val_labels)
    print(f"\n训练集标签分布: label 0 = {train_label_dist[0]}, label 1 = {train_label_dist[1]}")
    print(f"验证集标签分布: label 0 = {val_label_dist[0]}, label 1 = {val_label_dist[1]}")

    # ========== 3. 创建模型和分词器 ==========
    print("\n[3/7] 创建模型和分词器...")

    # 创建分词器
    tokenizer = create_tokenizer(config.MODEL_NAME)
    print(f"✓ 分词器加载完成")

    # 创建模型
    model = create_model(config.MODEL_NAME, num_labels=2)
    model = model.to(device)
    print(f"✓ 模型加载完成，参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 4. 创建 DataLoader ==========
    print("\n[4/7] 创建 DataLoader...")

    train_loader, val_loader = create_dataloaders(
        train_texts1, train_texts2, train_labels,
        val_texts1, val_texts2, val_labels,
        tokenizer, config
    )

    print(f"训练集 batch 数: {len(train_loader)}")
    print(f"验证集 batch 数: {len(val_loader)}")

    # ========== 5. 创建优化器、损失函数、学习率调度器 ==========
    print("\n[5/7] 创建优化器和调度器...")

    # 计算类别权重（处理不平衡）
    if config.USE_CLASS_WEIGHT:
        class_weights = compute_class_weights(train_labels)
        class_weights = class_weights.to(device)
        print(f"类别权重: label 0 = {class_weights[0]:.4f}, label 1 = {class_weights[1]:.4f}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # 创建优化器（AdamW）
    # 为什么用 AdamW？
    # - Adam 的改进版本，解耦了权重衰减
    # - BERT 类模型的标准优化器
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # 计算总训练步数
    total_steps = len(train_loader) * config.NUM_EPOCHS // config.GRADIENT_ACCUMULATION_STEPS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)

    print(f"总训练步数: {total_steps}")
    print(f"Warmup 步数: {warmup_steps} ({config.WARMUP_RATIO * 100:.0f}%)")

    # 创建学习率调度器（Warmup + Linear Decay）
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 创建 AMP scaler（如果使用混合精度）
    scaler = None
    if config.USE_AMP and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        print("✓ 启用自动混合精度训练（AMP）")
    
    # 创建 FGM 对抗训练对象（如果配置中启用）
    fgm = None
    if hasattr(config, 'USE_ADVERSARIAL') and config.USE_ADVERSARIAL:
        fgm = FGM(model, epsilon=getattr(config, 'ADV_EPSILON', 1.0))
        print(f"✓ 启用 FGM 对抗训练（epsilon={getattr(config, 'ADV_EPSILON', 1.0)}）")


    # ========== 6. 打印训练配置 ==========
    print("\n" + "=" * 80)
    print("训练配置")
    print("=" * 80)
    print(f"模型: {config.MODEL_NAME}")
    print(f"最大序列长度: {config.MAX_LENGTH}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"梯度累积步数: {config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"有效 Batch Size: {config.BATCH_SIZE * config.GRADIENT_ACCUMULATION_STEPS}")
    print(f"学习率: {config.LEARNING_RATE}")
    print(f"训练轮数: {config.NUM_EPOCHS}")
    print(f"使用 AMP: {config.USE_AMP and device.type == 'cuda'}")
    print(f"使用类别权重: {config.USE_CLASS_WEIGHT}")
    print(f"评估指标: {config.METRIC_FOR_BEST_MODEL}")
    print(f"早停耐心值: {config.EARLY_STOPPING_PATIENCE}")
    print("=" * 80)

    # ========== 7. 训练循环 ==========
    print("\n[6/7] 开始训练...")

    best_metric = 0.0
    patience_counter = 0

    # 用于记录每个 epoch 的结果
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1_macro': [],
    }

    for epoch in range(config.NUM_EPOCHS):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{config.NUM_EPOCHS}")
        print(f"{'='*80}")

        # 记录开始时间
        epoch_start_time = time.time()

        # ========== 训练阶段 ==========
        train_loss = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            scaler=scaler,
            accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
            fgm=fgm
        )

        # ========== 验证阶段 ==========
        val_loss, val_metrics, _, _ = evaluate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device
        )

        # 计算耗时
        epoch_time = time.time() - epoch_start_time

        # ========== 打印结果 ==========
        print(f"\nEpoch {epoch + 1} 结果:")
        print(f"  训练损失: {train_loss:.4f}")
        print(f"  验证损失: {val_loss:.4f}")
        print_metrics(val_metrics, prefix="  验证集 - ")
        print(f"  耗时: {epoch_time:.2f}秒")

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1_macro'].append(val_metrics['f1_macro'])

        # ========== 保存最佳模型 ==========
        current_metric = val_metrics[config.METRIC_FOR_BEST_MODEL]

        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0

            # 保存最佳模型
            best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                save_path=best_model_path,
                scheduler=scheduler
            )

            print(f"  ✓ 新的最佳模型！{config.METRIC_FOR_BEST_MODEL} = {best_metric:.4f}")
        else:
            patience_counter += 1
            print(f"  最佳 {config.METRIC_FOR_BEST_MODEL}: {best_metric:.4f} (当前: {current_metric:.4f})")
            print(f"  早停计数: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")

        # ========== 早停检查 ==========
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n早停触发！连续 {config.EARLY_STOPPING_PATIENCE} 个 epoch 没有提升。")
            break

        # 保存最后一个模型（用于恢复训练）
        last_model_path = os.path.join(config.CHECKPOINT_DIR, 'last_model.pt')
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=val_metrics,
            save_path=last_model_path,
            scheduler=scheduler
        )

    # ========== 8. 最终评估（使用最佳模型） ==========
    print("\n" + "=" * 80)
    print("[7/7] 最终评估（加载最佳模型）")
    print("=" * 80)

    best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pt')
    load_checkpoint(model, best_model_path, device=device)

    # 在验证集上最终评估
    _, final_metrics, all_preds, all_labels = evaluate(model, val_loader, criterion, device)

    print("\n最终验证集结果:")
    compute_metrics(all_labels, all_preds, verbose=True)  # 打印详细报告

    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"最佳模型保存在: {best_model_path}")
    print(f"最佳 {config.METRIC_FOR_BEST_MODEL}: {best_metric:.4f}")
    print("\n训练历史:")
    for i, (tl, vl, va, vf) in enumerate(zip(
        history['train_loss'],
        history['val_loss'],
        history['val_accuracy'],
        history['val_f1_macro']
    )):
        print(f"  Epoch {i+1}: train_loss={tl:.4f}, val_loss={vl:.4f}, "
              f"val_acc={va:.4f}, val_f1={vf:.4f}")

    print("=" * 80)


# ============================================
# 程序入口
# ============================================

if __name__ == '__main__':
    """
    运行训练脚本

    使用方式:
        python train.py

    注意事项:
        1. 确保 config/config.py 中的配置正确
        2. 确保 dataset/train.jsonl 存在
        3. 确保模型路径正确（checkpoints/chinese-macbert-base）
        4. 首次运行会比较慢（需要加载模型）
        5. 训练过程中可以用 Ctrl+C 中断，最后的模型会保存在 last_model.pt

    预期运行时间（RTX 4060）:
        - 单个 epoch: 约 8-10 分钟
        - 3 个 epochs: 约 25-30 分钟
    """
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断！")
        print("最后的模型已保存在: checkpoints/macbert/last_model.pt")
    except Exception as e:
        print(f"\n\n训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
