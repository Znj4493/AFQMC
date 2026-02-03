"""
评估指标与辅助工具

本模块提供：
1. 评估指标计算（针对不平衡数据优化）
2. 类别权重计算（处理类别不平衡）
3. 训练辅助函数（随机种子、模型保存/加载）

核心功能：
- compute_metrics(): 计算 accuracy, precision, recall, f1_macro 等
- compute_class_weights(): 自动计算类别权重
- set_seed(): 保证实验可复现
- save/load_checkpoint(): 模型保存与恢复

使用方式：
    from utils import compute_metrics, set_seed

    # 设置随机种子
    set_seed(42)

    # 计算评估指标
    metrics = compute_metrics(y_true, y_pred)
    print(f"F1-macro: {metrics['f1_macro']:.4f}")
"""

import os
import random
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight


# ============================================
# 评估指标计算
# ============================================

def compute_metrics(y_true, y_pred, average='macro', verbose=False):
    """
    计算分类任务的评估指标

    这个函数针对二分类任务（特别是类别不平衡的情况）设计，
    计算多个评估指标来全面评估模型性能。

    参数:
        y_true: array-like, shape (n_samples,)
            真实标签

        y_pred: array-like, shape (n_samples,)
            预测标签

        average: str, default='macro'
            多分类指标的平均方式
            - 'macro': 每个类别权重相同（推荐用于不平衡数据）
            - 'weighted': 按样本数加权
            - 'binary': 二分类（针对label=1计算）

        verbose: bool, default=False
            是否打印详细的分类报告

    返回:
        dict: 包含各种评估指标的字典
            {
                'accuracy': float,           # 准确率
                'precision_macro': float,    # 宏平均精确率
                'recall_macro': float,       # 宏平均召回率
                'f1_macro': float,           # 宏平均F1（⭐ 主要指标）
                'precision_weighted': float, # 加权精确率
                'recall_weighted': float,    # 加权召回率
                'f1_weighted': float,        # 加权F1
                'confusion_matrix': ndarray, # 混淆矩阵
            }

    关键指标说明：
        - accuracy: 整体准确率，但不平衡数据下会被多数类主导
        - f1_macro: 每个类别的F1取平均，适合不平衡数据 ⭐
        - confusion_matrix: 可以看出模型在哪个类别上表现不好

    示例:
        >>> y_true = [0, 1, 0, 1, 0, 1]
        >>> y_pred = [0, 1, 0, 0, 0, 1]
        >>> metrics = compute_metrics(y_true, y_pred)
        >>> print(f"F1-macro: {metrics['f1_macro']:.4f}")
        F1-macro: 0.8333
    """

    # 将输入转换为numpy数组（兼容torch tensor和list）
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 1. 计算准确率（最简单的指标）
    accuracy = accuracy_score(y_true, y_pred)

    # 2. 计算宏平均指标（每个类别权重相同）
    # 对于不平衡数据，这是最重要的指标！
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # 3. 计算加权平均指标（按样本数加权）
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 4. 混淆矩阵（查看预测分布）
    cm = confusion_matrix(y_true, y_pred)

    # 5. 如果需要，打印详细报告
    if verbose:
        print("\n" + "=" * 80)
        print("模型评估报告 (Model Evaluation Report)")
        print("=" * 80)

        # 1. 整体性能指标
        print("\n【整体性能指标】")
        print(f"  准确率 (Accuracy):        {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  宏平均F1 (F1-macro):      {f1_macro:.4f} ({f1_macro*100:.2f}%)")
        print(f"  加权F1 (F1-weighted):     {f1_weighted:.4f} ({f1_weighted*100:.2f}%)")

        # 2. 各类别详细指标
        print("\n【各类别详细指标】")
        print("─" * 80)
        print(f"{'类别':<15} {'精确率':>10} {'召回率':>10} {'F1分数':>10} {'样本数':>10}")
        print("─" * 80)
        print(f"Class 0 (不相似)  {precision_score(y_true, y_pred, pos_label=0, zero_division=0):>9.4f}  "
              f"{recall_score(y_true, y_pred, pos_label=0, zero_division=0):>9.4f}  "
              f"{f1_score(y_true, y_pred, pos_label=0, zero_division=0):>9.4f}  "
              f"{(y_true == 0).sum():>9d}")
        print(f"Class 1 (相似)    {precision_score(y_true, y_pred, pos_label=1, zero_division=0):>9.4f}  "
              f"{recall_score(y_true, y_pred, pos_label=1, zero_division=0):>9.4f}  "
              f"{f1_score(y_true, y_pred, pos_label=1, zero_division=0):>9.4f}  "
              f"{(y_true == 1).sum():>9d}")
        print("─" * 80)

        # 3. 混淆矩阵（增强版）
        print("\n【混淆矩阵 (Confusion Matrix)】")
        print("─" * 80)
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total = tn + fp + fn + tp

        print(f"{'':>20} {'预测: 不相似':>15} {'预测: 相似':>15} {'总计':>10} {'召回率':>10}")
        print("─" * 80)
        print(f"{'实际: 不相似 (0)':>20} {tn:>15d} {fp:>15d} {tn+fp:>10d} {tn/(tn+fp)*100:>9.2f}%")
        print(f"{'实际: 相似 (1)':>20} {fn:>15d} {tp:>15d} {fn+tp:>10d} {tp/(fn+tp)*100:>9.2f}%")
        print("─" * 80)
        print(f"{'总计':>20} {tn+fn:>15d} {fp+tp:>15d} {total:>10d}")
        print(f"{'精确率':>20} {tn/(tn+fn)*100:>14.2f}% {tp/(fp+tp)*100:>14.2f}%")
        print("─" * 80)

        # 4. 错误分析
        print("\n【错误分析】")
        print(f"  [√] 正确预测: {tn + tp:>4d} 个样本 ({(tn+tp)/total*100:.2f}%)")
        print(f"  [X] 错误预测: {fp + fn:>4d} 个样本 ({(fp+fn)/total*100:.2f}%)")
        print(f"      - 假阳性 (FP): {fp:>4d} 个 (误判为相似)   {fp/total*100:.2f}%")
        print(f"      - 假阴性 (FN): {fn:>4d} 个 (误判为不相似) {fn/total*100:.2f}%")

        # 5. 性能评价
        print("\n【性能评价】")
        if f1_macro >= 0.76:
            print("  >> 模型性能: 优秀 (已达到阶段二目标)")
        elif f1_macro >= 0.70:
            print("  >> 模型性能: 良好 (接近目标，需优化)")
        elif f1_macro >= 0.65:
            print("  >> 模型性能: 中等 (需要改进)")
        else:
            print("  >> 模型性能: 较差 (需要大幅优化)")

        # 6. 建议
        print("\n【改进建议】")
        if f1_score(y_true, y_pred, pos_label=1, zero_division=0) < f1_score(y_true, y_pred, pos_label=0, zero_division=0) - 0.15:
            print("  * Class 1 (相似) 性能明显低于 Class 0，建议：")
            print("    - 增大类别权重")
            print("    - 增加训练轮数")
            print("    - 使用数据增强")
        if fp > fn * 1.2:
            print("  * 假阳性 (FP) 过多，模型倾向于过度预测为相似")
            print("    - 考虑提高预测阈值")
        if fn > fp * 1.2:
            print("  * 假阴性 (FN) 过多，模型倾向于过度预测为不相似")
            print("    - 考虑降低预测阈值或增大 Class 1 权重")

        print("=" * 80 + "\n")

    # 返回所有指标
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,           # ⭐ 最重要的指标
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
    }

    return metrics


def print_metrics(metrics, prefix=""):
    """
    格式化打印评估指标（用于训练过程中）

    参数:
        metrics: dict
            compute_metrics() 返回的指标字典
        prefix: str
            前缀字符串，例如 "Epoch 1 - Train" 或 "Validation"

    示例:
        >>> metrics = compute_metrics(y_true, y_pred)
        >>> print_metrics(metrics, prefix="Validation")
        Validation - Accuracy: 0.8234, F1-macro: 0.7891, F1-weighted: 0.8123
    """
    output = f"{prefix}" if prefix else ""
    output += f"Accuracy: {metrics['accuracy']:.4f}, "
    output += f"F1-macro: {metrics['f1_macro']:.4f}, "
    output += f"F1-weighted: {metrics['f1_weighted']:.4f}"
    print(output)


# ============================================
# 类别权重计算（处理不平衡）
# ============================================

def compute_class_weights(labels, method='balanced'):
    """
    计算类别权重（用于处理类别不平衡）

    为什么需要类别权重？
        在我们的数据集中，label 0 占 69.1%，label 1 占 30.9%
        如果不加权重，模型会倾向于预测多数类（label 0）
        通过给少数类更高的权重，让模型更重视少数类的学习

    参数:
        labels: array-like, shape (n_samples,)
            训练集的标签

        method: str, default='balanced'
            权重计算方式
            - 'balanced': sklearn默认方法，权重 = n_samples / (n_classes * n_samples_per_class)
            - 'manual': 手动指定权重（需要修改代码）

    返回:
        torch.Tensor: shape (n_classes,)
            每个类别的权重，可以直接传给 nn.CrossEntropyLoss(weight=...)

    工作原理:
        假设训练集有 28800 个样本：
        - label 0: 19900 个样本（69.1%）
        - label 1: 8900 个样本（30.9%）

        计算权重：
        - weight_0 = 28800 / (2 * 19900) ≈ 0.72  （降低多数类的权重）
        - weight_1 = 28800 / (2 * 8900) ≈ 1.62   （提高少数类的权重）

        这样，少数类的损失会被放大 1.62 倍，模型会更重视学习少数类

    示例:
        >>> labels = [0, 0, 0, 1, 0, 0, 1]  # 5个label 0, 2个label 1
        >>> weights = compute_class_weights(labels)
        >>> print(weights)
        tensor([0.7000, 1.7500])

        >>> # 在损失函数中使用
        >>> criterion = nn.CrossEntropyLoss(weight=weights)
    """

    # 转换为numpy数组
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    labels = np.array(labels)

    # 获取所有类别
    classes = np.unique(labels)

    if method == 'balanced':
        # 使用sklearn的compute_class_weight计算
        # 这是标准的平衡权重计算方法
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=labels
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # 转换为PyTorch tensor
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    return class_weights


# ============================================
# 随机种子设置（保证可复现性）
# ============================================

def set_seed(seed=42):
    """
    设置所有随机种子，保证实验可复现

    为什么需要设置随机种子？
        深度学习中有很多随机过程：
        1. 权重初始化（随机初始化神经网络参数）
        2. Dropout（随机丢弃神经元）
        3. 数据shuffle（打乱训练数据顺序）
        4. 数据增强（随机变换）

        不设置种子 → 每次运行结果不同 → 无法复现实验
        设置种子 → 每次运行结果一致 → 可以准确对比不同方法

    参数:
        seed: int, default=42
            随机种子值（通常使用42，这是一个经典的选择）

    设置范围:
        1. Python内置random模块
        2. NumPy
        3. PyTorch（CPU）
        4. PyTorch（GPU）- 如果使用CUDA

    注意:
        - 设置种子后，即使代码相同，不同硬件（不同GPU）可能也有微小差异
        - 这是因为GPU的浮点运算顺序可能不同
        - 但整体趋势和最终性能应该是一致的

    示例:
        >>> set_seed(42)
        >>> # 之后的所有随机操作都是可复现的
        >>> model = create_model(...)  # 权重初始化是固定的
        >>> train_loader = DataLoader(..., shuffle=True)  # shuffle顺序是固定的
    """
    # 1. Python内置random
    random.seed(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch
    torch.manual_seed(seed)

    # 4. 如果使用CUDA（GPU）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU

        # 以下两行可以进一步提高可复现性，但会降低性能
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    print(f"✓ 随机种子已设置为: {seed}")


# ============================================
# 模型保存与加载
# ============================================

def save_checkpoint(model, optimizer, epoch, metrics, save_path, scheduler=None):
    """
    保存模型检查点（Checkpoint）

    什么是Checkpoint？
        训练过程中定期保存模型状态，包括：
        - 模型权重（model.state_dict()）
        - 优化器状态（optimizer.state_dict()）- 包含学习率、动量等
        - 当前训练进度（epoch）
        - 当前最佳性能（metrics）

    为什么需要保存Checkpoint？
        1. 训练中断恢复：如果训练被中断（断电、OOM等），可以从上次保存的地方继续
        2. 模型选择：保存验证集上表现最好的模型
        3. 推理部署：训练完成后，加载最佳模型用于预测

    参数:
        model: nn.Module
            要保存的模型

        optimizer: torch.optim.Optimizer
            优化器（包含学习率等状态）

        epoch: int
            当前训练轮数

        metrics: dict
            当前的评估指标（例如：{'f1_macro': 0.85, 'accuracy': 0.88}）

        save_path: str
            保存路径，例如 'checkpoints/best_model.pt'

        scheduler: torch.optim.lr_scheduler, optional
            学习率调度器（如果有的话）

    保存内容:
        {
            'epoch': 当前轮数,
            'model_state_dict': 模型权重,
            'optimizer_state_dict': 优化器状态,
            'scheduler_state_dict': 学习率调度器状态（如果有）,
            'metrics': 评估指标,
        }

    示例:
        >>> # 在训练循环中
        >>> for epoch in range(num_epochs):
        >>>     train(...)
        >>>     metrics = evaluate(...)
        >>>
        >>>     # 如果当前模型最好，保存checkpoint
        >>>     if metrics['f1_macro'] > best_f1:
        >>>         best_f1 = metrics['f1_macro']
        >>>         save_checkpoint(model, optimizer, epoch, metrics,
        >>>                        'checkpoints/best_model.pt')
    """
    # 创建保存目录（如果不存在）
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 构建checkpoint字典
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }

    # 如果有学习率调度器，也保存它的状态
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    # 保存到文件
    torch.save(checkpoint, save_path)

    print(f"✓ Checkpoint已保存: {save_path}")
    print(f"  Epoch: {epoch}, F1-macro: {metrics.get('f1_macro', 0):.4f}")


def load_checkpoint(model, checkpoint_path, optimizer=None, scheduler=None, device='cpu'):
    """
    加载模型检查点

    参数:
        model: nn.Module
            要加载权重的模型（需要先创建好模型架构）

        checkpoint_path: str
            checkpoint文件路径

        optimizer: torch.optim.Optimizer, optional
            如果要恢复训练，传入优化器

        scheduler: torch.optim.lr_scheduler, optional
            如果要恢复训练，传入学习率调度器

        device: str or torch.device
            加载到哪个设备（'cpu' 或 'cuda'）

    返回:
        dict: checkpoint字典，包含epoch和metrics等信息

    使用场景:
        1. 推理部署：只加载模型权重
        >>> model = create_model(...)
        >>> load_checkpoint(model, 'checkpoints/best_model.pt', device='cuda')
        >>> model.eval()
        >>> # 开始推理...

        2. 恢复训练：加载模型、优化器、调度器
        >>> model = create_model(...)
        >>> optimizer = AdamW(model.parameters(), lr=2e-5)
        >>> checkpoint = load_checkpoint(model, 'checkpoints/last.pt',
        >>>                             optimizer=optimizer, device='cuda')
        >>> start_epoch = checkpoint['epoch'] + 1
        >>> # 从start_epoch继续训练...
    """
    # 检查文件是否存在
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint文件不存在: {checkpoint_path}")

    # 加载checkpoint
    # weights_only=False: 允许加载包含numpy对象的checkpoint（PyTorch 2.6+需要）
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])

    # 如果提供了优化器，加载优化器状态
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 如果提供了学习率调度器，加载调度器状态
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print(f"✓ Checkpoint已加载: {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    if 'metrics' in checkpoint:
        print(f"  F1-macro: {checkpoint['metrics'].get('f1_macro', 0):.4f}")

    return checkpoint


# ============================================
# 测试代码
# ============================================

if __name__ == '__main__':
    """
    测试本模块的各个功能
    """

    print("=" * 60)
    print("测试 utils.py 功能")
    print("=" * 60)

    # 1. 测试设置随机种子
    print("\n1. 测试设置随机种子...")
    set_seed(42)

    # 2. 测试评估指标计算
    print("\n2. 测试评估指标计算...")

    # 模拟预测结果（模拟一个类别不平衡的场景）
    # 假设有100个样本：70个label 0，30个label 1
    y_true = np.array([0] * 70 + [1] * 30)
    # 模型预测（故意让模型在label 1上表现较差）
    y_pred = np.array([0] * 65 + [1] * 5 + [1] * 20 + [0] * 10)

    print(f"样本数: {len(y_true)}")
    print(f"真实分布: label 0 = {sum(y_true == 0)}, label 1 = {sum(y_true == 1)}")
    print(f"预测分布: label 0 = {sum(y_pred == 0)}, label 1 = {sum(y_pred == 1)}")

    # 计算指标（带详细报告）
    metrics = compute_metrics(y_true, y_pred, verbose=True)

    # 打印关键指标
    print_metrics(metrics, prefix="测试结果 - ")

    # 3. 测试类别权重计算
    print("\n3. 测试类别权重计算...")

    # 模拟训练集标签（类别不平衡：2.23:1）
    train_labels = np.array([0] * 22106 + [1] * 9894)
    print(f"训练集样本数: {len(train_labels)}")
    print(f"  label 0: {sum(train_labels == 0)} ({sum(train_labels == 0) / len(train_labels) * 100:.1f}%)")
    print(f"  label 1: {sum(train_labels == 1)} ({sum(train_labels == 1) / len(train_labels) * 100:.1f}%)")

    class_weights = compute_class_weights(train_labels)
    print(f"\n计算得到的类别权重:")
    print(f"  label 0 权重: {class_weights[0]:.4f}  (多数类，权重较小)")
    print(f"  label 1 权重: {class_weights[1]:.4f}  (少数类，权重较大)")

    # 验证权重的作用
    print(f"\n权重比例: {class_weights[1] / class_weights[0]:.2f}:1")
    print(f"说明: label 1 的损失会被放大 {class_weights[1] / class_weights[0]:.2f} 倍")

    # 4. 测试checkpoint保存与加载
    print("\n4. 测试checkpoint保存与加载...")

    # 创建一个简单的模型用于测试
    from model_macbert import create_model
    try:
        # 尝试加载模型（如果有的话）
        print("尝试创建测试模型...")
        test_model = torch.nn.Linear(10, 2)  # 简单的线性模型用于测试
        test_optimizer = torch.optim.Adam(test_model.parameters(), lr=0.001)

        # 保存checkpoint
        test_metrics = {'f1_macro': 0.85, 'accuracy': 0.88}
        test_save_path = '../checkpoints/test_checkpoint.pt'

        save_checkpoint(
            model=test_model,
            optimizer=test_optimizer,
            epoch=5,
            metrics=test_metrics,
            save_path=test_save_path
        )

        # 加载checkpoint
        test_model_new = torch.nn.Linear(10, 2)
        test_optimizer_new = torch.optim.Adam(test_model_new.parameters(), lr=0.001)

        loaded_checkpoint = load_checkpoint(
            model=test_model_new,
            checkpoint_path=test_save_path,
            optimizer=test_optimizer_new,
            device='cpu'
        )

        print(f"✓ Checkpoint保存和加载测试成功!")

        # 清理测试文件
        if os.path.exists(test_save_path):
            os.remove(test_save_path)
            print(f"✓ 测试文件已清理")

    except Exception as e:
        print(f"✗ Checkpoint测试失败: {e}")

    print("\n" + "=" * 60)
    print("✓ 所有测试完成!")
    print("=" * 60)
