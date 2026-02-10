"""
损失函数模块

包含各种用于处理类别不平衡的损失函数：
1. Focal Loss - 专门处理类别不平衡和难易样本不平衡
2. 支持与类别权重结合使用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss 实现

    论文: "Focal Loss for Dense Object Detection" (https://arxiv.org/abs/1708.02002)

    核心思想:
        1. 降低易分样本的损失权重，让模型更关注难分样本
        2. 通过 gamma 参数控制聚焦程度
        3. 通过 alpha 参数平衡类别权重

    公式:
        FL(pt) = -α(1-pt)^γ * log(pt)

        其中:
        - pt: 模型预测的真实类别概率
        - α: 类别权重（平衡正负样本）
        - γ: 聚焦参数（降低易分样本权重）

    为什么 Focal Loss 对类别不平衡有效？
        1. (1-pt)^γ 项:
           - 易分样本（pt 接近 1）: (1-pt)^γ 接近 0，损失被大幅降低
           - 难分样本（pt 接近 0.5）: (1-pt)^γ 接近 0.5^γ，损失保持较高
           - 这样模型会更关注那些难以分类的样本

        2. α 项:
           - 可以为少数类设置更高的权重
           - 进一步平衡类别不平衡

    参数:
        alpha: float or list, 类别权重
            - None: 不使用类别权重
            - float: 对少数类的权重（0类权重为1-alpha，1类权重为alpha）
            - list: 每个类别的权重列表
        gamma: float, 聚焦参数
            - gamma=0: 退化为交叉熵损失
            - gamma>0: 降低易分样本权重
            - 推荐值: 2.0
        reduction: str, 损失聚合方式
            - 'mean': 返回平均损失
            - 'sum': 返回总损失
            - 'none': 返回每个样本的损失

    使用示例:
        >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = model(inputs)  # shape: (batch_size, num_classes)
        >>> loss = criterion(logits, labels)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        初始化 Focal Loss

        参数:
            alpha: 类别权重
                - None: 不使用权重
                - float: 对类别1的权重（0.25表示1类权重0.25，0类权重0.75）
                - list/tensor: 每个类别的权重
            gamma: 聚焦参数，推荐2.0
            reduction: 损失聚合方式 ('mean', 'sum', 'none')
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        # 处理 alpha 参数
        if alpha is None:
            # 不使用类别权重
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            # alpha 是单个值，转换为 [1-alpha, alpha]
            self.alpha = torch.tensor([1 - alpha, alpha])
        else:
            # alpha 是列表或张量
            self.alpha = torch.tensor(alpha) if not isinstance(alpha, torch.Tensor) else alpha

    def forward(self, logits, labels):
        """
        前向传播

        参数:
            logits: 模型输出的 logits，shape: (batch_size, num_classes)
            labels: 真实标签，shape: (batch_size,)

        返回:
            loss: 标量损失值
        """
        # 计算 softmax 概率
        # shape: (batch_size, num_classes)
        probs = F.softmax(logits, dim=-1)

        # 获取真实类别的概率 pt
        # shape: (batch_size,)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)

        # 计算基础的交叉熵损失（不使用 reduction）
        # shape: (batch_size,)
        ce_loss = F.cross_entropy(logits, labels, reduction='none')

        # 计算 Focal Loss 的调制因子: (1-pt)^gamma
        # shape: (batch_size,)
        focal_weight = (1 - pt) ** self.gamma

        # Focal Loss = focal_weight * ce_loss
        loss = focal_weight * ce_loss

        # 应用类别权重 alpha
        if self.alpha is not None:
            # 将 alpha 移到与 loss 相同的设备
            if self.alpha.device != loss.device:
                self.alpha = self.alpha.to(loss.device)

            # 获取每个样本对应的类别权重
            # shape: (batch_size,)
            alpha_t = self.alpha[labels]

            # 应用权重
            loss = alpha_t * loss

        # 根据 reduction 方式聚合损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def create_loss_function(loss_type='focal', alpha=None, gamma=2.0, class_weights=None):
    """
    创建损失函数

    参数:
        loss_type: str, 损失函数类型
            - 'focal': Focal Loss
            - 'ce': Cross Entropy Loss
        alpha: Focal Loss 的 alpha 参数（仅当 loss_type='focal' 时有效）
        gamma: Focal Loss 的 gamma 参数（仅当 loss_type='focal' 时有效）
        class_weights: 类别权重张量（仅当 loss_type='ce' 时有效）

    返回:
        criterion: 损失函数对象
    """
    if loss_type == 'focal':
        # 使用 Focal Loss
        criterion = FocalLoss(alpha=alpha, gamma=gamma)
        print(f"使用 Focal Loss (alpha={alpha}, gamma={gamma})")
    elif loss_type == 'ce':
        # 使用交叉熵损失
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            print(f"使用 Cross Entropy Loss with class weights")
        else:
            criterion = nn.CrossEntropyLoss()
            print(f"使用 Cross Entropy Loss")
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")

    return criterion
