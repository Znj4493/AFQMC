"""
FGM 对抗训练模块

Fast Gradient Method (FGM) 对抗训练实现
用于提升模型的鲁棒性，减少过拟合

使用方法:
    fgm = FGM(model, epsilon=1.0)

    for batch in dataloader:
        # 1. 正常训练
        loss = model(batch).loss
        loss.backward()

        # 2. 对抗训练
        fgm.attack()  # 添加扰动
        loss_adv = model(batch).loss
        loss_adv.backward()  # 累积梯度
        fgm.restore()  # 恢复embedding

        # 3. 更新参数
        optimizer.step()
        optimizer.zero_grad()
"""

import torch


class FGM:
    """
    FGM 对抗训练类

    核心思想：
        在 embedding 层添加对抗扰动，强迫模型学习更鲁棒的特征

    工作流程：
        1. 正常前向传播 + 反向传播（计算梯度）
        2. 调用 attack()：在 embedding 上添加扰动
        3. 对抗样本前向传播 + 反向传播（梯度累积）
        4. 调用 restore()：恢复原始 embedding
        5. 更新参数（使用累积的梯度）
    """

    def __init__(self, model, epsilon=1.0, emb_name='word_embeddings'):
        """
        初始化 FGM

        参数:
            model: PyTorch 模型
            epsilon: 扰动强度（默认 1.0）
                - 越大，对抗性越强，但可能影响正常样本学习
                - 推荐范围：0.5 ~ 2.0
            emb_name: embedding 层的名称关键字（默认 'word_embeddings'）
                - MacBERT 的 embedding 层包含 'word_embeddings' 关键字
        """
        self.model = model
        self.epsilon = epsilon
        self.emb_name = emb_name
        self.backup = {}  # 用于保存原始 embedding 参数


    def attack(self):
        """
        在 embedding 层添加对抗扰动

        数学原理:
            r = ε * g / ||g||₂

            其中:
            - r: 对抗扰动
            - ε: 扰动强度（epsilon）
            - g: 当前参数的梯度
            - ||g||₂: 梯度的 L2 范数（确保扰动方向归一化）

        为什么沿梯度方向添加扰动？
            - 梯度方向是损失函数增长最快的方向
            - 沿这个方向扰动，能制造"最难"的对抗样本
            - 强迫模型学习更鲁棒的特征
        """
        # 遍历模型的所有参数
        for name, param in self.model.named_parameters():
            # 只对 embedding 层操作，且参数需要梯度
            if param.requires_grad and self.emb_name in name:
                # 保存原始参数（用于后续恢复）
                self.backup[name] = param.data.clone()

                # 计算梯度的 L2 范数
                norm = torch.norm(param.grad)

                # 如果梯度不为0，添加扰动
                if norm != 0 and not torch.isnan(norm):
                    # 计算扰动: r = ε * g / ||g||₂
                    r_at = self.epsilon * param.grad / norm
                    # 添加扰动到参数上
                    param.data.add_(r_at)


    def restore(self):
        """
        恢复 embedding 层的原始参数

        为什么要恢复？
            - 扰动只用于对抗训练，不应该永久改变模型参数
            - 恢复后，参数更新（optimizer.step）基于原始参数位置
            - 确保模型参数的更新是正确的
        """
        # 遍历所有备份的参数
        for name, param in self.model.named_parameters():
            if name in self.backup:
                # 恢复原始参数值
                param.data = self.backup[name]

        # 清空备份字典
        self.backup = {}
