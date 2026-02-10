"""
mermaid
flowchart TD
    A[Create LoRAConfig] --> B[Call get_peft_config()]
    B --> C[Build peft.LoraConfig]
    C --> D[Inject LoRA into Model]
    D --> E[Train / Infer]
"""

from dataclasses import dataclass, field
from typing import List

import peft


@dataclass
class LoRAConfig:
    """LoRA 超参数配置类。"""

    rank: int = 16
    # LoRA 秩：从 8 提升到 16，增强模型对语义细微差异的拟合能力
    alpha: int = 32
    # 缩放因子：保持 rank 的 2 倍比例（16×2=32），控制 LoRA 层的学习强度
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention 层
            "gate_proj", "up_proj", "down_proj",       # MLP 层
        ]
    )
    # 目标模块：覆盖 Attention（Q/K/V/O）+ MLP（gate/up/down），
    # 让模型的"阅读理解"和"推理判断"能力都能被微调
    dropout: float = 0.1
    # 防止过拟合：增大 dropout 从 0.05 到 0.1，缓解之前 loss 下降但 F1 停滞的过拟合现象
    bias: str = "none"
    # 不训练 bias：减少可训练参数，降低显存占用与过拟合风险
    task_type: str = "CAUSAL_LM"
    # 任务类型：因果语言模型任务，匹配 Qwen-Chat 的生成式建模方式

    def get_peft_config(self) -> peft.LoraConfig:
        """Create a PEFT LoRA configuration.

        Returns:
            peft.LoraConfig: The LoRA configuration object to be injected into a model.
        """
        lora_config = peft.LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            bias=self.bias,
            task_type=peft.TaskType[self.task_type],
        )

        return lora_config
