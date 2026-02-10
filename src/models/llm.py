"""
mermaid
flowchart TD
    A[Get Model Path] --> B[Configure 4-bit Quantization]
    B --> C[Load Base Model]
    C --> D[Load Tokenizer]
    D --> E[Enable Gradient Checkpointing]
    E --> F[Inject LoRA Adapter]
    F --> G[Count Trainable Params]
    G --> H[Return Model and Tokenizer]
"""

from typing import Tuple
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import peft

from lora_config import LoRAConfig


def load_model_with_lora() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Qwen-1.5-1.8B model and inject a LoRA adapter.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The LoRA-ready model and its tokenizer.

    Example:
        model, tokenizer = load_model_with_lora()
    """
    # 步骤1：获取模型路径（使用相对路径，便于在不同机器上迁移项目）
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "checkpoints",
        "Qwen1.5-1.8B-Chat",
    )

    # 步骤2：配置 4-bit 量化（降低显存占用，适配 RTX 4060 8GB）
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 启用 4-bit 量化，显著减少显存占用
        bnb_4bit_quant_type="nf4",  # NF4 量化类型，精度/速度平衡较好
        bnb_4bit_compute_dtype=torch.float16,  # 计算使用 FP16，进一步节省显存
        bnb_4bit_use_double_quant=True,  # 双重量化，进一步压缩模型权重
    )

    # 步骤3：加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",  # 自动分配设备，避免手动指定 GPU/CPU
        trust_remote_code=True,
    )

    # 步骤4：加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # 步骤5 + 6：先注入 LoRA，再启用 gradient checkpointing
    # 顺序很重要！必须先注入 LoRA 让参数变为可训练，再启用 checkpointing
    # 否则 checkpointing 层看不到可训练参数，反向传播会报错
    lora_cfg = LoRAConfig()
    peft_config = lora_cfg.get_peft_config()
    model = peft.get_peft_model(model, peft_config)

    # 启用 gradient checkpointing（用计算换显存）
    # use_reentrant=False 是 4-bit 量化模型的必需设置，确保梯度正确传播
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    # 让模型输入的 embedding 层也能传递梯度
    model.enable_input_require_grads()

    # 步骤7：计算并打印可训练参数
    # 思路：遍历所有参数，分别累计总参数量与 requires_grad=True 的可训练参数量
    total_params = 0
    trainable_params = 0
    for param in model.parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count

    trainable_ratio = (trainable_params / total_params) * 100 if total_params else 0.0
    print(
        f"可训练参数: {trainable_params:,} / {total_params:,} "
        f"({trainable_ratio:.2f}%)"
    )

    # 步骤8：返回结果
    return model, tokenizer


if __name__ == "__main__":
    # 直接运行此文件时，测试模型加载
    print("=" * 60)
    print("测试：加载 Qwen-1.8B + LoRA")
    print("=" * 60)
    model, tokenizer = load_model_with_lora()
    print("✅ 模型加载成功！")
