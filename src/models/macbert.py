"""
MacBERT文本对分类模型

模型架构流程:
    输入文本对
    → Tokenizer (分词)
    → MacBERT Encoder (12层Transformer，预训练好的)
    → [CLS] token的输出 (包含整个句子对的语义信息)
    → Dropout (防止过拟合)
    → Linear分类层 (768维 → 2维)
    → 输出 Logits (两个类别的分数)

使用方式:
    from model_macbert import create_model
    model = create_model(model_name='path/to/macbert', num_labels=2)
    outputs = model(input_ids, attention_mask, labels)
    loss = outputs.loss
    logits = outputs.logits
"""

import sys
import os
# 添加项目根目录到路径，以便导入config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    BertConfig
)

# 导入配置文件
try:
    from config import config
except ImportError:
    # 如果导入失败，使用默认值
    print("警告: 无法导入config，使用默认值")
    config = None


def create_model(model_name, num_labels=2, dropout_prob=0.1):
    """
    创建MacBERT分类模型

    这个函数封装了HuggingFace的BertForSequenceClassification，
    它已经包含了完整的BERT + 分类头，非常方便使用。

    参数:
        model_name: str
            预训练模型路径，例如:
            - 'hfl/chinese-macbert-base' (从HuggingFace下载)
            - './checkpoints/chinese-macbert-base' (本地路径)

        num_labels: int, default=2
            分类类别数
            - 二分类任务设置为2
            - 多分类任务设置为类别数

        dropout_prob: float, default=0.1
            Dropout概率，用于防止过拟合
            - 训练时随机丢弃这个比例的神经元
            - 推理时自动关闭
            - 常用范围: 0.1 ~ 0.3

    返回:
        model: BertForSequenceClassification
            可以直接用于训练和推理的模型

    模型的forward方法会返回一个对象，包含:
        - loss: 如果提供了labels，会自动计算损失
        - logits: 模型的原始输出 [batch_size, num_labels]
        - hidden_states: 所有层的隐藏状态（需要配置才返回）
        - attentions: 注意力权重（需要配置才返回）

    使用示例:
        >>> model = create_model('hfl/chinese-macbert-base')
        >>> # 训练时
        >>> outputs = model(input_ids, attention_mask, labels=labels)
        >>> loss = outputs.loss  # 自动计算的损失
        >>> loss.backward()
        >>> # 推理时
        >>> outputs = model(input_ids, attention_mask)
        >>> logits = outputs.logits  # [batch_size, 2]
        >>> probs = torch.softmax(logits, dim=-1)  # 转为概率
        >>> preds = torch.argmax(logits, dim=-1)  # 预测类别
    """

    # 加载预训练的MacBERT分类模型
    # 关键点:
    # 1. from_pretrained 会自动下载/加载预训练权重
    # 2. num_labels=2 会自动创建一个 Linear(768, 2) 分类层
    # 3. 模型会保留预训练的BERT权重，只随机初始化分类层
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,

        # Dropout配置（防止过拟合）
        hidden_dropout_prob=dropout_prob,      # 隐藏层之间的dropout
        attention_probs_dropout_prob=dropout_prob,  # 注意力权重的dropout

        # 可选配置（如果需要查看中间层输出）
        output_hidden_states=False,  # 是否输出所有层的hidden states
        output_attentions=False,     # 是否输出注意力权重
    )

    return model


def create_tokenizer(model_name):
    """
    创建MacBERT的Tokenizer（分词器）

    Tokenizer的作用:
        将文本转换为模型可以理解的数字ID序列

    处理流程:
        "如何更换花呗"
        → 分词: ["如何", "更换", "花", "呗"]
        → 转ID: [1234, 5678, 910, 1112]
        → 添加特殊token: [CLS] + text1 + [SEP] + text2 + [SEP]
        → Padding到固定长度

    参数:
        model_name: str
            预训练模型路径（与模型保持一致）

    返回:
        tokenizer: BertTokenizer

    使用示例:
        >>> tokenizer = create_tokenizer('hfl/chinese-macbert-base')
        >>> encoded = tokenizer(
        ...     text='如何更换花呗',
        ...     text_pair='花呗更改',
        ...     max_length=64,
        ...     padding='max_length',
        ...     truncation=True,
        ...     return_tensors='pt'
        ... )
        >>> print(encoded['input_ids'].shape)  # [1, 64]
        >>> print(encoded['attention_mask'].shape)  # [1, 64]
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    return tokenizer


def get_model_info(model):
    """
    获取模型的基本信息（用于调试和了解模型）

    参数:
        model: BertForSequenceClassification

    返回:
        dict: 包含模型各种信息的字典
    """
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 获取模型配置
    config = model.config

    info = {
        '模型类型': model.__class__.__name__,
        '总参数量': f'{total_params:,}',
        '可训练参数量': f'{trainable_params:,}',
        '隐藏层维度': config.hidden_size,
        'Transformer层数': config.num_hidden_layers,
        '注意力头数': config.num_attention_heads,
        '词表大小': config.vocab_size,
        '最大序列长度': config.max_position_embeddings,
        '分类类别数': config.num_labels,
    }

    return info


def print_model_info(model):
    """打印模型信息"""
    info = get_model_info(model)
    print("=" * 60)
    print("模型信息")
    print("=" * 60)
    for key, value in info.items():
        print(f"{key:20s}: {value}")
    print("=" * 60)


# ============================================
# 测试代码
# ============================================
if __name__ == '__main__':
    """
    这个测试代码演示了如何:
    1. 创建模型
    2. 创建tokenizer
    3. 处理文本输入
    4. 前向传播
    5. 查看输出
    """

    print("开始测试MacBERT模型...")

    # 1. 从config获取参数（如果config不可用则使用默认值）
    if config is not None:
        model_name = config.MODEL_NAME
        max_length = config.MAX_LENGTH
        random_seed = config.RANDOM_SEED
        device = config.DEVICE
        print(f"\n使用config.py中的配置:")
        print(f"  模型路径: {model_name}")
        print(f"  最大序列长度: {max_length}")
        print(f"  随机种子: {random_seed}")
        print(f"  设备: {device}")
    else:
        # 如果config不可用，使用默认值
        model_name = '../checkpoints/chinese-macbert-base'
        max_length = 64
        random_seed = 42
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\n使用默认配置:")
        print(f"  模型路径: {model_name}")
        print(f"  最大序列长度: {max_length}")

    # 2. 设置随机种子（保证可复现）
    torch.manual_seed(random_seed)

    try:
        # 3. 创建模型和tokenizer
        print(f"\n正在加载模型...")
        model = create_model(model_name, num_labels=2)
        tokenizer = create_tokenizer(model_name)

        # 将模型移到指定设备
        model = model.to(device)
        print(f"✓ 模型加载成功! (设备: {device})")

        # 4. 打印模型信息
        print_model_info(model)

        # 5. 准备测试数据
        print("\n准备测试数据...")
        text1_list = ["如何更换花呗绑定银行卡", "借呗如何还款"]
        text2_list = ["花呗更改绑定银行卡", "如何开通花呗"]
        labels = torch.tensor([1, 0]).to(device)  # 移到相同设备

        # 6. Tokenize（文本转ID）
        print("\n进行分词...")
        encoded = tokenizer(
            text1_list,
            text2_list,
            max_length=max_length,  # 使用config中的max_length
            padding='max_length',
            truncation=True,
            return_tensors='pt'  # 返回PyTorch tensor
        )

        # 将输入移到相同设备
        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        print(f"input_ids shape: {input_ids.shape}")
        print(f"attention_mask shape: {attention_mask.shape}")

        # 7. 前向传播（训练模式，计算loss）
        print("\n前向传播（训练模式）...")
        model.train()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        print(f"Loss: {outputs.loss.item():.4f}")
        print(f"Logits shape: {outputs.logits.shape}")  # [2, 2]
        print(f"Logits:\n{outputs.logits}")

        # 8. 推理模式（获取预测）
        print("\n前向传播（推理模式）...")
        model.eval()
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        # 9. 获取预测结果
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)  # 转为概率
        preds = torch.argmax(logits, dim=-1)   # 预测类别

        print(f"预测概率:\n{probs}")
        print(f"预测类别: {preds}")
        print(f"真实标签: {labels}")

        print("\n✓ 测试完成!")

    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        print("\n可能的原因:")
        print("1. 模型路径不正确")
        print("2. 缺少transformers库")
        print("3. PyTorch版本不兼容")
        raise
