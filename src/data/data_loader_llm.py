"""
AFQMC LLM 数据加载模块。

设计理念：
1) 将数据读取、prompt 构建、Dataset 封装进行分层，便于复用和测试；
2) 使用统一的 prompt 模板和标签文本映射，确保训练和推理一致。
"""

import json
from typing import List, Dict
from pathlib import Path

import torch
from torch.utils.data import Dataset


PROMPT_TEMPLATE = (
    # Prompt 模板：任务说明 + 输入句子 + 结论
    # 结论使用"是"/"否"单 token 标签，解决之前"相似"(1 token) vs "不相似"(2 tokens) 的
    # loss 信号不对称问题（2 token 的 loss 权重是 1 token 的 2 倍，导致模型偏向多数类）
    "任务：判断两个句子是否表达相同的意思。\n"
    '要求：结论部分只能回答"是"或"否"，不要输出其他内容。\n\n'
    "句子1：{sentence1}\n"
    "句子2：{sentence2}\n\n"
    "结论：{conclusion}"
)

LABEL_TO_TEXT = {
    # 标签到结论文本的映射
    # 使用"是"/"否"：都是单 token，确保两个类别的 loss 信号权重相等
    1: "是",
    0: "否",
}


def load_jsonl(file_path: str) -> List[Dict]:
    """
    读取 JSONL 文件并返回数据列表。

    为什么这样设计：
    - JSONL 每行一个 JSON 对象，边读边解析便于流式处理，也便于定位错误行。

    Args:
        file_path: JSONL 文件路径。

    Returns:
        包含所有样本的列表，每个元素为一条 JSON 记录。
    """
    data: List[Dict] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 逐行解析可以在内存受限时保持稳定，也便于排查非法行。
            item = json.loads(line.strip())
            data.append(item)

    # 打印读取数量有助于初学者验证数据是否完整读取。
    print(f"Loaded {len(data)} samples from {file_path}")
    return data


def load_train_data(data_dir: str = './dataset') -> List[Dict]:
    """
    读取训练集数据（含标签）。

    为什么不直接接收文件路径：
    - 统一的数据目录规范能将配置与代码分离，方便一键切换数据集。

    Args:
        data_dir: 数据目录，默认为 `./dataset`。

    Returns:
        训练集记录列表。
    """
    train_file = Path(data_dir) / 'train.jsonl'
    return load_jsonl(str(train_file))


def load_test_data(data_dir: str = './dataset') -> List[Dict]:
    """
    读取测试集数据（不含标签）。

    为什么和训练集分开函数：
    - 方便在调用层清晰区分有标签和无标签的数据，减少流程逻辑判断。

    Args:
        data_dir: 数据目录，默认为 `./dataset`。

    Returns:
        测试集记录列表。
    """
    test_file = Path(data_dir) / 'test.jsonl'
    return load_jsonl(str(test_file))


def build_prompt(sentence1: str, sentence2: str, conclusion: str) -> str:
    """
    构建用于 LLM 的 prompt 文本。

    Args:
        sentence1: 句子1。
        sentence2: 句子2。
        conclusion: 结论文本（训练时为"是"/"否"，验证时为空字符串）。

    Returns:
        填充后的 prompt 字符串。
    """
    return PROMPT_TEMPLATE.format(
        sentence1=sentence1,
        sentence2=sentence2,
        conclusion=conclusion,
    )


class AFQMCLLMDataset(Dataset):
    """
    AFQMC 数据集封装，返回 LLM prompts。

    Modes:
        - train: include conclusion (label text)
        - infer: leave conclusion empty

    属性说明：
        data: 原始样本列表，用于 __getitem__ 抽取字段。
        mode: 当前模式，决定是否生成结论和标签。

    方法说明：
        __len__: 支持 DataLoader 运行时获取样本数量。
        __getitem__: 将原始数据转为与任务一致的 prompt 样本。
    """

    def __init__(self, data: List[Dict], mode: str = 'train'):
        # 先做参数合法性检查，可以提前报错，避免后续更难排查。
        if mode not in ('train', 'infer'):
            raise ValueError("mode must be 'train' or 'infer'")
        self.data = data
        self.mode = mode

    def __len__(self):
        # Dataset 的标准接口，让 DataLoader 能正确迭代。
        return len(self.data)

    def __getitem__(self, idx):
        # 从原始列表中取出一条记录，使用 get 防止字段缺失导致崩溃。
        item = self.data[idx]
        text1 = str(item.get('text1', ''))
        text2 = str(item.get('text2', ''))
        label = item.get('label', None)

        if self.mode == 'train':
            if label is None:
                # 训练模式下必须有标签，这是为了保持损失计算的合法性。
                raise ValueError("train mode requires label")
            # 结论文本从映射表中获取
            conclusion = LABEL_TO_TEXT[int(label)]
        else:
            # 推理模式下不给结论，让模型自己预测
            conclusion = ""

        prompt = build_prompt(
            sentence1=text1,
            sentence2=text2,
            conclusion=conclusion,
        )

        # 将各字段统一返回为字典，便于 collate 函数批量处理。
        sample = {
            'prompt': prompt,
            'text1': text1,
            'text2': text2,
        }

        if label is not None:
            # 保留整数标签，便于训练时转为 Tensor。
            sample['label'] = int(label)

        return sample


def collate_fn(batch: List[Dict]) -> Dict:
    """
    将 Dataset 返回的多条样本拼接成一批。

    设计说明：
    - 将 prompt 文本保留为列表，方便直接送入 LLM 或后续文本编码模块。
    - 标签用 Tensor 封装，方便与 PyTorch 训练流程对接。

    Args:
        batch: 由 __getitem__ 生成的样本列表。

    Returns:
        批处理后的字典，包含 prompts 与可选 labels。
    """
    prompts = [item['prompt'] for item in batch]
    texts1 = [item['text1'] for item in batch]
    texts2 = [item['text2'] for item in batch]

    output = {
        'prompts': prompts,
        'texts1': texts1,
        'texts2': texts2,
    }

    if 'label' in batch[0]:
        # 检查第一条样本是否有 label，以变通用处理训练和推理。
        labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
        output['labels'] = labels

    return output
