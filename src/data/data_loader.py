"""
数据加载模块
功能：加载AFQMC数据集，提供数据统计和预处理功能
"""

import json
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path


def load_jsonl(file_path: str) -> List[Dict]:
    """
    加载JSONL格式的数据文件

    JSONL格式说明：每行是一个独立的JSON对象
    优点：适合大规模数据，可以逐行读取，节省内存

    Args:
        file_path: JSONL文件路径

    Returns:
        包含所有样本的列表，每个样本是一个字典
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # strip()去除首尾空白字符，json.loads()解析JSON字符串
            item = json.loads(line.strip())
            data.append(item)

    print(f"成功从 {file_path} 加载 {len(data)} 条数据")
    return data


def load_train_data(data_dir: str = './dataset') -> pd.DataFrame:
    """
    加载训练数据并转换为DataFrame格式

    Args:
        data_dir: 数据集所在目录

    Returns:
        包含text1, text2, label三列的DataFrame
    """
    train_file = Path(data_dir)
    data = load_jsonl(str(train_file))
    df = pd.DataFrame(data)

    print(f"\n训练数据统计:")
    print(f"  - 总样本数: {len(df)}")
    print(f"  - 列名: {df.columns.tolist()}")
    print(f"  - 标签分布:")
    print(df['label'].value_counts().sort_index())

    return df


def load_test_data(data_dir: str = './dataset') -> pd.DataFrame:
    """
    加载测试数据并转换为DataFrame格式

    注意：测试数据没有label字段

    Args:
        data_dir: 数据集所在目录

    Returns:
        包含text1, text2两列的DataFrame
    """
    test_file = Path(data_dir) / 'test.jsonl'
    data = load_jsonl(str(test_file))
    df = pd.DataFrame(data)

    print(f"\n测试数据统计:")
    print(f"  - 总样本数: {len(df)}")
    print(f"  - 列名: {df.columns.tolist()}")

    return df


def get_data_statistics(df: pd.DataFrame) -> Dict:
    """
    计算数据集的详细统计信息

    Args:
        df: 数据DataFrame

    Returns:
        统计信息字典
    """
    stats = {
        'total_samples': len(df),
        'text1_lengths': df['text1'].str.len().describe().to_dict(),
        'text2_lengths': df['text2'].str.len().describe().to_dict(),
    }

    # 如果有label字段，统计标签分布
    if 'label' in df.columns:
        label_counts = df['label'].value_counts().to_dict()
        stats['label_distribution'] = label_counts
        stats['label_0_ratio'] = label_counts.get(0, 0) / len(df)
        stats['label_1_ratio'] = label_counts.get(1, 0) / len(df)

    return stats


def print_statistics(stats: Dict):
    """
    打印数据统计信息（格式化输出）

    Args:
        stats: 统计信息字典
    """
    print("\n" + "="*60)
    print("数据集统计信息")
    print("="*60)

    print(f"\n总样本数: {stats['total_samples']}")

    print("\ntext1 长度统计:")
    for key, value in stats['text1_lengths'].items():
        print(f"  {key}: {value:.2f}")

    print("\ntext2 长度统计:")
    for key, value in stats['text2_lengths'].items():
        print(f"  {key}: {value:.2f}")

    if 'label_distribution' in stats:
        print("\n标签分布:")
        for label, count in stats['label_distribution'].items():
            ratio = count / stats['total_samples'] * 100
            print(f"  label {label}: {count} ({ratio:.2f}%)")

        print(f"\n数据不平衡比例: {stats['label_0_ratio']:.2f} : {stats['label_1_ratio']:.2f}")
        print(f"  这是一个{'不平衡' if abs(stats['label_0_ratio'] - 0.5) > 0.1 else '相对平衡'}的数据集")

    print("="*60)


def split_train_val(df: pd.DataFrame,
                     val_ratio: float = 0.1,
                     random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    将训练数据切分为训练集和验证集

    重要概念：
    - 训练集(train)：用于训练模型参数
    - 验证集(validation)：用于调整超参数和早停
    - 测试集(test)：最终评估模型性能（比赛中不可见标签）

    Args:
        df: 完整的训练数据
        val_ratio: 验证集比例（默认10%）
        random_state: 随机种子，确保可复现

    Returns:
        (train_df, val_df) 元组
    """
    from sklearn.model_selection import train_test_split

    # stratify=df['label'] 确保训练集和验证集的标签分布一致
    train_df, val_df = train_test_split(
        df,
        test_size=val_ratio,
        random_state=random_state,
        stratify=df['label']  # 分层采样，保持标签比例
    )

    print(f"\n数据集切分完成:")
    print(f"  训练集: {len(train_df)} 样本")
    print(f"  验证集: {len(val_df)} 样本")
    print(f"  验证集比例: {val_ratio*100:.1f}%")

    # 验证标签分布是否一致
    print(f"\n训练集标签分布: {train_df['label'].value_counts().to_dict()}")
    print(f"验证集标签分布: {val_df['label'].value_counts().to_dict()}")

    return train_df, val_df


def apply_data_augmentation(
    train_df: pd.DataFrame,
    strategy: str = 'minority',
    alpha: float = 0.1,
    num_aug: int = 1,
    augment_both: bool = False
) -> pd.DataFrame:
    """
    对训练数据应用EDA数据增强

    策略说明：
    - 'minority': 只增强少数类（label=1），缓解类别不平衡
    - 'all': 增强所有样本
    - 'balanced': 只增强少数类，使其数量接近多数类

    Args:
        train_df: 训练数据DataFrame
        strategy: 增强策略
        alpha: EDA增强强度（0-1之间）
        num_aug: 每个样本生成的增强样本数
        augment_both: 是否同时增强两个句子

    Returns:
        增强后的训练数据DataFrame
    """
    from src.data.data_augmentation import augment_sentence_pair

    # 统计原始数据
    label_counts = train_df['label'].value_counts()
    print(f"\n{'='*60}")
    print("数据增强")
    print(f"{'='*60}")
    print(f"原始训练集: {len(train_df)} 样本")
    print(f"  - Label 0: {label_counts.get(0, 0)} 样本")
    print(f"  - Label 1: {label_counts.get(1, 0)} 样本")

    # 确定需要增强的样本
    if strategy == 'minority':
        # 只增强少数类（label=1）
        minority_label = 1
        samples_to_augment = train_df[train_df['label'] == minority_label]
        samples_to_keep = train_df[train_df['label'] != minority_label]
        print(f"\n增强策略: 只增强少数类 (label={minority_label})")
        print(f"需要增强的样本数: {len(samples_to_augment)}")

    elif strategy == 'balanced':
        # 增强少数类，使其数量接近多数类
        minority_label = label_counts.idxmin()
        majority_label = label_counts.idxmax()
        minority_count = label_counts[minority_label]
        majority_count = label_counts[majority_label]

        # 计算需要生成多少增强样本才能平衡
        samples_needed = majority_count - minority_count
        num_aug_per_sample = max(1, samples_needed // minority_count)

        samples_to_augment = train_df[train_df['label'] == minority_label]
        samples_to_keep = train_df[train_df['label'] == majority_label]

        print(f"\n增强策略: 平衡少数类 (label={minority_label})")
        print(f"少数类样本: {minority_count}, 多数类样本: {majority_count}")
        print(f"需要生成约 {samples_needed} 个增强样本")
        print(f"每个原始样本生成 {num_aug_per_sample} 个增强样本")

        # 更新 num_aug
        num_aug = num_aug_per_sample

    elif strategy == 'all':
        # 增强所有样本
        samples_to_augment = train_df
        samples_to_keep = pd.DataFrame(columns=train_df.columns)
        print(f"\n增强策略: 增强所有样本")

    else:
        raise ValueError(f"不支持的增强策略: {strategy}")

    # 执行数据增强
    print(f"\n开始生成增强样本...")
    print(f"  - 增强强度 (alpha): {alpha}")
    print(f"  - 每个样本生成增强数: {num_aug}")
    print(f"  - 同时增强两个句子: {augment_both}")

    augmented_samples = []
    total_augmented = 0

    for idx, row in samples_to_augment.iterrows():
        # 对每个样本进行增强
        aug_pairs = augment_sentence_pair(
            sentence1=row['text1'],
            sentence2=row['text2'],
            label=row['label'],
            alpha=alpha,
            num_aug=num_aug,
            augment_both=augment_both
        )

        # 将增强后的样本添加到列表
        for sent1, sent2, label in aug_pairs:
            augmented_samples.append({
                'text1': sent1,
                'text2': sent2,
                'label': label
            })
            total_augmented += 1

    print(f"  生成了 {total_augmented} 个增强样本")

    # 合并原始数据和增强数据
    aug_df = pd.DataFrame(augmented_samples)
    combined_df = pd.concat([train_df, aug_df], ignore_index=True)

    # 打乱数据
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 统计增强后的数据
    final_label_counts = combined_df['label'].value_counts()
    print(f"\n增强后的训练集: {len(combined_df)} 样本 (+{len(combined_df) - len(train_df)})")
    print(f"  - Label 0: {final_label_counts.get(0, 0)} 样本")
    print(f"  - Label 1: {final_label_counts.get(1, 0)} 样本")

    # 计算增强比例
    if strategy in ['minority', 'balanced']:
        minority_label = 1 if strategy == 'minority' else label_counts.idxmin()
        original_minority = label_counts[minority_label]
        final_minority = final_label_counts[minority_label]
        augmentation_ratio = final_minority / original_minority
        print(f"  少数类增强倍数: {augmentation_ratio:.2f}x")

    print(f"{'='*60}\n")

    return combined_df


def display_samples(df: pd.DataFrame, n: int = 3, label: int = None):
    """
    展示数据样本（用于理解数据）

    Args:
        df: 数据DataFrame
        n: 展示样本数量
        label: 如果指定，只展示该标签的样本
    """
    print("\n" + "="*60)
    print(f"数据样本展示 (共{n}条)")
    print("="*60)

    if label is not None and 'label' in df.columns:
        samples = df[df['label'] == label].head(n)
        print(f"\n仅展示 label={label} 的样本:")
    else:
        samples = df.head(n)

    for idx, row in samples.iterrows():
        print(f"\n样本 #{idx + 1}:")
        print(f"  text1: {row['text1']}")
        print(f"  text2: {row['text2']}")
        if 'label' in row:
            print(f"  label: {row['label']} ({'相似' if row['label'] == 1 else '不相似'})")
        print("-" * 60)


# 测试代码（如果直接运行此文件）
if __name__ == "__main__":
    print("AFQMC 数据加载模块测试\n")

    # 1. 加载训练数据
    train_df = load_train_data('../dataset')

    # 2. 计算并展示统计信息
    stats = get_data_statistics(train_df)
    print_statistics(stats)

    # 3. 展示样本
    display_samples(train_df, n=3, label=1)  # 展示相似样本
    display_samples(train_df, n=3, label=0)  # 展示不相似样本

    # 4. 切分训练集和验证集
    train_split, val_split = split_train_val(train_df, val_ratio=0.1)

    # 5. 加载测试数据
    test_df = load_test_data('../dataset')

    print("\n✅ 数据加载模块测试完成！")
