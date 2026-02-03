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
