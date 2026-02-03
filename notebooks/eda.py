"""
探索性数据分析 (Exploratory Data Analysis, EDA)

目标：深入理解AFQMC数据集的特征，为模型设计提供依据
"""

import sys
import os
from pathlib import Path

# 自动添加项目根目录到路径（无论从哪里运行都能正确导入）
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re

# 导入我们自己的数据加载模块
from src.data_loader import load_train_data, load_test_data, get_data_statistics

# Set matplotlib font (use English to avoid font issues)
plt.rcParams['axes.unicode_minus'] = False

# Set plot style
sns.set_style("whitegrid")


def analyze_text_length(df: pd.DataFrame):
    """
    分析文本长度分布

    知识点：
    - 文本长度影响模型的max_length参数设置
    - 了解长度分布可以帮助我们优化padding策略
    """
    print("\n" + "="*60)
    print("1. 文本长度分析")
    print("="*60)

    # 计算文本长度
    df['text1_len'] = df['text1'].str.len()
    df['text2_len'] = df['text2'].str.len()
    df['total_len'] = df['text1_len'] + df['text2_len']

    # 统计信息
    print("\ntext1 长度统计:")
    print(df['text1_len'].describe())

    print("\ntext2 长度统计:")
    print(df['text2_len'].describe())

    print("\n总长度 (text1 + text2) 统计:")
    print(df['total_len'].describe())

    # 关键发现
    max_len_95 = df['total_len'].quantile(0.95)
    max_len_99 = df['total_len'].quantile(0.99)

    print(f"\n💡 关键发现:")
    print(f"  - 95% 的样本总长度 <= {max_len_95:.0f} 字符")
    print(f"  - 99% 的样本总长度 <= {max_len_99:.0f} 字符")
    print(f"  - 建议的 max_length 参数: 128 (覆盖大部分样本)")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # text1 length distribution
    axes[0].hist(df['text1_len'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Text Length')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Text1 Length Distribution')
    axes[0].axvline(df['text1_len'].mean(), color='red', linestyle='--', label=f'Mean={df["text1_len"].mean():.1f}')
    axes[0].legend()

    # text2 length distribution
    axes[1].hist(df['text2_len'], bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_xlabel('Text Length')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Text2 Length Distribution')
    axes[1].axvline(df['text2_len'].mean(), color='red', linestyle='--', label=f'Mean={df["text2_len"].mean():.1f}')
    axes[1].legend()

    # Total length distribution
    axes[2].hist(df['total_len'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[2].set_xlabel('Text Length')
    axes[2].set_ylabel('Count')
    axes[2].set_title('Total Length (text1 + text2)')
    axes[2].axvline(max_len_95, color='red', linestyle='--', label=f'95th={max_len_95:.0f}')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('results/text_length_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表已保存至 results/text_length_distribution.png")

    return df


def analyze_label_distribution(df: pd.DataFrame):
    """
    分析标签分布

    知识点：
    - 类别不平衡会影响模型训练
    - 可以通过调整类别权重、使用Focal Loss等方法处理
    """
    print("\n" + "="*60)
    print("2. 标签分布分析")
    print("="*60)

    label_counts = df['label'].value_counts().sort_index()
    label_ratios = df['label'].value_counts(normalize=True).sort_index()

    print("\n标签统计:")
    for label in [0, 1]:
        count = label_counts[label]
        ratio = label_ratios[label]
        print(f"  label {label} ({'不相似' if label == 0 else '相似  '}): {count:5d} ({ratio*100:5.2f}%)")

    # 计算不平衡比例
    imbalance_ratio = label_counts[0] / label_counts[1]
    print(f"\n不平衡比例: {imbalance_ratio:.2f} : 1")

    if imbalance_ratio > 2:
        print(f"⚠️  数据集存在显著的类别不平衡！")
        print(f"   建议处理方法:")
        print(f"   1. 使用类别权重 (class_weight)")
        print(f"   2. 使用 Focal Loss")
        print(f"   3. 数据重采样（过采样/欠采样）")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bar chart
    axes[0].bar([0, 1], label_counts.values, color=['#FF6B6B', '#4ECDC4'], edgecolor='black')
    axes[0].set_xlabel('Label')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Label Distribution (Bar Chart)')
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(['0 (Dissimilar)', '1 (Similar)'])

    # Add value labels
    for i, v in enumerate(label_counts.values):
        axes[0].text(i, v + 500, str(v), ha='center', va='bottom', fontweight='bold')

    # Pie chart
    axes[1].pie(label_counts.values,
                labels=['0 (Dissimilar)', '1 (Similar)'],
                autopct='%1.1f%%',
                colors=['#FF6B6B', '#4ECDC4'],
                startangle=90)
    axes[1].set_title('Label Distribution (Pie Chart)')

    plt.tight_layout()
    plt.savefig('results/label_distribution.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表已保存至 results/label_distribution.png")


def analyze_text_by_label(df: pd.DataFrame):
    """
    对比不同标签的文本特征

    目标：发现相似和不相似文本对的区别
    """
    print("\n" + "="*60)
    print("3. 不同标签的文本特征对比")
    print("="*60)

    # 分组统计
    for label in [0, 1]:
        subset = df[df['label'] == label]
        print(f"\nlabel {label} ({'不相似' if label == 0 else '相似  '}) 的文本长度:")
        print(f"  text1: 平均 {subset['text1_len'].mean():.1f}, 中位数 {subset['text1_len'].median():.1f}")
        print(f"  text2: 平均 {subset['text2_len'].mean():.1f}, 中位数 {subset['text2_len'].median():.1f}")
        print(f"  总长: 平均 {subset['total_len'].mean():.1f}, 中位数 {subset['total_len'].median():.1f}")

    # Visualization comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # text1 length comparison
    df[df['label'] == 0]['text1_len'].hist(bins=30, alpha=0.5, label='label 0', ax=axes[0], color='red')
    df[df['label'] == 1]['text1_len'].hist(bins=30, alpha=0.5, label='label 1', ax=axes[0], color='blue')
    axes[0].set_xlabel('text1 Length')
    axes[0].set_ylabel('Count')
    axes[0].set_title('text1 Length Distribution by Label')
    axes[0].legend()

    # Total length comparison
    df[df['label'] == 0]['total_len'].hist(bins=30, alpha=0.5, label='label 0', ax=axes[1], color='red')
    df[df['label'] == 1]['total_len'].hist(bins=30, alpha=0.5, label='label 1', ax=axes[1], color='blue')
    axes[1].set_xlabel('Total Length')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Total Length Distribution by Label')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('results/length_by_label.png', dpi=300, bbox_inches='tight')
    print(f"\n✅ 图表已保存至 results/length_by_label.png")


def check_data_quality(df: pd.DataFrame):
    """
    数据质量检查

    检查项：
    - 缺失值
    - 重复样本
    - 异常值
    """
    print("\n" + "="*60)
    print("4. 数据质量检查")
    print("="*60)

    # 1. 缺失值检查
    print("\n缺失值统计:")
    missing = df.isnull().sum()
    print(missing)

    if missing.sum() == 0:
        print("✅ 没有发现缺失值")

    # 2. 重复样本检查
    duplicates = df.duplicated().sum()
    print(f"\n重复样本数量: {duplicates}")

    if duplicates > 0:
        print(f"⚠️  发现 {duplicates} 条重复样本，建议去重")
    else:
        print("✅ 没有发现重复样本")

    # 3. 空文本检查
    empty_text1 = (df['text1'].str.strip() == '').sum()
    empty_text2 = (df['text2'].str.strip() == '').sum()

    print(f"\n空文本统计:")
    print(f"  text1 为空: {empty_text1} 条")
    print(f"  text2 为空: {empty_text2} 条")

    if empty_text1 + empty_text2 == 0:
        print("✅ 没有发现空文本")

    # 4. 异常长文本检查
    very_long_threshold = 200
    very_long = df[df['total_len'] > very_long_threshold]

    print(f"\n异常长文本 (总长度 > {very_long_threshold}):")
    print(f"  数量: {len(very_long)} ({len(very_long)/len(df)*100:.2f}%)")

    if len(very_long) > 0:
        print(f"\n  最长的3个样本:")
        for idx, row in very_long.nlargest(3, 'total_len').iterrows():
            print(f"    长度={row['total_len']:.0f}, text1={row['text1'][:30]}..., text2={row['text2'][:30]}...")


def sample_analysis(df: pd.DataFrame):
    """
    样本分析：展示典型的相似和不相似样本
    """
    print("\n" + "="*60)
    print("5. 典型样本分析")
    print("="*60)

    print("\n【相似样本示例 (label=1)】")
    similar_samples = df[df['label'] == 1].sample(5, random_state=42)
    for i, (idx, row) in enumerate(similar_samples.iterrows(), 1):
        print(f"\n样本 {i}:")
        print(f"  text1: {row['text1']}")
        print(f"  text2: {row['text2']}")

    print("\n\n【不相似样本示例 (label=0)】")
    dissimilar_samples = df[df['label'] == 0].sample(5, random_state=42)
    for i, (idx, row) in enumerate(dissimilar_samples.iterrows(), 1):
        print(f"\n样本 {i}:")
        print(f"  text1: {row['text1']}")
        print(f"  text2: {row['text2']}")


def generate_eda_report(df: pd.DataFrame):
    """
    生成EDA总结报告
    """
    print("\n" + "="*80)
    print(" "*25 + "EDA 总结报告")
    print("="*80)

    stats = get_data_statistics(df)

    print(f"\n📊 数据集规模:")
    print(f"  - 总样本数: {stats['total_samples']:,}")
    print(f"  - Label 0 (不相似): {stats['label_distribution'][0]:,} ({stats['label_0_ratio']*100:.1f}%)")
    print(f"  - Label 1 (相似):   {stats['label_distribution'][1]:,} ({stats['label_1_ratio']*100:.1f}%)")

    print(f"\n📏 文本长度特征:")
    print(f"  - text1 平均长度: {df['text1_len'].mean():.1f} 字符")
    print(f"  - text2 平均长度: {df['text2_len'].mean():.1f} 字符")
    print(f"  - 总长度 95% 分位: {df['total_len'].quantile(0.95):.0f} 字符")

    print(f"\n💡 模型设计建议:")
    print(f"  1. max_length 设置: 128 (可覆盖大部分样本)")
    print(f"  2. 类别不平衡处理: 使用类别权重或 Focal Loss")
    print(f"  3. 验证集划分: 建议使用分层采样 (stratified split)")
    print(f"  4. 评估指标: 除了准确率，还应关注 F1-score、AUC 等")

    print("\n" + "="*80)
    print("EDA 分析完成！所有图表已保存至 results/ 目录")
    print("="*80)


def main():
    """主函数：执行完整的EDA流程"""
    print("\n" + "🔍 "*20)
    print("AFQMC 数据集探索性分析 (EDA)")
    print("🔍 "*20)

    # 使用项目根目录的绝对路径
    dataset_path = project_root / 'dataset'

    # 加载训练数据
    df = load_train_data(str(dataset_path))

    # 1. 文本长度分析
    df = analyze_text_length(df)

    # 2. 标签分布分析
    analyze_label_distribution(df)

    # 3. 不同标签的文本特征
    analyze_text_by_label(df)

    # 4. 数据质量检查
    check_data_quality(df)

    # 5. 样本分析
    sample_analysis(df)

    # 6. 生成总结报告
    generate_eda_report(df)

    print("\n\n✅ EDA 分析全部完成！")
    print("\n下一步: 开始构建模型 🚀")


if __name__ == "__main__":
    # 确保results目录存在（使用项目根目录）
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)

    # 切换工作目录到项目根目录（确保图片保存路径正确）
    os.chdir(project_root)

    main()
