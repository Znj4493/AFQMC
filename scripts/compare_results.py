"""
对比3个模型的预测结果

对比指标：
1. 预测分布（0和1的数量）
2. 预测一致性（3个模型预测相同的样本数）
3. 预测差异分析（哪些样本预测不一致）
"""

import json
import pandas as pd
from collections import Counter

def load_predictions(file_path):
    """加载预测结果"""
    predictions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            predictions.append(int(data['response']))
    return predictions

def main():
    # 文件路径
    results_dir = '../results'
    macbert_file = f'{results_dir}/macbert_result.jsonl'
    glm_file = f'{results_dir}/result_glm4.6.jsonl'
    qwen_file = f'{results_dir}/qwen3_next_result.jsonl'

    # 加载预测结果
    print("=" * 80)
    print("加载预测结果...")
    print("=" * 80)

    macbert_preds = load_predictions(macbert_file)
    glm_preds = load_predictions(glm_file)
    qwen_preds = load_predictions(qwen_file)

    print(f"✓ MacBERT: {len(macbert_preds)} 条预测")
    print(f"✓ GLM-4.6: {len(glm_preds)} 条预测")
    print(f"✓ Qwen3-Next: {len(qwen_preds)} 条预测")

    # 1. 预测分布统计
    print("\n" + "=" * 80)
    print("1. 预测分布统计")
    print("=" * 80)

    macbert_dist = Counter(macbert_preds)
    glm_dist = Counter(glm_preds)
    qwen_dist = Counter(qwen_preds)

    print(f"\nMacBERT:")
    print(f"  Label 0: {macbert_dist[0]} ({macbert_dist[0]/len(macbert_preds)*100:.2f}%)")
    print(f"  Label 1: {macbert_dist[1]} ({macbert_dist[1]/len(macbert_preds)*100:.2f}%)")
    print(f"  比例: {macbert_dist[0]/macbert_dist[1]:.2f}:1")

    print(f"\nGLM-4.6:")
    print(f"  Label 0: {glm_dist[0]} ({glm_dist[0]/len(glm_preds)*100:.2f}%)")
    print(f"  Label 1: {glm_dist[1]} ({glm_dist[1]/len(glm_preds)*100:.2f}%)")
    print(f"  比例: {glm_dist[0]/glm_dist[1]:.2f}:1")

    print(f"\nQwen3-Next:")
    print(f"  Label 0: {qwen_dist[0]} ({qwen_dist[0]/len(qwen_preds)*100:.2f}%)")
    print(f"  Label 1: {qwen_dist[1]} ({qwen_dist[1]/len(qwen_preds)*100:.2f}%)")
    print(f"  比例: {qwen_dist[0]/qwen_dist[1]:.2f}:1")

    # 2. 预测一致性分析
    print("\n" + "=" * 80)
    print("2. 预测一致性分析")
    print("=" * 80)

    # 统计3个模型预测完全一致的样本
    all_agree = sum(1 for m, g, q in zip(macbert_preds, glm_preds, qwen_preds)
                    if m == g == q)

    # 统计两两一致的样本
    macbert_glm_agree = sum(1 for m, g in zip(macbert_preds, glm_preds) if m == g)
    macbert_qwen_agree = sum(1 for m, q in zip(macbert_preds, qwen_preds) if m == q)
    glm_qwen_agree = sum(1 for g, q in zip(glm_preds, qwen_preds) if g == q)

    print(f"\n3个模型完全一致: {all_agree} ({all_agree/len(macbert_preds)*100:.2f}%)")
    print(f"\nMacBERT vs GLM-4.6 一致: {macbert_glm_agree} ({macbert_glm_agree/len(macbert_preds)*100:.2f}%)")
    print(f"MacBERT vs Qwen3-Next 一致: {macbert_qwen_agree} ({macbert_qwen_agree/len(macbert_preds)*100:.2f}%)")
    print(f"GLM-4.6 vs Qwen3-Next 一致: {glm_qwen_agree} ({glm_qwen_agree/len(macbert_preds)*100:.2f}%)")

    # 3. 预测差异分析
    print("\n" + "=" * 80)
    print("3. 预测差异分析")
    print("=" * 80)

    # 统计不同的预测组合
    prediction_patterns = Counter()
    for m, g, q in zip(macbert_preds, glm_preds, qwen_preds):
        pattern = f"M:{m} G:{g} Q:{q}"
        prediction_patterns[pattern] += 1

    print("\n预测组合分布（前10种）:")
    for pattern, count in prediction_patterns.most_common(10):
        print(f"  {pattern}: {count} ({count/len(macbert_preds)*100:.2f}%)")

    # 4. 找出分歧最大的样本
    print("\n" + "=" * 80)
    print("4. 分歧样本分析")
    print("=" * 80)

    # 3个模型预测都不同的情况（理论上不可能，因为只有0和1两个类别）
    # 但可以找出2:1分歧的样本
    split_votes = []
    for idx, (m, g, q) in enumerate(zip(macbert_preds, glm_preds, qwen_preds)):
        if not (m == g == q):  # 不是完全一致
            votes = [m, g, q]
            if votes.count(0) == 1 or votes.count(1) == 1:  # 2:1的情况
                split_votes.append({
                    'index': idx,
                    'macbert': m,
                    'glm': g,
                    'qwen': q,
                    'majority': max(set(votes), key=votes.count)
                })

    print(f"\n2:1分歧样本数: {len(split_votes)} ({len(split_votes)/len(macbert_preds)*100:.2f}%)")

    # 统计每个模型作为少数派的次数
    macbert_minority = sum(1 for s in split_votes if s['macbert'] != s['majority'])
    glm_minority = sum(1 for s in split_votes if s['glm'] != s['majority'])
    qwen_minority = sum(1 for s in split_votes if s['qwen'] != s['majority'])

    print(f"\n作为少数派的次数:")
    print(f"  MacBERT: {macbert_minority} ({macbert_minority/len(split_votes)*100:.2f}%)")
    print(f"  GLM-4.6: {glm_minority} ({glm_minority/len(split_votes)*100:.2f}%)")
    print(f"  Qwen3-Next: {qwen_minority} ({qwen_minority/len(split_votes)*100:.2f}%)")

    # 5. 生成对比报告
    print("\n" + "=" * 80)
    print("5. 生成详细对比报告")
    print("=" * 80)

    # 创建DataFrame保存所有预测
    df = pd.DataFrame({
        'index': range(len(macbert_preds)),
        'macbert': macbert_preds,
        'glm4.6': glm_preds,
        'qwen3_next': qwen_preds
    })

    # 添加一致性标记
    df['all_agree'] = (df['macbert'] == df['glm4.6']) & (df['macbert'] == df['qwen3_next'])
    df['majority_vote'] = df[['macbert', 'glm4.6', 'qwen3_next']].mode(axis=1)[0]

    # 保存对比结果
    output_file = f'{results_dir}/model_comparison.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n✓ 详细对比结果已保存至: {output_file}")

    # 6. 总结
    print("\n" + "=" * 80)
    print("6. 总结")
    print("=" * 80)

    print(f"\n模型一致性排名（两两对比）:")
    agreements = [
        ('MacBERT vs GLM-4.6', macbert_glm_agree),
        ('MacBERT vs Qwen3-Next', macbert_qwen_agree),
        ('GLM-4.6 vs Qwen3-Next', glm_qwen_agree)
    ]
    agreements.sort(key=lambda x: x[1], reverse=True)
    for i, (pair, count) in enumerate(agreements, 1):
        print(f"  {i}. {pair}: {count} ({count/len(macbert_preds)*100:.2f}%)")

    print(f"\n预测分布对比:")
    print(f"  MacBERT: Label 0/1 = {macbert_dist[0]/macbert_dist[1]:.2f}:1")
    print(f"  GLM-4.6: Label 0/1 = {glm_dist[0]/glm_dist[1]:.2f}:1")
    print(f"  Qwen3-Next: Label 0/1 = {qwen_dist[0]/qwen_dist[1]:.2f}:1")
    print(f"  训练集分布: Label 0/1 ≈ 2.23:1 (参考)")

    print("\n" + "=" * 80)
    print("对比分析完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()