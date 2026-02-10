"""
分析GLM-4.6和Qwen3-Next预测不一致的样本

找出两个大模型预测不同的文本对，分析差异原因
"""

import json
import pandas as pd

def load_predictions(file_path):
    """加载预测结果"""
    predictions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            predictions.append(int(data['response']))
    return predictions

def load_test_data(file_path):
    """加载测试集数据"""
    test_data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            test_data.append(data)
    return test_data

def main():
    # 文件路径
    results_dir = '../results'
    dataset_dir = '../dataset'

    glm_file = f'{results_dir}/result_glm4.6.jsonl'
    qwen_file = f'{results_dir}/qwen3_next_result.jsonl'
    test_file = f'{dataset_dir}/test.jsonl'

    # 加载数据
    print("=" * 80)
    print("加载数据...")
    print("=" * 80)

    glm_preds = load_predictions(glm_file)
    qwen_preds = load_predictions(qwen_file)
    test_data = load_test_data(test_file)

    print(f"✓ GLM-4.6预测: {len(glm_preds)} 条")
    print(f"✓ Qwen3-Next预测: {len(qwen_preds)} 条")
    print(f"✓ 测试集数据: {len(test_data)} 条")

    # 找出预测不一致的样本
    print("\n" + "=" * 80)
    print("分析预测不一致的样本")
    print("=" * 80)

    disagreements = []
    for idx, (glm_pred, qwen_pred, test_item) in enumerate(zip(glm_preds, qwen_preds, test_data)):
        if glm_pred != qwen_pred:
            disagreements.append({
                'index': idx,
                'text1': test_item['text1'],
                'text2': test_item['text2'],
                'glm_pred': glm_pred,
                'qwen_pred': qwen_pred
            })

    print(f"\n找到 {len(disagreements)} 个预测不一致的样本 ({len(disagreements)/len(glm_preds)*100:.2f}%)")

    # 统计不一致的类型
    glm0_qwen1 = sum(1 for d in disagreements if d['glm_pred'] == 0 and d['qwen_pred'] == 1)
    glm1_qwen0 = sum(1 for d in disagreements if d['glm_pred'] == 1 and d['qwen_pred'] == 0)

    print(f"\n不一致类型分布:")
    print(f"  GLM预测0, Qwen预测1: {glm0_qwen1} ({glm0_qwen1/len(disagreements)*100:.2f}%)")
    print(f"  GLM预测1, Qwen预测0: {glm1_qwen0} ({glm1_qwen0/len(disagreements)*100:.2f}%)")

    # 显示前20个不一致的样本
    print("\n" + "=" * 80)
    print("前20个预测不一致的样本")
    print("=" * 80)

    for i, item in enumerate(disagreements[:20], 1):
        print(f"\n【样本 {i}】(索引: {item['index']})")
        print(f"句子1: {item['text1']}")
        print(f"句子2: {item['text2']}")
        print(f"GLM-4.6预测: {item['glm_pred']} | Qwen3-Next预测: {item['qwen_pred']}")
        print("-" * 80)

    # 保存所有不一致样本到CSV
    print("\n" + "=" * 80)
    print("保存详细结果")
    print("=" * 80)

    df = pd.DataFrame(disagreements)
    output_file = f'{results_dir}/glm_qwen_disagreements.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')  # 使用utf-8-sig以便Excel正确显示中文
    print(f"\n✓ 所有 {len(disagreements)} 个不一致样本已保存至: {output_file}")

    # 分析文本长度特征
    print("\n" + "=" * 80)
    print("文本长度分析")
    print("=" * 80)

    df['len1'] = df['text1'].str.len()
    df['len2'] = df['text2'].str.len()
    df['total_len'] = df['len1'] + df['len2']

    print(f"\n不一致样本的文本长度统计:")
    print(f"  句子1平均长度: {df['len1'].mean():.1f} 字")
    print(f"  句子2平均长度: {df['len2'].mean():.1f} 字")
    print(f"  总长度平均: {df['total_len'].mean():.1f} 字")
    print(f"  总长度中位数: {df['total_len'].median():.1f} 字")
    print(f"  总长度范围: {df['total_len'].min()} - {df['total_len'].max()} 字")

    # 按预测类型分组分析
    print("\n" + "=" * 80)
    print("按预测类型分组分析")
    print("=" * 80)

    glm0_qwen1_df = df[(df['glm_pred'] == 0) & (df['qwen_pred'] == 1)]
    glm1_qwen0_df = df[(df['glm_pred'] == 1) & (df['qwen_pred'] == 0)]

    print(f"\nGLM预测0, Qwen预测1 ({len(glm0_qwen1_df)} 个样本):")
    print(f"  平均总长度: {glm0_qwen1_df['total_len'].mean():.1f} 字")
    print("\n  示例:")
    for i, row in glm0_qwen1_df.head(3).iterrows():
        print(f"    句子1: {row['text1']}")
        print(f"    句子2: {row['text2']}")
        print()

    print(f"\nGLM预测1, Qwen预测0 ({len(glm1_qwen0_df)} 个样本):")
    print(f"  平均总长度: {glm1_qwen0_df['total_len'].mean():.1f} 字")
    print("\n  示例:")
    for i, row in glm1_qwen0_df.head(3).iterrows():
        print(f"    句子1: {row['text1']}")
        print(f"    句子2: {row['text2']}")
        print()

    print("=" * 80)
    print("分析完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()