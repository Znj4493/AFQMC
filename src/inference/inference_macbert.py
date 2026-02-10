"""
MacBERT 测试集推理脚本

用法：
    cd src && python inference_macbert.py

输出：
    results/macbert_result.jsonl
    每行格式：{"response": "0"} 或 {"response": "1"}
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
from tqdm import tqdm
from config.config import config
from src.models.macbert import create_model, create_tokenizer
from src.utils.utils import load_checkpoint


def main():
    print("=" * 60)
    print("MacBERT 测试集推理")
    print("=" * 60)

    device = config.DEVICE
    print(f"设备: {device}")

    # --- 1. 加载模型 ---
    print("\n[1/4] 加载模型...")
    model = create_model(config.MODEL_NAME, num_labels=2)
    model = model.to(device)

    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
    load_checkpoint(model, checkpoint_path, device=device)
    model.eval()
    print(f"已加载: {checkpoint_path}")

    # --- 2. 加载 tokenizer ---
    print("\n[2/4] 加载 tokenizer...")
    tokenizer = create_tokenizer(config.MODEL_NAME)

    # --- 3. 加载测试集 ---
    print("\n[3/4] 加载测试集...")
    test_data = []
    with open(config.TEST_FILE, "r", encoding="utf-8") as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    print(f"测试集: {len(test_data)} 条")

    # --- 4. 推理 ---
    print("\n[4/4] 开始推理...")
    results = []

    with torch.no_grad():
        for item in tqdm(test_data, desc="推理中"):
            text1 = str(item.get("text1", ""))
            text2 = str(item.get("text2", ""))

            encoding = tokenizer(
                text1,
                text2,
                max_length=config.MAX_LENGTH,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                verbose=False,
            )
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=-1).item()
            results.append(pred)

    # --- 5. 写入结果 ---
    output_path = os.path.join(config.RESULTS_DIR, "macbert_result.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for pred in results:
            f.write(json.dumps({"response": str(pred)}, ensure_ascii=False) + "\n")

    # 统计
    count_0 = results.count(0)
    count_1 = results.count(1)
    print(f"\n推理完成!")
    print(f"  总数: {len(results)}")
    print(f"  预测 0: {count_0} ({count_0/len(results)*100:.1f}%)")
    print(f"  预测 1: {count_1} ({count_1/len(results)*100:.1f}%)")
    print(f"  结果保存: {output_path}")


if __name__ == "__main__":
    main()
