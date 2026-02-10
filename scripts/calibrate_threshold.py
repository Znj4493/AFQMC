"""
阈值校准脚本 - 在验证集上搜索最优判定阈值 delta

原理：
    当前评估用 logits_是 > logits_否 → 预测"相似"（阈值 delta=0）
    但模型可能存在系统性偏差（如 P(否) 整体偏高），固定阈值 0 非最优
    本脚本搜索最优 delta：logits_是 - logits_否 > delta → 预测"相似"

用法：
    cd src && python calibrate_threshold.py
"""

import sys
import os
import random

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from src.data.data_loader_llm import load_train_data, AFQMCLLMDataset, build_prompt


# --- 配置 ---
SEED = 42
VAL_RATIO = 0.1  # 与 train_lora.py 保持一致
CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "qwen-lora"
)
BASE_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "Qwen1.5-1.8B-Chat"
)


def main():
    print("=" * 60)
    print("阈值校准 - 搜索最优 delta")
    print("=" * 60)

    # --- 1. 设置随机种子（与训练时一致，确保验证集划分相同）---
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --- 2. 加载数据，复现训练时的验证集划分 ---
    print("\n[1/4] 加载数据...")
    all_data = load_train_data()
    random.shuffle(all_data)
    val_size = int(len(all_data) * VAL_RATIO)
    val_data = all_data[:val_size]
    print(f"验证集: {len(val_data)} 条")

    # --- 3. 加载模型 ---
    print("\n[2/4] 加载模型...")

    # 检查 checkpoint 路径：兼容旧格式（直接在 qwen-lora/ 下）和新格式（run_xxx/best_f1/）
    if os.path.exists(os.path.join(CHECKPOINT_DIR, "adapter_model.safetensors")):
        lora_path = CHECKPOINT_DIR
    else:
        # 新格式：找最新的 run 目录
        runs = sorted([
            d for d in os.listdir(CHECKPOINT_DIR)
            if d.startswith("run_") and os.path.isdir(os.path.join(CHECKPOINT_DIR, d))
        ])
        if not runs:
            print("❌ 未找到已训练的模型！请先运行 train_lora.py")
            return
        latest_run = runs[-1]
        lora_path = os.path.join(CHECKPOINT_DIR, latest_run, "best_f1")
        if not os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
            print(f"❌ 未找到模型文件: {lora_path}")
            return

    print(f"加载 LoRA 权重: {lora_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_DIR, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    device = next(model.parameters()).device

    # --- 4. 收集验证集所有样本的 logits ---
    print("\n[3/4] 收集验证集 logits...")
    yes_token_id = tokenizer.encode("是", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("否", add_special_tokens=False)[0]
    print(f"Token ID: '是'={yes_token_id}, '否'={no_token_id}")

    all_logits_yes = []
    all_logits_no = []
    all_labels = []

    with torch.no_grad():
        for i, item in enumerate(tqdm(val_data, desc="收集 logits")):
            text1 = str(item.get("text1", ""))
            text2 = str(item.get("text2", ""))
            label = int(item.get("label", 0))

            prompt = build_prompt(sentence1=text1, sentence2=text2, conclusion="")
            marker_pos = prompt.find("结论：")
            input_text = prompt[:marker_pos + len("结论：")]

            inputs = tokenizer(input_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)

            outputs = model(input_ids=input_ids)
            next_logits = outputs.logits[0, -1, :]

            logit_yes = next_logits[yes_token_id].item()
            logit_no = next_logits[no_token_id].item()

            all_logits_yes.append(logit_yes)
            all_logits_no.append(logit_no)
            all_labels.append(label)

    all_logits_yes = np.array(all_logits_yes)
    all_logits_no = np.array(all_logits_no)
    all_labels = np.array(all_labels)
    diffs = all_logits_yes - all_logits_no

    # --- 5. 搜索最优阈值 ---
    print("\n[4/4] 搜索最优阈值...")
    print(f"\nLogits 差值 (P(是)-P(否)) 统计:")
    print(f"  均值: {diffs.mean():.3f}")
    print(f"  中位数: {np.median(diffs):.3f}")
    print(f"  标准差: {diffs.std():.3f}")
    print(f"  范围: [{diffs.min():.3f}, {diffs.max():.3f}]")
    print(f"  label=1 样本均值: {diffs[all_labels == 1].mean():.3f}")
    print(f"  label=0 样本均值: {diffs[all_labels == 0].mean():.3f}")

    # 网格搜索
    best_f1 = 0.0
    best_delta_f1 = 0.0
    best_acc = 0.0
    best_delta_acc = 0.0

    results = []
    for delta in np.arange(-3.0, 3.0, 0.05):
        preds = (diffs > delta).astype(int)
        f1 = f1_score(all_labels, preds, average="macro")
        acc = accuracy_score(all_labels, preds)
        results.append((delta, f1, acc))

        if f1 > best_f1:
            best_f1 = f1
            best_delta_f1 = delta
        if acc > best_acc:
            best_acc = acc
            best_delta_acc = delta

    # 输出结果
    print("\n" + "=" * 60)
    print("阈值校准结果")
    print("=" * 60)

    # 当前基线（delta=0）
    baseline_preds = (diffs > 0).astype(int)
    baseline_f1 = f1_score(all_labels, baseline_preds, average="macro")
    baseline_acc = accuracy_score(all_labels, baseline_preds)

    print(f"\n当前基线 (delta=0):")
    print(f"  F1-macro: {baseline_f1:.4f}")
    print(f"  Accuracy: {baseline_acc:.4f}")

    print(f"\n最优 F1 阈值:")
    print(f"  delta = {best_delta_f1:.2f}")
    print(f"  F1-macro: {best_f1:.4f} (提升 {best_f1 - baseline_f1:+.4f})")
    opt_f1_preds = (diffs > best_delta_f1).astype(int)
    opt_f1_acc = accuracy_score(all_labels, opt_f1_preds)
    print(f"  Accuracy: {opt_f1_acc:.4f} (提升 {opt_f1_acc - baseline_acc:+.4f})")

    print(f"\n最优 Accuracy 阈值:")
    print(f"  delta = {best_delta_acc:.2f}")
    print(f"  Accuracy: {best_acc:.4f} (提升 {best_acc - baseline_acc:+.4f})")
    opt_acc_preds = (diffs > best_delta_acc).astype(int)
    opt_acc_f1 = f1_score(all_labels, opt_acc_preds, average="macro")
    print(f"  F1-macro: {opt_acc_f1:.4f} (提升 {opt_acc_f1 - baseline_f1:+.4f})")

    # 打印附近区间的详细结果
    print(f"\n阈值-指标对照表（最优 F1 附近）:")
    print(f"  {'delta':>8s}  {'F1-macro':>10s}  {'Accuracy':>10s}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}")
    for delta, f1, acc in results:
        if abs(delta - best_delta_f1) <= 0.55:
            marker = " ←" if abs(delta - best_delta_f1) < 0.03 else ""
            print(f"  {delta:>8.2f}  {f1:>10.4f}  {acc:>10.4f}{marker}")

    print("=" * 60)


if __name__ == "__main__":
    main()
