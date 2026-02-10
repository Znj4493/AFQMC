"""
LoRA 微调训练脚本 - Qwen-1.5-1.8B

核心设计：
1. 使用 Causal LM 的方式训练（预测下一个 token）
2. 只对"结论"部分计算 loss（不让模型学习 prompt 本身）
3. AMP 混合精度 + 梯度累积，适配 8GB 显存
4. 验证集评估使用 F1-macro，与 MacBERT 对比口径一致

使用方式:
    cd src && python train_lora.py

硬件约束:
    RTX 4060 8GB VRAM
"""

import sys
import os

# 添加项目根目录到路径，确保能导入 src 下的模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# 导入项目模块
from src.models.llm import load_model_with_lora
from src.data.data_loader_llm import (
    load_train_data,
    AFQMCLLMDataset,
    LABEL_TO_TEXT,
    build_prompt,
)


# ============================================
# 训练超参数（集中管理，方便调参）
# ============================================

# --- 数据相关 ---
VAL_RATIO = 0.1          # 验证集比例：10% 的数据用于验证
MAX_LENGTH = 256          # 最大 token 长度：prompt 较长，需要比 MacBERT 的 64 更大
BATCH_SIZE = 4            # 批次大小：4-bit 模型 + 8GB 显存，只能用很小的 batch
GRAD_ACCUM_STEPS = 4      # 梯度累积步数：有效 batch = 2 × 8 = 16

# --- 训练相关 ---
NUM_EPOCHS = 3            # 训练轮数：LoRA 微调通常 2-5 轮即可
LEARNING_RATE = 5e-5      # 学习率：降低以减缓过拟合，之前 2e-4 导致 loss 快速下降但 F1 停滞
WARMUP_RATIO = 0.1        # 预热比例：前 10% 步数线性增加学习率
WEIGHT_DECAY = 0.01       # 权重衰减：L2 正则化，防止过拟合
MAX_GRAD_NORM = 1.0       # 梯度裁剪：防止梯度爆炸

# --- 评估与保存 ---
EVAL_STEPS = 500          # 常规评估间隔
EARLY_EVAL_STEPS = 50     # 前期评估间隔（前 200 步每 50 步评估一次，快速发现问题）
EARLY_EVAL_UNTIL = 200    # 前多少步使用更频繁的评估
EARLY_STOPPING_PATIENCE = 5  # 连续多少次评估没提升就停止（给模型更多机会）
CHECKPOINT_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "checkpoints", "qwen-lora"
)
# SAVE_DIR 在 train() 中按时间戳动态生成，每次训练独立保存，不会覆盖历史模型

# --- 随机种子 ---
SEED = 42

# --- 类别权重（处理 69:31 类别不平衡）---
# label=0（"否"，多数类）权重 1.0，label=1（"是"，少数类）权重 3.0
# 从 2.2 提高到 3.0，进一步补偿少数类的 loss 贡献（阈值校准显示模型仍偏向"否"）
CLASS_WEIGHTS = {0: 1.0, 1: 3.0}


def set_seed(seed: int):
    """设置随机种子，确保实验可复现。

    Args:
        seed: 随机种子值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================
# 核心函数1：Tokenize 一个 batch 的 prompt
# ============================================

def tokenize_batch(prompts, tokenizer, max_length: int = 256):
    """将一批 prompt 文本转换为模型输入的 token IDs。

    工作原理：
    - tokenizer 把文字变成数字（token IDs）
    - padding="max_length" 把短句子补齐到统一长度
    - truncation=True 把超长句子截断
    - 返回 input_ids 和 attention_mask

    Args:
        prompts: prompt 文本列表。
        tokenizer: 分词器。
        max_length: 最大 token 长度。

    Returns:
        tokenizer 的输出字典，包含 input_ids 和 attention_mask。
    """
    # tokenizer 会自动处理：文字 → token IDs → padding → attention_mask
    encodings = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encodings


# ============================================
# 核心函数2：构建 Causal LM 的 labels
# ============================================

def build_causal_lm_labels(input_ids, prompts, tokenizer):
    """构建因果语言模型的训练标签。

    关键设计：
    - Causal LM 的训练目标是"预测下一个 token"
    - 但我们不希望模型学习 prompt 部分（任务说明、句子等）
    - 只希望模型学习"结论"部分（是/否）
    - 所以把 prompt 部分的 label 设为 -100（PyTorch 会忽略这些位置的 loss）

    举例：
        输入: "任务：...句子1：...句子2：...结论：是"
        标签: [-100, -100, ..., -100, 是]
                ↑ prompt 部分忽略      ↑ 只学这里

    Args:
        input_ids: token IDs 张量，shape = [batch_size, seq_len]。
        prompts: 原始 prompt 文本列表。
        tokenizer: 分词器。

    Returns:
        labels 张量，shape = [batch_size, seq_len]，prompt 部分为 -100。
    """
    labels = input_ids.clone()

    # 获取 pad_token_id，用于 mask padding 位置
    pad_token_id = tokenizer.pad_token_id

    for i, prompt in enumerate(prompts):
        # 找到"结论：" 在 prompt 中的位置
        conclusion_marker = "结论："
        marker_pos = prompt.find(conclusion_marker)

        if marker_pos == -1:
            # 如果找不到标记，整条都忽略（安全兜底）
            labels[i, :] = -100
            continue

        # 把"结论："之前的文本（含"结论："本身）单独 tokenize，得到长度
        prefix = prompt[:marker_pos + len(conclusion_marker)]
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        prefix_len = len(prefix_ids)

        # 第1步：将 prefix 部分的 label 设为 -100（不计算 prompt 的 loss）
        labels[i, :prefix_len] = -100

        # 第2步：将 padding 位置的 label 也设为 -100（不计算 padding 的 loss）
        # 这一步非常关键！否则模型 99% 的精力在学习预测 padding token
        if pad_token_id is not None:
            labels[i, input_ids[i] == pad_token_id] = -100

    return labels


# ============================================
# 核心函数3：验证集评估
# ============================================

def evaluate(model, tokenizer, val_loader, device):
    """在验证集上评估模型性能（基于 token 概率）。

    核心思路（重要，面试常考）：
    - 不再让模型"生成"文本（生成式评估），而是直接看概率（判别式评估）
    - 给模型输入 prompt（到"结论："为止），获取下一个 token 的概率分布
    - 比较"是"和"否"这两个 token 的概率，哪个高就预测哪个
    - 这样完全绕开了"模型输出什么词"的问题

    类比：
    - 生成式 = 让学生写作文，从作文里找答案（不可控）
    - 概率式 = 给学生选择题 A/B，看他选哪个（可控）

    Args:
        model: LoRA 模型。
        tokenizer: 分词器。
        val_loader: 验证集 DataLoader。
        device: 计算设备。

    Returns:
        dict: 包含 f1_macro 和 accuracy 的字典。
    """
    model.eval()
    all_preds = []
    all_labels = []

    # 预先获取"是"和"否"的 token ID
    # "是" → 模型预测这个 token 说明是 label=1（相似）
    # "否" → 模型预测这个 token 说明是 label=0（不相似）
    yes_token_id = tokenizer.encode("是", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("否", add_special_tokens=False)[0]

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="验证中", leave=False):
            labels = batch["labels"].tolist()

            for j, prompt in enumerate(batch["prompts"]):
                # 截取到"结论："为止
                conclusion_marker = "结论："
                marker_pos = prompt.find(conclusion_marker)
                if marker_pos == -1:
                    all_preds.append(0)
                    all_labels.append(labels[j])
                    continue

                # 只给模型看"结论："之前的内容（含"结论："）
                input_text = prompt[:marker_pos + len(conclusion_marker)]

                # 编码并送入模型，获取 logits（不是 generate，而是直接前向传播）
                inputs = tokenizer(input_text, return_tensors="pt")
                input_ids = inputs["input_ids"].to(device)

                # 前向传播，获取每个位置的 logits
                outputs = model(input_ids=input_ids)

                # 取最后一个 token 位置的 logits（这就是模型对"下一个 token"的预测）
                # logits shape: [1, seq_len, vocab_size]
                next_token_logits = outputs.logits[0, -1, :]  # shape: [vocab_size]

                # 只看"是"和"否"这两个 token 的分数
                score_yes = next_token_logits[yes_token_id].item()
                score_no = next_token_logits[no_token_id].item()

                # 哪个分数高，就预测哪个
                pred = 1 if score_yes > score_no else 0

                # 调试：打印前 5 条结果
                if len(all_preds) < 5:
                    print(f"  [调试] 样本{len(all_preds)+1}: "
                          f"P(是)={score_yes:.2f}, P(否)={score_no:.2f}, "
                          f"预测={pred}, 真实={labels[j]}")

                all_preds.append(pred)
                all_labels.append(labels[j])

    # 计算指标
    f1 = f1_score(all_labels, all_preds, average="macro")
    acc = accuracy_score(all_labels, all_preds)

    model.train()
    return {"f1_macro": f1, "accuracy": acc}


# ============================================
# 核心函数4：训练主循环
# ============================================

def train():
    """LoRA 微调训练主函数。

    完整流程：
    1. 设置随机种子
    2. 加载模型和数据
    3. 划分训练集/验证集
    4. 构建 DataLoader
    5. 配置优化器和学习率调度器
    6. 训练循环（前向 → 反向 → 累积 → 更新）
    7. 定期评估 + 早停 + 保存最佳模型
    """
    print("=" * 60)
    print("LoRA 微调训练 - Qwen-1.5-1.8B")
    print("=" * 60)

    # --- 步骤1：设置随机种子 ---
    set_seed(SEED)
    print(f"随机种子: {SEED}")

    # --- 步骤2：加载模型 ---
    print("\n[1/5] 加载模型...")
    model, tokenizer = load_model_with_lora()
    device = next(model.parameters()).device
    print(f"模型设备: {device}")

    # 确保 tokenizer 有 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 步骤3：加载并划分数据 ---
    print("\n[2/5] 加载数据...")
    all_data = load_train_data()

    # 打乱数据
    random.shuffle(all_data)

    # 划分训练集和验证集
    val_size = int(len(all_data) * VAL_RATIO)
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]
    print(f"训练集: {len(train_data)} 条, batch验证集: {len(val_data)} 条")

    # --- 步骤4：构建 Dataset 和 DataLoader ---
    print("\n[3/5] 构建 DataLoader...")
    train_dataset = AFQMCLLMDataset(train_data, mode="train")
    val_dataset = AFQMCLLMDataset(val_data, mode="train")  # 验证集也用 train 模式（需要标签）

    # collate_fn：把一个 batch 的样本整理成统一格式
    def train_collate_fn(batch):
        prompts = [item["prompt"] for item in batch]
        labels_list = [item["label"] for item in batch]

        # tokenize
        encodings = tokenize_batch(prompts, tokenizer, MAX_LENGTH)
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # 构建 causal LM labels（只对"结论"部分计算 loss）
        lm_labels = build_causal_lm_labels(input_ids, prompts, tokenizer)

        # 计算每个样本的类别权重（用于加权 loss）
        sample_weights = torch.tensor(
            [CLASS_WEIGHTS[label] for label in labels_list],
            dtype=torch.float32,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": lm_labels,
            "sample_weights": sample_weights,
        }

    def val_collate_fn(batch):
        labels_list = [item["label"] for item in batch]

        # 重新构建不含答案的 prompt（结论留空，让模型自己预测）
        prompts = []
        for item in batch:
            prompt = build_prompt(
                sentence1=item["text1"],
                sentence2=item["text2"],
                conclusion="",  # 结论留空，让模型自己生成
            )
            prompts.append(prompt)

        return {
            "prompts": prompts,
            "labels": torch.tensor(labels_list, dtype=torch.long),
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_collate_fn,
        num_workers=0,  # Windows 建议设为 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 验证时逐条生成，batch=1
        shuffle=False,
        collate_fn=val_collate_fn,
        num_workers=0,
    )

    # --- 步骤5：配置优化器和学习率调度器 ---
    print("\n[4/5] 配置优化器...")

    # TODO: 第3个填空 - 创建 AdamW 优化器
    # 提示：
    #   1. 只优化 requires_grad=True 的参数（即 LoRA 参数）
    #   2. 设置 lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # 计算总训练步数和 warmup 步数
    total_steps = (len(train_loader) // GRAD_ACCUM_STEPS) * NUM_EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)

    # 学习率调度器：先 warmup 再线性衰减
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"总训练步数: {total_steps}, Warmup 步数: {warmup_steps}")
    print(f"有效 batch size: {BATCH_SIZE} × {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS}")

    # ============================================
    # 预检流程：训练前快速验证，避免浪费时间
    # ============================================
    print("\n" + "=" * 60)
    print("[预检] 训练前快速验证")
    print("=" * 60)

    preflight_passed = True

    # 预检1：打印一条训练样本的 prompt，确认格式正确
    print("\n[预检1] 训练样本 prompt 格式：")
    sample_train = train_dataset[0]
    print(f"  {sample_train['prompt'][:200]}...")
    if "结论：" not in sample_train["prompt"]:
        print("  ❌ 训练样本缺少'结论：'标记！")
        preflight_passed = False
    else:
        print("  ✅ 格式正确")

    # 预检2：验证训练和验证的 prompt prefix 一致（"结论："之前的部分）
    print("\n[预检2] 训练/验证 prompt 一致性检查：")
    val_sample = val_dataset[0]
    val_prompt = build_prompt(
        sentence1=val_sample["text1"],
        sentence2=val_sample["text2"],
        conclusion="",
    )
    print(f"  {val_prompt}")
    # 检查模板结构：用相同的句子构建训练和验证 prompt，确认"结论："之前完全一致
    check_train = build_prompt(sentence1="A", sentence2="B", conclusion="是")
    check_val = build_prompt(sentence1="A", sentence2="B", conclusion="")
    check_train_prefix = check_train[:check_train.find("结论：") + len("结论：")]
    if check_train_prefix == check_val:
        print("  ✅ 训练/验证 prompt 结构一致")
    else:
        print("  ❌ 训练/验证 prompt 结构不一致！")
        print(f"    训练 prefix: {repr(check_train_prefix)}")
        print(f"    验证 prompt: {repr(check_val)}")
        preflight_passed = False

    # 预检3：验证"是"/"否"为单 token + 概率评估逻辑测试（3 条）
    print("\n[预检3] Token 验证 + 概率评估测试：")
    yes_ids = tokenizer.encode("是", add_special_tokens=False)
    no_ids = tokenizer.encode("否", add_special_tokens=False)
    print(f"  '是' token IDs: {yes_ids} (数量={len(yes_ids)})")
    print(f"  '否' token IDs: {no_ids} (数量={len(no_ids)})")
    if len(yes_ids) != 1 or len(no_ids) != 1:
        print("  ❌ '是'或'否'不是单 token！标签设计失效！")
        preflight_passed = False
    else:
        print("  ✅ '是'和'否'均为单 token")
    yes_token_id = yes_ids[0]
    no_token_id = no_ids[0]
    print(f"  Token ID: '是'={yes_token_id}, '否'={no_token_id}")

    model.eval()
    with torch.no_grad():
        for i in range(min(3, len(val_data))):
            item = val_data[i]
            test_prompt = build_prompt(
                sentence1=str(item.get("text1", "")),
                sentence2=str(item.get("text2", "")),
                conclusion="",
            )
            marker_pos = test_prompt.find("结论：")
            input_text = test_prompt[:marker_pos + len("结论：")]
            inputs = tokenizer(input_text, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            outputs = model(input_ids=input_ids)
            next_logits = outputs.logits[0, -1, :]
            s_score = next_logits[yes_token_id].item()
            n_score = next_logits[no_token_id].item()
            pred = 1 if s_score > n_score else 0
            print(f"  样本{i+1}: P(是)={s_score:.2f}, P(否)={n_score:.2f}, "
                  f"预测={pred}, 真实={item.get('label', '?')}")

    # 预检4：验证 labels 中非 -100 位置数量（Step 0b）
    # 确保每个样本只有 1 个有效 token 参与 loss 计算（"是"或"否"）
    print("\n[预检4] Labels 有效位置数量验证：")
    test_batch_for_labels = next(iter(train_loader))
    test_lm_labels = test_batch_for_labels["labels"]
    labels_ok = True
    for i in range(test_lm_labels.size(0)):
        valid_count = (test_lm_labels[i] != -100).sum().item()
        if valid_count != 1:
            print(f"  ❌ 样本{i+1}: 非-100位置数量={valid_count}，期望=1")
            labels_ok = False
            preflight_passed = False
    if labels_ok:
        print(f"  ✅ 所有样本均只有 1 个有效 token 参与 loss（batch_size={test_lm_labels.size(0)}）")

    # 预检5：跑 1 个 batch 的前向+反向，确认加权 loss 和梯度正常
    print("\n[预检5] 单 batch 前向+反向测试（加权 loss）：")
    model.train()
    test_input_ids = test_batch_for_labels["input_ids"].to(device)
    test_attention_mask = test_batch_for_labels["attention_mask"].to(device)
    test_labels = test_batch_for_labels["labels"].to(device)
    test_weights = test_batch_for_labels["sample_weights"].to(device)
    with autocast(dtype=torch.float16):
        test_outputs = model(
            input_ids=test_input_ids,
            attention_mask=test_attention_mask,
        )
        # 手动计算加权 loss（与训练循环逻辑一致）
        t_logits = test_outputs.logits
        t_shift_logits = t_logits[..., :-1, :].contiguous()
        t_shift_labels = test_labels[..., 1:].contiguous()
        t_loss_fct = CrossEntropyLoss(reduction="none")
        t_per_token = t_loss_fct(
            t_shift_logits.view(-1, t_shift_logits.size(-1)),
            t_shift_labels.view(-1),
        ).view(t_shift_labels.size())
        t_valid_mask = (t_shift_labels != -100).float()
        t_per_sample = (t_per_token * t_valid_mask).sum(dim=1) / t_valid_mask.sum(dim=1).clamp(min=1)
        test_loss = (t_per_sample * test_weights).mean()
    test_loss.backward()
    # 检查是否有梯度
    has_grad = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    optimizer.zero_grad()
    if has_grad:
        print(f"  ✅ 加权Loss={test_loss.item():.4f}, 梯度正常")
        print(f"     样本权重: {test_weights.tolist()}")
    else:
        print(f"  ❌ 加权Loss={test_loss.item():.4f}, 但没有梯度！")
        preflight_passed = False

    # 预检总结
    print("\n" + "-" * 40)
    if preflight_passed:
        print("[预检通过] ✅ 所有检查通过，开始正式训练")
    else:
        print("[预检失败] ❌ 发现问题，请检查后重新运行")
        return
    print("=" * 60)

    # --- 步骤6：训练循环 ---
    print("\n[5/5] 开始训练...")
    print("=" * 60)

    # AMP 混合精度
    scaler = GradScaler()

    # 早停和最佳模型跟踪
    best_f1 = 0.0
    best_acc = 0.0
    patience_counter = 0
    global_step = 0

    # 按时间戳创建本次训练的保存目录，避免覆盖历史模型
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(CHECKPOINT_ROOT, f"run_{run_timestamp}")
    save_dir_f1 = os.path.join(save_dir, "best_f1")
    save_dir_acc = os.path.join(save_dir, "best_acc")
    os.makedirs(save_dir_f1, exist_ok=True)
    os.makedirs(save_dir_acc, exist_ok=True)
    print(f"本次训练保存目录: {save_dir}")

    model.train()
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_steps = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
            leave=True,
        )

        for step, batch in enumerate(progress_bar):
            # 将数据移到 GPU
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            sample_weights = batch["sample_weights"].to(device)  # [batch_size]

            # --- 前向传播（AMP 混合精度）---
            # 不传 labels 给模型，手动计算加权 loss
            with autocast(dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                # 手动计算加权 Causal LM loss
                # HuggingFace 内部做法：logits 左移、labels 右移，实现"预测下一个 token"
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()  # [batch, seq_len-1, vocab]
                shift_labels = labels[..., 1:].contiguous()       # [batch, seq_len-1]

                # 逐 token 计算 cross-entropy（不求平均，保留每个位置的 loss）
                loss_fct = CrossEntropyLoss(reduction="none")
                # 展平后计算，再恢复形状
                per_token_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                ).view(shift_labels.size())  # [batch, seq_len-1]

                # 对每个样本：只有非 -100 位置有 loss（其余位置 CE 输出 0）
                # 求每个样本的平均 loss（除以该样本的有效 token 数）
                valid_mask = (shift_labels != -100).float()        # [batch, seq_len-1]
                per_sample_loss = (per_token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)  # [batch]

                # 用类别权重加权每个样本的 loss，再求 batch 平均
                weighted_loss = (per_sample_loss * sample_weights).mean()

                # loss 除以累积步数，保证梯度量级一致
                loss = weighted_loss / GRAD_ACCUM_STEPS

            # --- 反向传播 ---
            scaler.scale(loss).backward()

            epoch_loss += loss.item() * GRAD_ACCUM_STEPS
            epoch_steps += 1

            # --- 梯度累积：攒够了再更新 ---
            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), MAX_GRAD_NORM
                )

                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

                global_step += 1

                # 更新进度条
                avg_loss = epoch_loss / epoch_steps
                current_lr = scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step": global_step,
                })

                # --- 定期评估（前期更频繁，快速发现问题）---
                current_eval_steps = EARLY_EVAL_STEPS if global_step <= EARLY_EVAL_UNTIL else EVAL_STEPS
                if global_step % current_eval_steps == 0:
                    print(f"\n--- 第 {global_step} 步评估 ---")
                    metrics = evaluate(model, tokenizer, val_loader, device)
                    f1 = metrics["f1_macro"]
                    acc = metrics["accuracy"]
                    print(f"F1-macro: {f1:.4f}, Accuracy: {acc:.4f}")

                    # 检查是否是最佳 F1 模型
                    if f1 > best_f1:
                        best_f1 = f1
                        patience_counter = 0
                        model.save_pretrained(save_dir_f1)
                        tokenizer.save_pretrained(save_dir_f1)
                        print(f"保存最佳F1模型! F1: {best_f1:.4f}, Acc: {acc:.4f}")
                    else:
                        patience_counter += 1
                        print(f"F1未提升 ({patience_counter}/{EARLY_STOPPING_PATIENCE})")

                    # 检查是否是最佳 Accuracy 模型（独立保存，不影响早停）
                    if acc > best_acc:
                        best_acc = acc
                        model.save_pretrained(save_dir_acc)
                        tokenizer.save_pretrained(save_dir_acc)
                        print(f"保存最佳Acc模型! Acc: {best_acc:.4f}, F1: {f1:.4f}")

                    # 早停检查
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        print("触发早停，训练结束。")
                        break

                    model.train()

        # 早停跳出外层循环
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            break

        # 每个 epoch 结束也评估一次
        print(f"\n=== Epoch {epoch + 1} 结束 ===")
        metrics = evaluate(model, tokenizer, val_loader, device)
        f1 = metrics["f1_macro"]
        acc = metrics["accuracy"]
        print(f"F1-macro: {f1:.4f}, Accuracy: {acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
            model.save_pretrained(save_dir_f1)
            tokenizer.save_pretrained(save_dir_f1)
            print(f"保存最佳F1模型! F1: {best_f1:.4f}, Acc: {acc:.4f}")
        else:
            patience_counter += 1

        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(save_dir_acc)
            tokenizer.save_pretrained(save_dir_acc)
            print(f"保存最佳Acc模型! Acc: {best_acc:.4f}, F1: {f1:.4f}")

    # --- 训练总结 ---
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    print(f"总耗时: {minutes}分{seconds}秒")
    print(f"最佳 F1-macro: {best_f1:.4f} → {save_dir_f1}")
    print(f"最佳 Accuracy: {best_acc:.4f} → {save_dir_acc}")
    print(f"模型保存位置: {save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    train()