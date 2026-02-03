# AFQMC 阶段二复盘：MacBERT Baseline 模型训练

> **作者视角**：阿里 P7 算法技术专家
> **项目阶段**：阶段二 - MacBERT 基线模型
> **完成时间**：2026-01-27
> **硬件环境**：RTX 4060 Laptop GPU (8GB VRAM)

---

## 目录

1. [项目代码结构与职责说明](#1-项目代码结构与职责说明)
2. [训练全流程深度梳理](#2-训练全流程深度梳理pipeline)
3. [当前模型性能诊断](#3-当前模型性能诊断)
4. [核心技术难点与知识点沉淀](#4-核心技术难点与知识点沉淀)
5. [面试官视角的 Q&A](#5-面试官视角的-qa)

---

## 1. 项目代码结构与职责说明

### 1.1 目录结构概览

```
AFQMC/
├── config/
│   └── config.py              # 全局配置中心
├── dataset/
│   ├── train.jsonl            # 原始训练数据
│   └── test.jsonl             # 原始测试数据
├── checkpoints/
│   ├── chinese-macbert-base/  # 预训练模型权重
│   └── macbert/              # 训练过程中的模型检查点
├── src/
│   ├── data_loader.py        # 数据加载与预处理
│   ├── model_macbert.py      # 模型定义与初始化
│   ├── train.py              # 训练主脚本
│   └── utils.py              # 工具函数集合
├── notebooks/
│   └── eda_analysis.ipynb    # 探索性数据分析
└── results/
    └── predictions/          # 预测结果输出
```

### 1.2 核心文件职责说明

#### **config/config.py** - 配置管理中心
**核心职责**：
- 集中管理所有超参数（学习率、batch size、epoch 数等）
- 定义路径配置（数据路径、模型保存路径）
- 硬件优化配置（AMP、梯度累积）

**设计亮点**：
- 基于 EDA 发现优化参数（如 `MAX_LENGTH=64` 而非盲目设置 128）
- 针对 8GB 显存的显存优化配置（`USE_AMP=True`, `GRADIENT_ACCUMULATION_STEPS=2`）
- 类别不平衡处理开关（`USE_CLASS_WEIGHT=True`）

---

#### **src/data_loader.py** - 数据管道
**核心职责**：
- `load_train_data()`: 从 JSONL 文件读取原始数据，返回 pandas DataFrame
- `split_train_val()`: 使用分层采样切分训练集和验证集（保持类别比例）
- 数据清洗：处理缺失值、去除异常样本

**关键逻辑**：
```
原始 JSONL → pandas DataFrame → 分层采样 → (train_texts, train_labels, val_texts, val_labels)
```

**为什么不在这里做 Tokenization？**
遵循"职责分离"原则：
- `data_loader.py` 只负责原始数据的加载和切分
- Tokenization 放在 `Dataset` 类中（train.py），因为需要结合 DataLoader 的批处理逻辑

---

#### **src/model_macbert.py** - 模型定义
**核心职责**：
- `create_tokenizer()`: 加载 MacBERT 的 Tokenizer
- `create_model()`: 构建完整的分类模型
  - 加载预训练的 MacBERT-base
  - 添加分类头（通常是一个线性层：768 → 2）

**模型架构**：
```
Input Text
    ↓
Tokenizer (文本 → Token IDs)
    ↓
MacBERT Encoder (12层Transformer)
    ↓
[CLS] Token 的输出向量 (768维)
    ↓
Dropout (防止过拟合)
    ↓
Linear Layer (768 → 2)
    ↓
Logits (未归一化的分数)
```

**每一步的详细数据变换：**

**Step 1: Input Text (输入文本)**
```python
# 原始输入
text1 = "如何更换花呗绑定银行卡"  # 句子A
text2 = "花呗更改绑定银行卡"      # 句子B

# 此时状态
类型: 字符串（str）
格式: 纯文本，计算机还无法理解
维度: 无（还不是数字）
```

---

**Step 2: Tokenizer (文本 → Token IDs)**

**输入：** 两个字符串

**处理过程：**
1. 分词：把文本切分成tokens
2. 添加特殊符号：`[CLS]` 在开头，`[SEP]` 分隔两个句子
3. 转换为ID：每个token转为词表中的数字编号
4. Padding：补齐到固定长度（MAX_LENGTH=64）

**输出：**
```python
# 分词后的token序列
tokens = ['[CLS]', '如', '何', '更', '换', '花', '呗', '绑', '定',
          '银', '行', '卡', '[SEP]', '花', '呗', '更', '改',
          '绑', '定', '银', '行', '卡', '[SEP]', '[PAD]', '[PAD]', ...]
                                                   ↑ 补齐到64个

# 转为整数ID序列
input_ids = [101, 1963, 862, 3291, 2938, 5709, 1440, 5011, 1348,
             7213, 6121, 1305, 102, 5709, 1440, 3291, 3121,
             5011, 1348, 7213, 6121, 1305, 102, 0, 0, ..., 0]
             ↑                    ↑                   ↑
           [CLS]               [SEP]              [PAD]

# 附加输出
attention_mask = [1, 1, 1, ..., 1, 0, 0, ..., 0]  # 1=真实token, 0=padding
                  ↑22个1         ↑42个0
token_type_ids = [0, 0, ..., 0, 1, 1, ..., 1, 0]  # 0=句子1, 1=句子2

# 形状变化
字符串 → [batch_size, 64] 整数数组
"如何更换..." → [[101, 1963, 862, ...]]
```

---

**Step 3: MacBERT Encoder (12层Transformer)**

**输入：**
```python
input_ids: [batch_size, 64]       # Token IDs
attention_mask: [batch_size, 64]  # 1=真实token, 0=padding
token_type_ids: [batch_size, 64]  # 0=句子1, 1=句子2
```

**处理过程：**

**3.1 Embedding层（ID → 向量）**
```python
# 每个ID查表得到768维向量
ID=101 ([CLS]) → [ 0.021, -0.134,  0.453, ...,  0.281]  (768个数)
ID=1963 ('如')  → [ 0.712,  0.033, -0.221, ...,  0.609]
ID=0 ([PAD])   → [ 0.000,  0.000,  0.000, ...,  0.000]

# 形状: [batch_size, 64] → [batch_size, 64, 768]
```

**3.2 12层Transformer处理**

每层包含：
- **Self-Attention**：让每个token看到其他所有token
- **Feed-Forward**：非线性变换

```python
# 第0层（Embedding层输出）
hidden_states[0]: [batch_size, 64, 768]
  [CLS]位置: [ 0.021, -0.134, ...,  0.281]

# 第1层（第1个Transformer）
hidden_states[1]: [batch_size, 64, 768]
  [CLS]位置: [ 0.143, -0.089, ...,  0.412]  ← 向量改变了！

# ... 经过12层处理 ...

# 第12层（最后一层）
hidden_states[12]: [batch_size, 64, 768]
  [CLS]位置: [ 1.234, -0.567, ...,  2.891]  ← 包含了丰富的语义信息
```

**输出：**
```python
encoder_output: [batch_size, 64, 768]

# 具体数据：
Position 0 ([CLS]):  [ 1.234, -0.567,  0.892, ...,  2.891]  ← 最重要！
Position 1 ('如'):   [ 0.423,  1.098, -0.334, ...,  1.567]
Position 2 ('何'):   [-0.678,  0.234,  1.456, ..., -0.123]
...
Position 22 ([SEP]): [ 0.891, -0.234,  0.567, ...,  0.345]

# 形状保持
[64] 整数 → [64, 768] 浮点向量
每个token从一个ID变成了一个768维的语义向量
```

---

**Step 4: [CLS] Token 提取 (768维向量)**

**输入：**
```python
encoder_output: [batch_size, 64, 768]  # 所有token的输出
```

**处理：** 只取第0个位置（[CLS] token的向量）

```python
cls_output = encoder_output[:, 0, :]  # shape: [batch_size, 768]

# 这768个数字代表：
# - 句子1的语义
# - 句子2的语义
# - 两个句子之间的关系（相似/不相似）

# 具体值
cls_output = [ 1.234, -0.567,  0.892, ...,  2.891]
```

**形状变化：**
```
[batch_size, 64, 768] → [batch_size, 768]
从64个向量中，只保留第0个（[CLS]）向量
```

**为什么只要[CLS]？**
- Transformer的自注意力机制让[CLS]能"看到"所有其他token
- 经过12层处理后，[CLS]已经包含了整个句子对的综合信息

---

**Step 5: Dropout (防止过拟合)**

**输入：**
```python
cls_output: [batch_size, 768]
```

**训练时（model.train()）：**
```python
# 随机将10%的值变为0（dropout_prob=0.1）
输入:  [ 1.234, -0.567,  0.892, ...,  2.891]
      ↓ 随机mask
输出:  [ 1.371,  0.000,  0.991, ...,  3.212]
              ↑ 被drop了，其他值放大补偿
```

**推理时（model.eval()）：**
```python
# 不做任何改变，直接通过
输入:  [ 1.234, -0.567,  0.892, ...,  2.891]
      ↓ 直接通过
输出:  [ 1.234, -0.567,  0.892, ...,  2.891]
```

**形状保持：** `[batch_size, 768] → [batch_size, 768]`

---

**Step 6: Linear Layer (768 → 2)**

**输入：**
```python
dropout_output: [batch_size, 768]
```

**处理：** 线性变换
```python
# 权重矩阵（训练过程中学习得到）
W: [768, 2]
b: [2]

# 矩阵乘法
logits = dropout_output @ W + b

# 具体计算（简化）：
logits[0] = 1.234*w[0,0] + (-0.567)*w[1,0] + ... + b[0]
logits[1] = 1.234*w[0,1] + (-0.567)*w[1,1] + ... + b[1]
```

**输出：**
```python
logits = [ 2.347, -1.892]  # shape: [batch_size, 2]
           ↑        ↑
        类别0     类别1
       (不相似)   (相似)
```

**形状变化：**
```
[batch_size, 768] → [batch_size, 2]
768个特征 → 2个类别的分数
```

---

**Step 7: Logits (未归一化的分数)**

**输出：**
```python
logits = [ 2.347, -1.892]
```

**含义：**
- `logits[0] = 2.347`：模型认为"不相似"的原始分数
- `logits[1] = -1.892`：模型认为"相似"的原始分数
- 这些是**未归一化**的分数（可以是任意实数）

**如何使用：**

**方法1：直接比大小（训练时）**
```python
loss = CrossEntropyLoss()(logits, labels)  # Loss函数内部会做softmax
```

**方法2：转为概率（推理时）**
```python
probabilities = softmax(logits)
              = [ 0.9876, 0.0124]

→ 不相似概率：98.76%
→ 相似概率：1.24%

predicted_class = argmax(logits) = 0  # 预测：不相似
```

---

**完整数据流总结：**

| 步骤 | 输入形状 | 输出形状 | 关键变化 |
|-----|---------|---------|---------|
| **Input Text** | 字符串 | - | 原始文本对 |
| **Tokenizer** | 字符串 | `[batch, 64]` | 文本→数字ID |
| **Embeddings** | `[batch, 64]` | `[batch, 64, 768]` | ID→768维向量 |
| **12层Transformer** | `[batch, 64, 768]` | `[batch, 64, 768]` | 提取语义特征 |
| **[CLS]提取** | `[batch, 64, 768]` | `[batch, 768]` | 取第0个位置 |
| **Dropout** | `[batch, 768]` | `[batch, 768]` | 随机置0（训练时） |
| **Linear** | `[batch, 768]` | `[batch, 2]` | 768→2 |
| **Logits** | `[batch, 2]` | `[batch, 2]` | 未归一化分数 |

**核心理解：**
1. **Tokenizer**：文本 → 数字（让计算机能处理）
2. **MacBERT Encoder**：数字 → 语义向量（理解句子含义）
3. **[CLS]**：提取综合信息（句子对的整体表示）
4. **Linear**：语义 → 分类分数（判断相似/不相似）

每一步都在逐步"压缩"信息：**文本 → 64个向量 → 1个向量 → 2个分数 → 1个预测**

---

**为什么选择 MacBERT？**
- **原生 BERT** 使用 `[MASK]` 进行预训练，但这个 token 在下游任务中不常见，导致 pretrain-finetune gap
- **MacBERT** 使用**相似词替换**而非 `[MASK]`，更贴近真实文本，中文任务上表现更好

---

#### **src/train.py** - 训练主脚本
**核心职责**：
1. **自定义 Dataset 类** (`AFQMCDataset`)
   - 实现 `__getitem__`: 对单个样本进行 Tokenization
   - 返回格式：`{'input_ids': Tensor, 'attention_mask': Tensor, 'labels': Tensor}`

2. **训练循环** (`train_epoch`)
   - 前向传播 + AMP 混合精度
   - 梯度累积（每 2 个 batch 更新一次参数）
   - 梯度裁剪（防止梯度爆炸）
   - 学习率调度（Warmup + Linear Decay）

3. **评估函数** (`evaluate`)
   - 在验证集上计算 Loss 和 Metrics
   - 不计算梯度（`torch.no_grad()`）
   - 收集所有预测结果用于混淆矩阵分析

4. **主训练流程** (`main`)
   - 加载数据 → 创建 DataLoader → 初始化模型 → 训练循环 → Early Stopping → 保存最佳模型

**显存优化的核心代码逻辑**：
```python
# AMP 混合精度
with torch.cuda.amp.autocast():
    outputs = model(input_ids, attention_mask, labels)
    loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS  # 梯度累积：损失除以累积步数

# 梯度累积：只在累积满后才更新参数
if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
    scaler.step(optimizer)  # 更新参数
    scaler.update()         # 更新 scaler 的 scale factor
    optimizer.zero_grad()   # 清空梯度
```

---

#### **src/utils.py** - 工具函数库
**核心职责**：
- `compute_metrics()`: 计算 Accuracy、Precision、Recall、F1-macro、混淆矩阵
- `compute_class_weights()`: 基于训练集标签分布自动计算类别权重
- `save_checkpoint()` / `load_checkpoint()`: 模型保存与加载
- `set_seed()`: 设置随机种子保证实验可复现

**为什么 F1-macro 是主要指标？**
- **Accuracy** 在类别不平衡时会被多数类主导（本项目 69:31 不平衡）
- **F1-macro** 对每个类别一视同仁，更能反映模型在少数类上的表现

---

## 2. 训练全流程深度梳理（Pipeline）

### 2.1 完整数据流转图

```
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 0: 原始数据                                                  │
├─────────────────────────────────────────────────────────────────┤
│ train.jsonl: {"sentence1": "蚂蚁借呗等额还款...", "label": "1"} │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 1: 数据加载 (data_loader.py)                                │
├─────────────────────────────────────────────────────────────────┤
│ load_train_data() → pandas DataFrame                            │
│ split_train_val() → 分层采样 (90% train, 10% val)                │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 2: Dataset & DataLoader (train.py)                          │
├─────────────────────────────────────────────────────────────────┤
│ AFQMCDataset.__getitem__(idx):                                  │
│   1. 获取 text1, text2, label                                    │
│   2. Tokenizer编码:                                              │
│      tokenizer(text1, text2, max_length=64, padding=True)       │
│   3. 返回: {input_ids, attention_mask, labels}                   │
│                                                                  │
│ DataLoader:                                                      │
│   - batch_size=32                                                │
│   - shuffle=True (训练集打乱)                                     │
│   - num_workers=0 (Windows兼容)                                  │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 3: 模型前向传播 (model_macbert.py)                          │
├─────────────────────────────────────────────────────────────────┤
│ Input:                                                           │
│   input_ids:      [batch_size, 64]  # Token IDs                │
│   attention_mask: [batch_size, 64]  # 1=真实token, 0=padding    │
│   labels:         [batch_size]      # 0 或 1                    │
│                                                                  │
│ MacBERT Encoder (12层Transformer):                              │
│   - Self-Attention: 捕捉词与词之间的关系                          │
│   - Feed-Forward: 特征变换                                        │
│   - Layer Norm + Residual: 稳定训练                              │
│                                                                  │
│ 输出:                                                             │
│   last_hidden_state: [batch_size, 64, 768]                      │
│   → 取 [CLS] token (第0个): [batch_size, 768]                   │
│                                                                  │
│ Classifier Head:                                                 │
│   Linear(768 → 2): [batch_size, 2]                             │
│   → Logits: 未归一化的分数 (可能是负数或>1)                        │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 4: 损失计算与反向传播                                         │
├─────────────────────────────────────────────────────────────────┤
│ Loss Function: CrossEntropyLoss (with class weights)            │
│   - 自动进行 softmax 归一化                                        │
│   - 计算负对数似然: -log(P(正确类别))                              │
│   - 类别权重: [0.72, 1.62] 提升少数类的损失权重                    │
│                                                                  │
│ 反向传播:                                                         │
│   loss.backward() → 计算梯度                                      │
│   梯度累积: 每2个batch累积一次梯度                                  │
│   梯度裁剪: clip_grad_norm_(max_norm=1.0)                         │
│   优化器更新: optimizer.step()                                    │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 5: 评估与预测                                                 │
├─────────────────────────────────────────────────────────────────┤
│ 评估模式: model.eval() + torch.no_grad()                         │
│   - 关闭 Dropout                                                 │
│   - 不计算梯度 (节省显存)                                          │
│                                                                  │
│ 预测:                                                             │
│   logits = model(input_ids, attention_mask)                     │
│   preds = torch.argmax(logits, dim=-1)  # 取最大值的索引          │
│                                                                  │
│ 指标计算:                                                         │
│   - Accuracy: 正确预测数 / 总样本数                                │
│   - F1-macro: (F1_class0 + F1_class1) / 2                       │
│   - Confusion Matrix: 分析错误类型                                │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 关键组件深度解析

#### **2.2.1 Tokenizer：从文本到数字**

**BERT 的输入格式**：
```
原始输入:
  sentence1 = "蚂蚁借呗等额还款"
  sentence2 = "借呗能否分期"

Tokenizer 处理后:
  [CLS] 蚂 蚁 借 呗 等 额 还 款 [SEP] 借 呗 能 否 分 期 [SEP] [PAD] [PAD] ...

Token IDs (数字化):
  [101, 5802, 5812, 100, 1962, 5023, 7722, 6820, 3563, 102, 100, 1962, 5543, 1555, 1298, 3309, 102, 0, 0, ...]

Attention Mask (标记有效token):
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...]
```

**特殊 Token 的含义**：
- `[CLS]` (ID=101): 分类任务的句子表示，Transformer 会将句子的语义信息聚合到这个 token
- `[SEP]` (ID=102): 句子分隔符，告诉模型两个句子的边界
- `[PAD]` (ID=0): 填充 token，用于对齐 batch 内的序列长度
- `[UNK]` (ID=100): 未知词，OOV（Out-of-Vocabulary）词汇的替代

**为什么需要 attention_mask？**
Transformer 的 Self-Attention 会计算所有 token 之间的相关性，但 `[PAD]` 是人为填充的无意义 token，不应该参与注意力计算。`attention_mask=0` 的位置会被设置为 `-inf`，经过 softmax 后权重接近 0。

---

#### **2.2.2 MacBERT Encoder：语义特征提取**

**Transformer 的核心机制**：

1. **Self-Attention（自注意力）**
   ```
   Query、Key、Value 都来自输入本身
   Attention(Q, K, V) = softmax(QK^T / √d_k) * V

   作用: 让每个词关注到与它相关的其他词
   例子: "借呗能否分期" 中，"分期"会关注到"借呗"（主语-谓语关系）
   ```

2. **Multi-Head Attention（多头注意力）**
   ```
   12个注意力头并行计算，捕捉不同类型的语义关系:
   - Head 1: 可能关注语法结构
   - Head 2: 可能关注语义相似度
   - Head 3: 可能关注实体关系
   ...
   ```

3. **Feed-Forward Network（前馈网络）**
   ```
   两层全连接: 768 → 3072 → 768
   GELU激活: 引入非线性，增强表达能力
   ```

4. **Layer Normalization + Residual（层归一化 + 残差连接）**
   ```
   output = LayerNorm(x + Sublayer(x))

   作用:
   - 残差连接: 缓解梯度消失，使得深层网络可训练
   - 层归一化: 稳定训练过程，加速收敛
   ```

**MacBERT vs 原生 BERT**：
| 特性 | 原生 BERT | MacBERT |
|-----|----------|---------|
| 预训练掩码方式 | `[MASK]` token | **相似词替换** |
| N-gram 掩码 | 不支持 | **支持**（如掩码整个词组） |
| 中文任务性能 | 良好 | **更优** |
| Pretrain-Finetune Gap | 较大 | **较小** |

**为什么 MacBERT 更适合这个任务？**
- AFQMC 是句子相似度任务，需要捕捉细粒度的语义差异
- MacBERT 的相似词替换使得预训练和微调的输入分布更一致
- 在多个中文 NLP 基准测试中，MacBERT 比 BERT 高 0.5-1.5 个百分点

---

#### **2.2.3 Classifier Head：从表示到分类**

```python
# 模型输出
outputs = model(input_ids, attention_mask)
last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, 768]

# 取 [CLS] token 的输出（第0个位置）
cls_output = last_hidden_state[:, 0, :]  # [batch_size, 768]

# 分类层
dropout_output = nn.Dropout(0.1)(cls_output)
logits = nn.Linear(768, 2)(dropout_output)  # [batch_size, 2]
```

**Logits vs Probabilities**：

| 概念 | 取值范围 | 含义 | 使用场景 |
|-----|---------|------|---------|
| **Logits** | (-∞, +∞) | 未归一化的分数，表示模型对每个类别的"置信度" | 训练时（与 CrossEntropyLoss 配合） |
| **Probabilities** | [0, 1] | 经过 softmax 归一化后的概率分布，和为1 | 推理时（输出预测概率） |

**例子**：
```python
logits = torch.tensor([[2.3, -1.5]])  # 模型输出
probs = torch.softmax(logits, dim=-1)  # [0.98, 0.02]
pred = torch.argmax(logits, dim=-1)    # 0（预测为不相似）
```

**为什么训练时用 logits 而不直接用 probabilities？**
- `CrossEntropyLoss` 内部会自动进行 softmax，再计算负对数似然
- 直接用 logits 可以利用 PyTorch 的数值稳定性优化（log-sum-exp trick）

---

#### **2.2.4 Loss Function：优化目标**

**CrossEntropyLoss with Class Weights**：

```python
# 类别权重计算（基于训练集标签分布）
class_weights = compute_class_weight('balanced', classes=[0, 1], y=train_labels)
# 结果: [0.72, 1.62]（少数类权重更高）

criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

# 损失计算
loss = criterion(logits, labels)
```

**数学原理**：
```
标准交叉熵:
L = -log(P(y_true))

加权交叉熵:
L = -w[y_true] * log(P(y_true))

其中 w[0]=0.72, w[1]=1.62
```

**为什么要加权？**
- 数据集中 Class 0 占 69%，Class 1 占 31%
- 不加权时，模型倾向于全预测为 Class 0（Accuracy 可达 69%但没有意义）
- 加权后，模型预测错 Class 1 的惩罚更大，强迫模型学习少数类特征

---

### 2.3 显存优化：为什么需要 AMP 和梯度累积？

#### **硬件约束分析**：RTX 4060 8GB

**MacBERT-base 的显存占用估算**：
```
模型参数:
  - 参数量: 102M（约400MB，FP32）
  - 梯度: 400MB（与参数量相同）
  - 优化器状态（AdamW）: 800MB（2倍参数量）

单个 batch 的激活值:
  - batch_size=32, seq_len=64, hidden_size=768
  - 12层 Transformer，每层存储中间激活
  - 估算: ~1.5GB（取决于具体实现）

总显存需求（无优化）:
  400MB + 400MB + 800MB + 1500MB ≈ 3.1GB（单batch）

如果 batch_size=64:
  ≈ 5.5GB（接近 8GB 极限，容易 OOM）
```

#### **优化策略 1：AMP（自动混合精度训练）**

**原理**：
```
FP32 (单精度浮点数):
  - 32位存储
  - 精度高，但占用显存多

FP16 (半精度浮点数):
  - 16位存储
  - 精度略低，但显存占用减半

AMP 策略:
  - 前向传播: 大部分用 FP16（节省显存）
  - 梯度计算: FP16
  - 参数更新: FP32（保证精度）
  - 自动处理溢出问题（GradScaler）
```

**显存节省**：
```
激活值显存: 1.5GB → 0.75GB（减少50%）
总显存: 3.1GB → 2.35GB
```

**代码实现**：
```python
scaler = torch.cuda.amp.GradScaler()

# 前向传播（使用 FP16）
with torch.cuda.amp.autocast():
    outputs = model(input_ids, attention_mask, labels)
    loss = outputs.loss

# 反向传播（梯度会被缩放，防止下溢）
scaler.scale(loss).backward()

# 参数更新（先 unscale，再更新）
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
scaler.step(optimizer)
scaler.update()
```

**为什么需要 GradScaler？**
- FP16 的表示范围比 FP32 小，梯度容易出现下溢（underflow）
- GradScaler 会将梯度放大（scale up），防止变成 0
- 更新参数前再缩小回去（unscale），确保数值正确

---

#### **优化策略 2：梯度累积（Gradient Accumulation）**

**问题**：
- 大 batch_size 通常带来更好的收敛（梯度估计更准确）
- 但受限于显存，无法直接使用 batch_size=64

**解决方案**：
```python
# 配置
BATCH_SIZE = 32  # 每次只加载32个样本到显存
GRADIENT_ACCUMULATION_STEPS = 2  # 累积2个batch的梯度
# 等效 batch_size = 32 * 2 = 64

# 训练循环
for step, batch in enumerate(dataloader):
    # 前向传播 + 反向传播
    loss = loss / GRADIENT_ACCUMULATION_STEPS  # ⚠️ 关键：损失除以累积步数
    loss.backward()  # 梯度会累加到 .grad 中

    # 每累积2个batch，才更新一次参数
    if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
        optimizer.step()     # 参数更新
        optimizer.zero_grad()  # 清空梯度
```

**为什么损失要除以累积步数？**
```
正常训练（batch_size=64）:
  loss = mean(loss_per_sample)  # 64个样本的平均损失
  loss.backward()
  → 梯度 = ∂loss/∂w

梯度累积（模拟 batch_size=64）:
  batch1: loss1 = mean(32个样本) / 2
          loss1.backward() → 梯度1 累加
  batch2: loss2 = mean(32个样本) / 2
          loss2.backward() → 梯度2 累加

  最终梯度 = 梯度1 + 梯度2
          = (∂loss1/∂w) + (∂loss2/∂w)
          = ∂(loss1 + loss2)/∂w
          = ∂(mean(64个样本))/∂w  ✅ 等价于 batch_size=64
```

**显存节省**：
```
无梯度累积（batch_size=64）: 5.5GB
有梯度累积（batch_size=32 × 2）: 2.35GB
```

**trade-off**：
- ✅ 节省显存，可以用更大的等效 batch size
- ❌ 训练时间略微增加（参数更新频率降低）
- ❌ 数据遍历次数增加（但梯度估计更准确，收敛更快）

---

## 3. 当前模型性能诊断

### 3.1 测试结果回顾

**整体性能指标**：
```
准确率 (Accuracy):        74.12% (2372/3200)
宏平均F1 (F1-macro):      70.19%
加权F1 (F1-weighted):     74.33%
```

**各类别详细指标**：
| 类别 | 精确率 | 召回率 | F1分数 | 样本数 |
|-----|--------|--------|--------|--------|
| Class 0 (不相似) | 82.15% | 79.92% | 81.02% | 2211 |
| Class 1 (相似) | 57.67% | 61.17% | 59.37% | 989 |

**混淆矩阵**：
```
                    预测: 不相似    预测: 相似    总计    召回率
实际: 不相似 (0)        1767          444       2211    79.92%
实际: 相似 (1)          384           605        989    61.17%
───────────────────────────────────────────────────────────────
总计                   2151          1049       3200
精确率                82.15%        57.67%
```

**错误分析**：
- ✓ 正确预测: 2372 个样本 (74.12%)
- ✗ 错误预测: 828 个样本 (25.88%)
  - 假阳性 (FP): 444 个（将不相似误判为相似）- 13.88%
  - 假阴性 (FN): 384 个（将相似误判为不相似）- 12.00%

---

### 3.2 核心问题分析

#### **问题 1：Class 1（相似类）性能明显低于 Class 0**

**数据对比**：
```
Class 0 (不相似):
  - F1-score: 81.02%
  - 精确率: 82.15%（预测为不相似时，82%是对的）
  - 召回率: 79.92%（实际不相似的，79%被找出来了）
  → 模型对"不相似"的识别能力较强 ✅

Class 1 (相似):
  - F1-score: 59.37%（比 Class 0 低 21.65 个百分点）
  - 精确率: 57.67%（预测为相似时，只有 58% 是对的）
  - 召回率: 61.17%（实际相似的，只有 61% 被找出来了）
  → 模型对"相似"的识别能力较弱 ❌
```

**根本原因**：**类别不平衡 + 模型偏向多数类**

1. **训练集分布不平衡**：
   - Class 0: 69.1%（19,900 个样本）
   - Class 1: 30.9%（8,900 个样本）
   - 比例: 2.23:1

2. **模型学习偏差**：
   ```
   模型看到的:
     - Class 0 样本: 19,900 次（学习充分）
     - Class 1 样本: 8,900 次（学习不足）

   结果:
     模型学到了更多 Class 0 的模式，
     对 Class 1 的特征提取不充分
   ```

3. **虽然使用了类别权重，但效果有限**：
   ```python
   class_weights = [0.72, 1.62]
   # Class 1 的损失权重是 Class 0 的 2.25 倍

   但是:
   - 权重只影响损失计算，不改变样本数量
   - 模型仍然"见过"更多 Class 0 的样例
   - 可能需要更激进的权重（如 [0.5, 2.5]）
   ```

---

#### **问题 2：为什么 Accuracy 看起来还行，但 F1 分数较低？**

**Accuracy 的"陷阱"**：

**场景模拟**：
```python
# 极端情况：模型全部预测为 Class 0
predictions = [0] * 3200
true_labels = [0] * 2211 + [1] * 989

accuracy = (2211) / 3200 = 69.1%  # 看起来还"可以"？
```

**实际情况**：
```
当前模型:
  - 2151 个样本被预测为 Class 0
  - 1049 个样本被预测为 Class 1
  - 模型有一定的区分能力，但仍偏向 Class 0

Accuracy = 74.12%:
  - 其中 1767 个 Class 0 样本预测对了（贡献 55.2%）
  - 其中 605 个 Class 1 样本预测对了（贡献 18.9%）
  - ⚠️ Class 1 的贡献远小于 Class 0
```

**F1-macro 的"真相"**：
```
F1-macro = (F1_class0 + F1_class1) / 2
         = (81.02% + 59.37%) / 2
         = 70.19%

Accuracy 和 F1-macro 的差距:
  74.12% - 70.19% = 3.93%

这个差距揭示:
  → 模型在多数类上表现好，在少数类上表现差
  → Accuracy 被多数类"拉高"了
  → F1-macro 暴露了模型在少数类上的真实能力
```

**为什么 F1-macro 更重要？**

**业务场景分析**：
```
假设这是一个搜索推荐系统:
  用户输入: "借呗能否分期"

  候选句子:
    1. "蚂蚁借呗等额还款" → 实际相似，模型预测不相似（FN）
    2. "花呗如何还款" → 实际不相似，模型预测相似（FP）

  错误代价:
    - FN（漏掉相关结果）: 用户体验下降，可能流失用户 ⚠️⚠️⚠️
    - FP（推荐不相关结果）: 用户略微不满，但可以继续浏览 ⚠️

  结论: FN 的代价通常更高！
```

**当前模型的风险**：
- **FN = 384 个**（12%）：漏掉了 38.8% 的相似句对
- 在推荐/搜索场景中，会导致大量相关内容无法被检索到
- **这是需要重点优化的方向**

---

#### **问题 3：训练过程中的过拟合现象**

**训练历史回顾**：
```
Epoch  Train Loss  Val Loss   Val F1    评价
─────────────────────────────────────────────────
  1      0.5829     0.5345    0.6696    正常学习
  2      0.4857     0.5101    0.6827    正常学习
  3      0.4078     0.5322    0.6977    ✅ 最佳平衡点
  4      0.3354     0.5667    0.7019    ⚠️ 开始过拟合
  5      0.2820     0.6333    0.6987    ❌ 严重过拟合
```

**过拟合的标志**：
```
Train Loss 持续下降: 0.5829 → 0.2820（-52%）
Val Loss 从 Epoch 3 开始上升: 0.5322 → 0.6333（+19%）

可视化:
  Val Loss
    0.65 ┤                              ╭───（过拟合）
    0.60 ┤                         ╭────╯
    0.55 ┤                    ╭────╯
    0.50 ┼───────╮       ╭────╯
    0.45 ┤       ╰───────╯
    0.40 ┤          ╰─────────────────（Train Loss）
    0.35 ┤               ╰──────────
    0.30 ┤                     ╰───────
         └─────┬─────┬─────┬─────┬─────
             Ep1   Ep2   Ep3   Ep4   Ep5
```

**原因分析**：
1. **模型容量过剩**：MacBERT-base 有 102M 参数，但训练集只有 28,800 个样本
2. **类别不平衡加剧过拟合**：模型在 Class 0 上过度拟合（记住了噪声）
3. **Early Stopping 未生效**：虽然有 Early Stopping，但 F1 在 Epoch 4 还在上升

**改进方向**：
- 增加正则化（Dropout、Weight Decay）
- 使用对抗训练（添加噪声，提升泛化能力）
- 数据增强（回译、同义词替换）

---

### 3.3 与阶段二目标的差距

**目标 vs 实际**：
| 指标 | 目标 | 实际 | 差距 | 达成率 |
|-----|------|------|------|--------|
| Accuracy | 80-83% | 74.12% | -5.88~-8.88% | 89.4%~92.9% |
| F1-macro | 76-79% | 70.19% | -5.81~-8.81% | 88.9%~92.3% |

**距离目标还有 6-9 个百分点的提升空间**

---

## 4. 核心技术难点与知识点沉淀

### 4.1 关键技术点总结

#### **4.1.1 BERT 的输入格式：`[CLS]...[SEP]...` 的意义**

**Single Sentence 输入**：
```
[CLS] 今天 天气 很 好 [SEP] [PAD] [PAD] ...
```

**Sentence Pair 输入（AFQMC 使用这种）**：
```
[CLS] 蚂蚁借呗等额还款 [SEP] 借呗能否分期 [SEP] [PAD] ...
```

**Token Type IDs（Segment Embeddings）**：
```
Token:         [CLS] 蚂 蚁 借 呗 ... [SEP] 借 呗 能 否 ... [SEP]
Token Type:      0   0  0  0  0  ...  0    1  1  1  1  ...  1
                 └─────────────────┘  └─────────────────────┘
                    Sentence A            Sentence B
```

**Position Embeddings（位置编码）**：
```
Position:      0   1  2  3  4  ...  10   11 12 13 14 ...  20
```

**最终输入 = Token Embeddings + Segment Embeddings + Position Embeddings**

**`[CLS]` token 的特殊作用**：
- BERT 在预训练时，会将整个句子的语义信息聚合到 `[CLS]` 的输出向量中
- 在 Fine-tuning 阶段，我们直接用 `[CLS]` 的输出做分类
- 数学上，`[CLS]` 通过 Self-Attention 机制，可以"看到"所有其他 token，因此能表示整个句子

---

#### **4.1.2 Logits vs Probabilities vs Predictions**

| 概念 | 计算方式 | 取值范围 | 使用场景 |
|-----|---------|---------|---------|
| **Logits** | 模型最后一层的输出 | (-∞, +∞) | 训练时（与 Loss 配合） |
| **Probabilities** | softmax(logits) | [0, 1]，和为1 | 推理时（输出置信度） |
| **Predictions** | argmax(logits) | {0, 1, ..., n_classes-1} | 最终预测结果 |

**代码示例**：
```python
# 模型输出
logits = model(input_ids, attention_mask).logits  # shape: [batch_size, 2]
# 例: tensor([[2.3, -1.5], [0.5, 1.8], [-0.3, 0.7]])

# 方法1: 训练时（直接用 logits）
loss = nn.CrossEntropyLoss()(logits, labels)
# CrossEntropyLoss 内部会自动 softmax

# 方法2: 推理时（计算概率）
probs = torch.softmax(logits, dim=-1)
# tensor([[0.9820, 0.0180],  # 98% 不相似，2% 相似
#         [0.2227, 0.7773],  # 22% 不相似，78% 相似
#         [0.3775, 0.6225]]) # 38% 不相似，62% 相似

# 方法3: 得到预测类别
preds = torch.argmax(logits, dim=-1)
# tensor([0, 1, 1])  # 第1个样本预测为0，第2、3个预测为1
```

**为什么训练时用 logits？**
- `CrossEntropyLoss` 内部实现了数值稳定的 log-softmax
- 手动先 softmax 再取 log 可能导致数值下溢（概率接近 0 时）

---

#### **4.1.3 DataLoader 的工作原理**

**为什么需要 DataLoader？**

**错误做法（一次性加载全部数据）**：
```python
# ❌ 不推荐
all_data = []
for i in range(28800):
    encoding = tokenizer(texts[i], max_length=64, ...)
    all_data.append(encoding)

# 问题:
# 1. 占用大量内存（28,800 * 64 * 4 bytes ≈ 7.4 MB，还没算模型）
# 2. 数据没有打乱，无法泛化
# 3. 无法并行加载
```

**正确做法（使用 DataLoader）**：
```python
# ✅ 推荐
class AFQMCDataset(Dataset):
    def __getitem__(self, idx):
        # 只在需要时才 tokenize 第 idx 个样本
        text1 = self.texts1[idx]
        text2 = self.texts2[idx]
        encoding = self.tokenizer(text1, text2, ...)
        return encoding

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 优点:
# 1. 懒加载（Lazy Loading）: 只在需要时才处理数据
# 2. 自动批处理（Batching）
# 3. 自动打乱（Shuffling）
# 4. 多进程加载（num_workers，Windows 上需=0）
```

**DataLoader 的内部流程**：
```
DataLoader 初始化
    ↓
创建 Sampler（决定取样顺序）
  - shuffle=True → RandomSampler
  - shuffle=False → SequentialSampler
    ↓
for batch in dataloader:
    1. Sampler 生成一个 batch 的索引: [3, 17, 5, 29, ...]
    2. 调用 Dataset.__getitem__(idx) 获取每个样本
    3. Collate Function 将样本拼接成 batch
       - input_ids: [32, 64]
       - attention_mask: [32, 64]
       - labels: [32]
    4. 返回 batch 字典
```

---

#### **4.1.4 学习率调度：Warmup + Linear Decay**

**为什么需要 Warmup？**

**问题场景**：
```python
# 不使用 Warmup
optimizer = AdamW(model.parameters(), lr=2e-5)

# 训练刚开始时:
# - 模型参数是随机初始化的（或预训练权重与任务不匹配）
# - 如果直接用 lr=2e-5，梯度可能很大，导致参数剧烈震荡
# - 可能直接"跳"过最优解，或陷入局部最优
```

**Warmup 策略**：
```
Epoch 1 的前 10% 步数:
  lr = 0 → 2e-5 （线性增长）

剩余 90% 步数:
  lr = 2e-5 → 0 （线性衰减）

可视化:
  2e-5 ┤      ╭───────╮
       │     ╱         ╲
       │    ╱           ╲
       │   ╱             ╲
       │  ╱               ╲
   0   ┼─╯                 ╲___
       └─────┬──────┬──────┬─────
           Warmup  Train   End
           (10%)   (90%)
```

**实现代码**：
```python
from transformers import get_linear_schedule_with_warmup

total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION_STEPS
warmup_steps = int(total_steps * WARMUP_RATIO)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# 训练循环中
for batch in train_loader:
    ...
    optimizer.step()
    scheduler.step()  # ⚠️ 每个 batch 后都要调用
```

**Warmup 的作用**：
1. **稳定训练初期**：避免初始梯度过大导致的参数震荡
2. **提升收敛速度**：预训练模型逐渐适应新任务
3. **提升最终性能**：通常能提升 0.5-1 个百分点

---

#### **4.1.5 梯度裁剪（Gradient Clipping）的数学原理**

**为什么需要梯度裁剪？**

**梯度爆炸场景**：
```python
# 某一层的梯度
layer1_grad = torch.tensor([100.5, -200.3, 50.1])  # 梯度很大
layer2_grad = torch.tensor([0.001, 0.002, -0.003])

# L2 范数（梯度的"总大小"）
grad_norm = sqrt(100.5² + 200.3² + 50.1² + 0.001² + ...)
          ≈ 225.4

# 如果 grad_norm > MAX_GRAD_NORM:
#   → 参数更新太大，可能"跳"过最优解
#   → Loss 震荡或 NaN
```

**裁剪策略**：
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 实现原理:
# 1. 计算所有参数梯度的 L2 范数
total_norm = sqrt(sum(p.grad.data.norm(2) ** 2 for p in model.parameters()))

# 2. 如果超过阈值，等比例缩小所有梯度
if total_norm > max_norm:
    clip_coef = max_norm / (total_norm + 1e-6)
    for p in model.parameters():
        p.grad.data.mul_(clip_coef)

# 例子:
# 原梯度: [100.5, -200.3, 50.1]
# total_norm = 225.4 > 1.0
# clip_coef = 1.0 / 225.4 ≈ 0.00444
# 裁剪后: [0.446, -0.889, 0.222]  ← 方向不变，大小被限制
```

**为什么用 L2 范数而不是直接限制每个梯度？**
```python
# 错误方法: 直接限制每个梯度
for p in model.parameters():
    p.grad.data.clamp_(-1.0, 1.0)  # ❌ 会改变梯度方向

# 正确方法: 等比例缩放
# ✅ 保持梯度方向，只调整大小
```

---

### 4.2 新手常见的"坑"及解决方案

#### **坑 1：显存溢出（CUDA Out of Memory）**

**错误信息**：
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
(GPU 0; 8.00 GiB total capacity; 6.50 GiB already allocated; ...)
```

**常见原因**：
1. **batch_size 太大**
2. **序列长度太长**（如 max_length=512）
3. **未使用 AMP**
4. **在评估时未使用 `torch.no_grad()`**

**解决方案**：
```python
# ✅ 方案1: 减小 batch_size
BATCH_SIZE = 16  # 从 32 → 16

# ✅ 方案2: 使用梯度累积（模拟大 batch）
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4  # 等效 batch_size = 64

# ✅ 方案3: 启用 AMP
USE_AMP = True

# ✅ 方案4: 评估时不计算梯度
with torch.no_grad():  # ⚠️ 关键
    outputs = model(input_ids, attention_mask)

# ✅ 方案5: 及时清空显存
del loss, outputs
torch.cuda.empty_cache()
```

---

#### **坑 2：Loss 不收敛（一直是随机值）**

**现象**：
```
Epoch 1, Loss: 0.693
Epoch 2, Loss: 0.691
Epoch 3, Loss: 0.694
Epoch 4, Loss: 0.692
...
```

**原因排查**：

1. **标签格式错误**
   ```python
   # ❌ 错误：标签是字符串
   labels = ["0", "1", "0", "1"]

   # ✅ 正确：标签是整数
   labels = [0, 1, 0, 1]
   labels = torch.tensor(labels, dtype=torch.long)
   ```

2. **学习率设置不当**
   ```python
   # ❌ 太小：模型几乎不学习
   LEARNING_RATE = 1e-8

   # ❌ 太大：Loss 震荡
   LEARNING_RATE = 1e-2

   # ✅ 推荐范围（BERT 系列）
   LEARNING_RATE = 2e-5  # 或 3e-5, 5e-5
   ```

3. **模型参数未更新**
   ```python
   # ❌ 忘记调用 optimizer.step()
   loss.backward()
   # optimizer.step()  ← 忘记了

   # ✅ 正确流程
   loss.backward()
   optimizer.step()
   optimizer.zero_grad()
   ```

4. **预训练模型加载失败**
   ```python
   # 检查模型是否正确加载
   print(model.bert.embeddings.word_embeddings.weight[0][:5])
   # 如果都是很小的随机数（如 1e-5），说明加载失败
   ```

---

#### **坑 3：维度不匹配（Shape Mismatch）**

**错误信息**：
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x768 and 512x2)
```

**常见场景**：

1. **分类层输出维度错误**
   ```python
   # ❌ 错误：输出维度与类别数不匹配
   classifier = nn.Linear(768, 3)  # 但实际只有 2 个类别

   # ✅ 正确
   classifier = nn.Linear(768, 2)  # num_classes=2
   ```

2. **取错位置的输出**
   ```python
   # ❌ 错误：取了整个 sequence 的输出
   outputs = model(input_ids, attention_mask)
   logits = outputs.last_hidden_state  # shape: [batch_size, seq_len, 768]

   # ✅ 正确：只取 [CLS] token
   cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
   # 或者直接用 pooler_output
   cls_output = outputs.pooler_output  # [batch_size, 768]
   ```

3. **Loss 函数输入格式错误**
   ```python
   # ❌ 错误：标签是 one-hot 编码
   labels = torch.tensor([[1, 0], [0, 1]])  # shape: [batch_size, 2]
   loss = nn.CrossEntropyLoss()(logits, labels)  # 报错

   # ✅ 正确：标签是类别索引
   labels = torch.tensor([0, 1])  # shape: [batch_size]
   loss = nn.CrossEntropyLoss()(logits, labels)
   ```

**调试技巧**：
```python
# 在关键位置打印 shape
print(f"input_ids shape: {input_ids.shape}")
print(f"outputs shape: {outputs.last_hidden_state.shape}")
print(f"logits shape: {logits.shape}")
print(f"labels shape: {labels.shape}")
```

---

#### **坑 4：训练速度极慢**

**问题表现**：
```
每个 batch 需要 5-10 秒
预计训练完成时间: 3 小时+
```

**原因排查**：

1. **未使用 GPU**
   ```python
   # 检查
   print(f"使用设备: {device}")  # 应该是 cuda:0
   print(f"GPU 是否可用: {torch.cuda.is_available()}")

   # 确保数据和模型都在 GPU 上
   model = model.to(device)
   input_ids = input_ids.to(device)
   ```

2. **DataLoader 的 num_workers 设置不当**
   ```python
   # ❌ Windows 上设置 num_workers > 0 可能导致卡死
   dataloader = DataLoader(dataset, num_workers=4)  # Windows 慎用

   # ✅ Windows 推荐
   dataloader = DataLoader(dataset, num_workers=0)
   ```

3. **未启用 AMP**
   ```python
   # ✅ 启用 AMP（速度提升 1.5-2 倍）
   USE_AMP = True
   scaler = torch.cuda.amp.GradScaler()

   with torch.cuda.amp.autocast():
       outputs = model(...)
   ```

4. **tqdm 进度条刷新太频繁**
   ```python
   # ✅ 降低刷新频率
   pbar = tqdm(dataloader, desc="Training", mininterval=1.0)
   ```

---

#### **坑 5：模型预测全是同一个类别**

**现象**：
```
模型预测: [0, 0, 0, 0, 0, 0, ...]
Accuracy: 69%（恰好等于多数类比例）
```

**原因**：

1. **类别权重未生效**
   ```python
   # 检查是否正确传入
   criterion = nn.CrossEntropyLoss(weight=class_weights)
   ```

2. **学习率太大，模型学偏了**
   ```python
   # 降低学习率重新尝试
   LEARNING_RATE = 1e-5  # 从 2e-5 → 1e-5
   ```

3. **训练不充分**
   ```python
   # 增加训练轮数
   NUM_EPOCHS = 5  # 从 3 → 5
   ```

4. **数据预处理错误**
   ```python
   # 检查标签分布
   print(f"训练集标签分布: {np.bincount(train_labels)}")
   # 应该是 [19900, 8900] 左右

   # 如果全是同一个类别，说明数据加载有问题
   ```

---

## 5. 面试官视角的 Q&A

### **Q1: 为什么选择 MacBERT 而不是原生 BERT？在中文 NLP 任务中，它们的核心差异是什么？**

**标准答案（P7 级别）**：

**核心差异**：
1. **预训练掩码策略不同**
   - **原生 BERT**：随机选择 15% 的 token，其中 80% 替换为 `[MASK]`，10% 替换为随机 token，10% 保持不变
   - **MacBERT**：使用 **WWM（Whole Word Masking，全词掩码）+ 相似词替换** 策略
     - 掩码时以"词"为单位（而非字符），更符合中文语言特性
     - 不使用 `[MASK]` token，而是用**相似词替换**，减少 pretrain-finetune gap

2. **N-gram Masking**
   - **原生 BERT**：单个 token 掩码
   - **MacBERT**：支持 N-gram masking（如掩码整个词组"蚂蚁借呗"）

3. **性能对比**（中文任务）
   - 在 CLUE、CMRC 等中文基准测试中，MacBERT 平均领先 BERT 0.5-1.5 个百分点
   - 特别是在需要理解词组语义的任务中（如本项目的文本匹配），优势更明显

**技术深度理解**：

**Pretrain-Finetune Gap 问题**：
```
预训练阶段:
  输入: "我喜欢吃[MASK]果"
  模型学习: 预测 [MASK] = "苹"

  问题: [MASK] 是一个特殊 token，在真实文本中不存在

微调阶段:
  输入: "我喜欢吃苹果"（没有 [MASK]）

  → 输入分布不一致，影响性能
```

**MacBERT 的改进**：
```
预训练阶段:
  原句: "我喜欢吃苹果"
  掩码: "我喜欢吃水果"（用相似词"水果"替换"苹果"）
  模型学习: 预测原词 = "苹果"

  优势: 输入是自然文本，与微调阶段一致

微调阶段:
  输入: "我喜欢吃苹果"

  → 输入分布一致，性能提升 ✅
```

**为什么在 AFQMC 任务中选择 MacBERT？**
1. **任务特性**：句子相似度匹配需要精准捕捉词级语义
   - 例："借呗能否分期" vs "借呗可以分期吗"
   - 需要理解"能否"和"可以"在语义上的相似性
   - MacBERT 的词级掩码训练更有利于这种理解

2. **中文特性**：中文是以词为基本单位（不是字符）
   - MacBERT 的 WWM 更符合中文语言规律

3. **实验验证**：在多个中文匹配任务中，MacBERT 比 BERT 高 0.5-1%

**加分项**：提到实际工程选择
```
工程选择策略:
  1. 任务难度低（如垃圾邮件分类）: BERT 足够
  2. 任务难度中（如本项目）: MacBERT（性价比最高）
  3. 任务难度高 + 资源充足: RoBERTa-wwm-ext-large
  4. 需要极致性能: 模型融合（BERT + MacBERT + RoBERTa）
```

---

### **Q2: 在有限的显存（8GB）下，如何通过 AMP 和梯度累积实现等效的大 batch size 训练？请从数学原理和工程实现两个角度解释。**

**标准答案（P7 级别）**：

#### **Part 1: 数学原理**

**标准 SGD 的梯度更新**：
```
给定 batch B = {x_1, x_2, ..., x_n}:

1. 前向传播，计算损失:
   L(B) = (1/n) * Σ L(x_i, y_i)

2. 反向传播，计算梯度:
   ∇L = (1/n) * Σ ∇L(x_i)

3. 参数更新:
   θ ← θ - lr * ∇L
```

**梯度累积的数学等价性**：
```
假设我们想用 batch_size=64，但显存只能容纳 32:

方法1（理想但 OOM）:
  B = {x_1, ..., x_64}
  L = (1/64) * Σ L(x_i)
  ∇L = (1/64) * Σ ∇L(x_i)
  θ ← θ - lr * ∇L

方法2（梯度累积）:
  B1 = {x_1, ..., x_32}
  B2 = {x_33, ..., x_64}

  # 第一个 mini-batch
  L1 = (1/64) * Σ_{i=1}^{32} L(x_i)  ← 注意：除以 64，不是 32
  L1.backward()  → 梯度累加到 .grad

  # 第二个 mini-batch
  L2 = (1/64) * Σ_{i=33}^{64} L(x_i)
  L2.backward()  → 梯度继续累加

  # 最终梯度
  ∇L_total = ∇L1 + ∇L2
           = (1/64) * [Σ_{i=1}^{32} ∇L(x_i) + Σ_{i=33}^{64} ∇L(x_i)]
           = (1/64) * Σ_{i=1}^{64} ∇L(x_i)
           ✅ 等价于 batch_size=64 的梯度

  θ ← θ - lr * ∇L_total
```

**关键点**：损失必须除以总的 batch size（64），而不是 mini-batch size（32）

---

#### **Part 2: AMP 的数值稳定性**

**FP32 vs FP16 的表示范围**：
```
FP32 (单精度):
  - 指数位: 8 bits  → 表示范围: [10^-38, 10^38]
  - 尾数位: 23 bits → 精度: ~7 位有效数字

FP16 (半精度):
  - 指数位: 5 bits  → 表示范围: [10^-8, 65504]
  - 尾数位: 10 bits → 精度: ~3 位有效数字

问题: 梯度通常很小（如 1e-5），容易下溢变成 0
```

**GradScaler 的工作原理**：
```
训练步骤                     FP16 值              实际值（FP32）
─────────────────────────────────────────────────────────────
1. 前向传播
   loss = 0.0001             0.0001 × 2^16        0.0001
   (接近下溢边界)              = 6.5536

2. 梯度缩放
   scaled_loss = loss * 2^16  6.5536              0.0001

3. 反向传播
   grad = ∂(scaled_loss)/∂w   0.032 × 2^16        0.032
   (梯度被放大，避免下溢)       = 2097.152

4. 梯度还原（unscale）
   unscaled_grad = grad / 2^16  0.032             0.032

5. 梯度裁剪
   clip_grad_norm_(grad, 1.0)

6. 参数更新
   w = w - lr * grad
```

**动态调整 scale factor**：
```python
scaler = torch.cuda.amp.GradScaler()

# 训练循环中
for batch in dataloader:
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()  # ⚠️ 关键

# update() 的逻辑:
if 检测到梯度溢出（NaN 或 Inf）:
    scale_factor *= 0.5  # 减小 scale
    跳过本次更新
else:
    连续成功 2000 次后:
        scale_factor *= 2  # 增大 scale（更激进地利用 FP16）
```

---

#### **Part 3: 工程实现细节**

```python
# 完整的训练步骤
def train_epoch(model, dataloader, optimizer, scheduler,
                scaler, accumulation_steps=2):
    model.train()
    optimizer.zero_grad()  # ⚠️ 放在循环外

    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # ========== AMP 前向传播 ==========
        with torch.cuda.amp.autocast():  # 自动选择 FP16/FP32
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs.loss

            # ⚠️ 关键：损失除以累积步数
            loss = loss / accumulation_steps

        # ========== 反向传播（梯度累积） ==========
        scaler.scale(loss).backward()  # 缩放后反向传播

        # ========== 参数更新（每 N 步） ==========
        if (step + 1) % accumulation_steps == 0:
            # 1. 梯度还原（unscale）
            scaler.unscale_(optimizer)

            # 2. 梯度裁剪（必须在 unscale 后）
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 3. 参数更新
            scaler.step(optimizer)

            # 4. 更新 scaler 的 scale factor
            scaler.update()

            # 5. 学习率调度
            scheduler.step()

            # 6. 清空梯度
            optimizer.zero_grad()
```

**常见错误**：
```python
# ❌ 错误1：忘记除以累积步数
loss = outputs.loss  # 应该是 loss / accumulation_steps
loss.backward()

# ❌ 错误2：梯度裁剪在 unscale 之前
scaler.scale(loss).backward()
torch.nn.utils.clip_grad_norm_(...)  # 此时梯度还是缩放后的
scaler.unscale_(optimizer)

# ❌ 错误3：每个 step 都 zero_grad
for step, batch in enumerate(dataloader):
    optimizer.zero_grad()  # ❌ 会清空累积的梯度
    ...
```

**显存节省对比**：
```
配置                         显存占用      训练速度    等效 batch
─────────────────────────────────────────────────────────────
FP32, batch=32              5.5 GB       1.0x        32
FP32, batch=64              OOM          -           -
FP16 (AMP), batch=32        3.2 GB       1.5x        32
FP16 + 梯度累积(x2)          3.2 GB       0.9x        64  ✅
FP16 + 梯度累积(x4)          3.2 GB       0.85x       128
```

**Trade-off 分析**：
- ✅ 显存节省：41.8%（5.5GB → 3.2GB）
- ✅ 速度提升：1.5x（FP16 的硬件加速）
- ⚠️ 精度损失：<0.1%（可忽略）
- ⚠️ 训练时间：+10%（梯度累积导致更新频率降低）

---

### **Q3: 当前模型在 Class 1（相似类）上的 F1 只有 59.37%，而 Class 0（不相似类）达到了 81.02%。请从数据、模型和训练策略三个角度分析原因，并提出改进方案。**

**标准答案（P7 级别）**：

#### **Part 1: 根因分析**

**1. 数据层面的问题**

**(1) 类别不平衡（根本原因）**
```
训练集分布:
  Class 0 (不相似): 19,900 个样本（69.1%）
  Class 1 (相似): 8,900 个样本（30.9%）
  比例: 2.23:1

影响:
  - 模型"见过"更多 Class 0 的样例
  - 学到的特征偏向 Class 0
  - 优化目标被 Class 0 主导
```

**(2) 少数类样本质量问题**
```python
# 可能存在的问题
1. 标注噪声:
   - Class 1 样本的标注错误率可能更高
   - "相似"的定义比"不相似"更主观

2. 样本多样性不足:
   - Class 1 的 8,900 个样本可能覆盖的模式较少
   - Class 0 的样本可能涵盖了更多元的"不相似"场景

3. 边界样本:
   - "相似"和"不相似"的边界很模糊
   - 模型倾向于保守预测（预测 Class 0）
```

**(3) 特征空间分布**
```
假设在 BERT 的特征空间中:
  Class 0: 分布广泛，占据特征空间的大部分区域
  Class 1: 分布集中，只占据小部分区域

  → 模型的决策边界倾向于将大部分空间划分给 Class 0
```

---

**2. 模型层面的问题**

**(1) 模型容量分配不均**
```
MacBERT-base 的 102M 参数:
  - 大部分参数学习 Class 0 的特征（因为见到的多）
  - 只有少部分参数专注于 Class 1

  类比: 100 个员工，69 个处理任务 A，31 个处理任务 B
       → 任务 A 的效率自然更高
```

**(2) 最后一层分类器的偏置**
```python
classifier = nn.Linear(768, 2)

# 分类器的偏置项（bias）
# 训练后可能是: bias = [0.5, -0.3]
# → 默认倾向于预测 Class 0

# 即使输入特征相同，logits 也会是:
# logits = [score + 0.5, score - 0.3]
#        → Class 0 的 logit 天然更高
```

---

**3. 训练策略的问题**

**(1) 类别权重不够激进**
```python
当前权重:
  class_weights = [0.72, 1.62]
  # Class 1 的损失权重是 Class 0 的 2.25 倍

问题:
  - 虽然 1.62 > 0.72，但可能还不够
  - 需要更激进的权重（如 [0.5, 2.5]）
```

**(2) 优化目标与评估指标不一致**
```
训练目标: CrossEntropyLoss（最小化总损失）
评估指标: F1-macro（关注每个类别的平衡）

问题:
  - CrossEntropyLoss 会被样本数多的类别主导
  - 即使 Class 1 的 F1 很低，只要 Class 0 的损失够低，
    总损失仍然可以很低
```

**(3) 过拟合现象**
```
训练历史:
  Epoch 3: train_loss=0.4078, val_loss=0.5322  ✅ 平衡
  Epoch 4: train_loss=0.3354, val_loss=0.5667  ⚠️ 开始过拟合
  Epoch 5: train_loss=0.2820, val_loss=0.6333  ❌ 严重过拟合

影响:
  - 模型在 Class 0 上过拟合（记住了噪声）
  - Class 1 样本少，更容易泛化失败
```

---

#### **Part 2: 改进方案（分层推进）**

**🥇 优先级1：数据层面优化（成本低，收益高）**

**(1) 调整类别权重（立即可做）**
```python
# 当前
class_weights = [0.72, 1.62]  # 基于 sklearn 的 'balanced' 策略

# 改进方案1: 更激进的权重
class_weights = [0.5, 2.5]  # 手动设置，提高 Class 1 的重要性

# 改进方案2: Focal Loss（更先进）
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 聚焦参数（难样本权重更高）

    def forward(self, logits, labels):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(logits, labels)
        pt = torch.exp(-ce_loss)  # 预测概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

# Focal Loss 的优势:
# - 自动给"难样本"更高权重
# - 对于 Class 1 的难区分样本，损失会更大
# - RetinaNet 论文中证明在不平衡问题上优于 CrossEntropy
```

预期提升：F1-macro +1~2%

---

**(2) 数据增强（Data Augmentation）**
```python
# 方案1: 回译（Back Translation）
def back_translate(text):
    # 中文 → 英文 → 中文
    # "蚂蚁借呗等额还款"
    # → "Ant borrow equal repayment"
    # → "蚂蚁借款等额偿还"
    pass

# 方案2: 同义词替换
def synonym_replacement(text, n=3):
    words = jieba.lcut(text)
    # 随机替换 n 个词为同义词
    # "借呗能否分期" → "借呗可以分期吗"
    pass

# 方案3: 对 Class 1 进行过采样
from imblearn.over_sampling import SMOTE, RandomOverSampler

# 在特征空间中对少数类进行插值生成新样本
```

预期提升：F1-macro +2~3%

---

**(3) 重新标注边界样本**
```python
# 识别模型"不确定"的样本
def find_uncertain_samples(model, dataloader, threshold=0.6):
    uncertain_samples = []
    for batch in dataloader:
        logits = model(batch)
        probs = torch.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1)[0]  # 最大概率

        # 如果最大概率 < 0.6，说明模型不确定
        uncertain_idx = (max_prob < threshold).nonzero()
        uncertain_samples.extend(uncertain_idx)

    return uncertain_samples

# 人工复核这些样本，重新标注
# 对于 Class 1，尤其需要确保标注质量
```

---

**🥈 优先级2：训练策略优化（工程实现）**

**(1) 对抗训练（FGM/PGD）**
```python
# Fast Gradient Method (FGM)
class FGM:
    def __init__(self, model, epsilon=1.0):
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self):
        # 在 embedding 层添加扰动
        for name, param in self.model.named_parameters():
            if 'word_embeddings' in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    # 朝着梯度方向添加扰动
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        # 恢复原始 embedding
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

# 训练循环中
fgm = FGM(model, epsilon=1.0)

for batch in dataloader:
    # 1. 正常训练
    loss = model(batch).loss
    loss.backward()

    # 2. 对抗训练
    fgm.attack()  # 添加扰动
    loss_adv = model(batch).loss  # 在对抗样本上计算损失
    loss_adv.backward()  # 累积梯度
    fgm.restore()  # 恢复 embedding

    # 3. 参数更新
    optimizer.step()
```

**对抗训练的作用**：
- 提升模型鲁棒性（对输入的微小变化不敏感）
- 减少过拟合（强迫模型学习更本质的特征）
- 对少数类尤其有效（因为样本少，更需要泛化能力）

预期提升：F1-macro +2~4%

---

**(2) 阈值调整（Threshold Tuning）**
```python
# 默认预测（argmax）
preds = torch.argmax(logits, dim=-1)
# 等价于: preds = (probs[:, 1] > 0.5).long()

# 优化后: 降低 Class 1 的阈值
probs = torch.softmax(logits, dim=-1)
preds = (probs[:, 1] > 0.4).long()  # 阈值从 0.5 → 0.4

# 效果:
# - Class 1 的召回率 ↑（更容易被预测为 Class 1）
# - Class 1 的精确率 ↓（会有更多误判）
# - 但整体 F1 可能提升（如果召回率提升 > 精确率下降）
```

**阈值搜索**：
```python
from sklearn.metrics import f1_score

best_threshold = 0.5
best_f1 = 0

for threshold in np.arange(0.3, 0.7, 0.05):
    preds = (probs[:, 1] > threshold).long()
    f1 = f1_score(labels, preds, average='macro')
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"最佳阈值: {best_threshold}, F1-macro: {best_f1}")
```

预期提升：F1-macro +0.5~1%

---

**(3) 多任务学习（Advanced）**
```python
# 主任务: 二分类（相似/不相似）
# 辅助任务: 语义相似度回归（0-1 连续值）

class MultiTaskModel(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(768, 2)  # 分类任务
        self.regressor = nn.Linear(768, 1)   # 回归任务

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_output)
        similarity_score = torch.sigmoid(self.regressor(cls_output))

        return logits, similarity_score

# 损失函数
loss_cls = nn.CrossEntropyLoss()(logits, labels)
loss_reg = nn.MSELoss()(similarity_score, true_similarity)
total_loss = loss_cls + 0.5 * loss_reg  # 权重可调
```

**优势**：
- 回归任务提供更细粒度的监督信号
- 迫使模型学习连续的语义相似度，而不只是硬分类
- 对边界样本（接近 0.5 的样本）学习更充分

预期提升：F1-macro +1~2%

---

**🥉 优先级3：模型层面优化（资源需求高）**

**(1) 换用更大的模型**
```
MacBERT-base → RoBERTa-wwm-ext-large
  参数量: 102M → 325M
  隐层维度: 768 → 1024

预期提升: F1-macro +2~3%
代价: 显存需求增加（需要更激进的优化）
```

**(2) 模型融合（Ensemble）**
```python
# 训练多个模型
models = [
    MacBERT_base,
    RoBERTa_base,
    ERNIE_base
]

# 预测时投票或平均
logits_list = [model(batch) for model in models]
avg_logits = torch.stack(logits_list).mean(dim=0)
preds = torch.argmax(avg_logits, dim=-1)
```

预期提升：F1-macro +2~4%
代价：推理时间增加 3 倍

---

#### **Part 3: 实施路线图**

**短期（1-2 天）**：
1. ✅ 调整类别权重：`[0.5, 2.5]`
2. ✅ 降低预测阈值：`0.4` → 搜索最佳阈值
3. ✅ 增加 Early Stopping patience：`3 → 5`

预期：F1-macro 从 70% → 72%

**中期（3-5 天）**：
4. ✅ 实现 FGM 对抗训练
5. ✅ 数据增强（回译 + 同义词替换）
6. ✅ 使用 Focal Loss

预期：F1-macro 从 72% → 75-76%（达到阶段二目标）

**长期（1-2 周）**：
7. ✅ 换用 RoBERTa-large
8. ✅ 模型融合
9. ✅ 多任务学习

预期：F1-macro 从 76% → 78-80%（超越阶段二目标）

---

**加分项：提到业务思考**
```
在实际业务中，还需要考虑:
  1. 错误代价不对等:
     - FN（漏掉相关内容）的代价可能 > FP（推荐不相关）
     - 应该优先优化召回率（recall）

  2. AB 测试:
     - 不能只看离线指标（F1-macro）
     - 需要上线 AB 测试，看对用户体验的实际影响

  3. 可解释性:
     - 对于预测错误的样本，需要分析模型的注意力
     - 使用 LIME/SHAP 解释为什么模型做出这样的预测
```

---

## 总结

### 本阶段关键成果

✅ **完成项**：
1. 成功搭建 MacBERT 基线模型，达到 70.19% F1-macro
2. 验证了完整的训练流程（数据加载 → 训练 → 评估）
3. 针对 8GB 显存进行了有效优化（AMP + 梯度累积）
4. 识别出核心问题（类别不平衡 + 过拟合）

⚠️ **待改进项**：
1. Class 1 性能明显低于 Class 0（F1 差距 21 个百分点）
2. 距离阶段二目标（76-79% F1-macro）还有 6-9 个百分点
3. 训练后期出现过拟合现象

### 技术能力提升

通过本阶段，你应该掌握了：
- ✅ BERT 系列模型的完整 Fine-tuning 流程
- ✅ 数据不平衡的处理策略
- ✅ 显存优化的工程实践（AMP + 梯度累积）
- ✅ 深度学习训练的调试能力
- ✅ 性能分析与诊断思路

### 面试准备建议

**高频考点**：
1. BERT 的输入格式和预训练任务
2. AMP 和梯度累积的原理与实现
3. 类别不平衡的处理方法
4. 过拟合的识别与解决
5. F1-macro vs Accuracy 的适用场景

**建议准备的深度问题**：
- "如何在有限资源下训练大模型？"
- "类别不平衡会导致什么问题？如何解决？"
- "为什么你的模型在多数类上表现好，少数类上表现差？"

---

**下一步：阶段三（对抗训练）**

根据前面的分析，建议直接进入阶段三，使用 FGM 对抗训练来：
1. 解决过拟合问题
2. 提升模型泛化能力
3. 重点改善 Class 1 的性能

预期：F1-macro 从 70% → 74-76%，达到阶段二目标 🎯

---

> **文档生成时间**: 2026-01-27
> **版本**: v1.0
> **适用阶段**: 阶段二总结与阶段三准备
> **下次更新**: 完成阶段三后

