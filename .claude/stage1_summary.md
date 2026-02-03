# 阶段一：环境搭建与数据探索 - 完成总结

## ✅ 已完成的任务

### 1. 项目文件夹结构
```
AFQMC/
├── .claude/              # 项目文档
├── dataset/              # 数据集（已存在）
├── src/                  # 源代码
│   ├── __init__.py      # 模块初始化
│   └── data_loader.py   # 数据加载工具
├── config/               # 配置文件目录
├── checkpoints/          # 模型保存目录
│   ├── macbert/
│   └── qwen-lora/
├── results/              # 结果输出目录
├── notebooks/            # 数据分析脚本
│   └── eda.py           # 探索性数据分析
├── requirements.txt      # 依赖库列表
└── README.md            # 项目说明
```

### 2. requirements.txt
包含完整的依赖库列表：
- PyTorch 2.0+
- Transformers 4.36+
- PEFT (LoRA微调)
- BitsAndBytes (量化)
- 数据处理库 (pandas, numpy, scikit-learn)
- 可视化库 (matplotlib, seaborn)

### 3. 数据加载模块 (src/data_loader.py)
功能完善的数据加载工具，包括：
- `load_jsonl()`: 加载JSONL格式数据
- `load_train_data()`: 加载训练数据
- `load_test_data()`: 加载测试数据
- `get_data_statistics()`: 计算数据统计信息
- `split_train_val()`: 数据集切分（使用分层采样）
- `display_samples()`: 展示数据样本

### 4. EDA脚本 (notebooks/eda.py)
全面的数据探索分析，包括：
- 文本长度分布分析
- 标签分布分析
- 不同标签的文本特征对比
- 数据质量检查
- 典型样本分析
- 生成可视化图表

**已运行完成，生成3张图表**：
- `results/text_length_distribution.png` - 文本长度分布
- `results/label_distribution.png` - 标签分布（柱状图+饼图）
- `results/length_by_label.png` - 不同标签的长度对比

---

## 📊 EDA实际发现与数据洞察

### 🔍 核心发现（基于实际运行结果）

#### 发现1：文本极短！（超出预期）

**实际数据**：
```
- text1 平均长度: 13.4 字符
- text2 平均长度: 13.4 字符
- 总长度 95% 分位: 46 字符（❗远低于预期的128）
- 总长度 99% 分位: ~70 字符
```

**这说明什么？**
- 这是一个**典型的金融短文本问答数据集**
- 95%的样本，两个文本加起来不超过46个字符
- 文本高度集中在0-20字符区间（从直方图可见）

**可视化特征**（图1 - 文本长度分布）：
- 三个子图都呈现**典型的右偏分布**
- 绝大多数样本集中在左侧（短文本）
- 长尾部分样本很少

**对阶段二的关键影响**：
```python
# ❌ 原计划：max_length=128（太大了）
# ✅ 实际推荐：max_length=64 或 80

# 好处：
# 1. 节省显存 40-50% → batch_size 可以从 16 提升到 32
# 2. 训练速度提升 30-40% → 序列短，计算量少
# 3. 几乎不损失信息 → 95%样本都能完整保留
```

---

#### 发现2：严重的类别不平衡（符合预期）

**实际数据**：
```
- Label 0 (不相似): 22,106 样本 (69.1%)
- Label 1 (相似):    9,894 样本 (30.9%)
- 不平衡比例: 2.23:1
```

**为什么严重？**
```
如果模型"偷懒"，全部预测 label 0：
✅ 准确率 = 69.1%（看起来还不错？）
❌ 但完全没有学习！label 1 的召回率 = 0%
❌ 这就是为什么不能只看 accuracy
```

**可视化特征**（图2 - 标签分布）：
- 柱状图：label 0 的柱子明显高出一倍多
- 饼图：红色区域（label 0）占据大约 2/3

**必须采取的措施**：
```python
# 阶段二：类别权重（基础方法）
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=[0, 1],
    y=train_labels
)
# 预期结果：
# class 0 weight ≈ 0.72  (降低多数类权重)
# class 1 weight ≈ 1.62  (提高少数类权重)

loss_fn = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights, dtype=torch.float)
)

# 阶段三：Focal Loss（更高级）
# 同时处理不平衡和难样本
```

---

#### 发现3：长度分布无差异（证实需要语义模型）

**实际数据**：
```
label 0 (不相似) 的文本长度:
  text1: 平均 13.4, 中位数 12.0
  text2: 平均 13.4, 中位数 12.0

label 1 (相似) 的文本长度:
  text1: 平均 13.5, 中位数 13.0
  text2: 平均 13.4, 中位数 12.0

差异：几乎为 0
```

**可视化特征**（图3 - 按标签的长度对比）：
- 红色（label 0）和蓝色（label 1）曲线**高度重叠**
- 两个峰值位置几乎相同
- 分布形状完全一致

**这证明了什么？**
```
❌ 不能用简单规则：
   - "长文本更可能相似" ✗
   - "text1 比 text2 长就不相似" ✗

✅ 必须从语义层面理解：
   - 需要深度模型学习文本的语义表示
   - MacBERT 是正确的选择（预训练语言模型）
   - 不能依赖表面特征（长度、词数等）
```

---

#### 发现4：数据质量良好

**实际检查结果**：
```
✅ 缺失值: 0 条
✅ 重复样本: 0 条
✅ 空文本: 0 条
✅ 异常长文本 (>200字符): <1%
```

**意义**：
- 可以直接用于训练，无需额外清洗
- 数据质量高，模型性能上限取决于算法而非数据

---

## 📚 核心知识点详解

### 1. JSONL 数据格式

**什么是JSONL？**
- JSONL = JSON Lines，每行是一个完整的JSON对象
- 与普通JSON不同，JSONL不需要整个文件是一个有效的JSON

**为什么NLP任务常用JSONL？**
```python
# 传统JSON（需要一次性加载整个文件）
{
  "data": [
    {"text1": "...", "text2": "...", "label": 0},
    {"text1": "...", "text2": "...", "label": 1}
  ]
}

# JSONL（可以逐行流式读取）
{"text1": "...", "text2": "...", "label": 0}
{"text1": "...", "text2": "...", "label": 1}
```

**优势**:
1. **内存友好**: 逐行读取，不需要一次性加载全部数据
2. **易于追加**: 可以直接append新数据到文件末尾
3. **容错性强**: 某一行损坏不影响其他行
4. **适合流处理**: 配合生成器可实现数据流处理

---

### 2. 数据集切分策略（重要！）

**三种数据集的作用**:

| 数据集 | 比例 | 作用 | 是否更新模型参数 |
|--------|------|------|-----------------|
| **训练集 (Train)** | 80-90% | 训练模型，学习参数 | ✅ 是 |
| **验证集 (Validation)** | 10-20% | 调参、早停、模型选择 | ❌ 否 |
| **测试集 (Test)** | 隐藏 | 最终评估（比赛提交） | ❌ 否 |

**为什么需要验证集？**
- **问题**: 如果只用训练集训练，用测试集评估，容易过拟合
- **解决**: 验证集作为"模拟测试集"，帮助调整超参数
- **原则**: 测试集只在最后使用一次，防止"信息泄露"

**分层采样（Stratified Sampling）**:
```python
# 普通随机切分（可能导致标签分布不一致）
train_df, val_df = train_test_split(df, test_size=0.1)

# 分层采样（保持标签比例）
train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    stratify=df['label']  # 关键！保持训练集和验证集的标签分布一致
)
```

**为什么要分层采样？**
- 确保训练集和验证集的数据分布一致
- 防止验证集过于简单或困难
- 对于不平衡数据集尤其重要

---

### 3. 类别不平衡问题（Class Imbalance）

**什么是类别不平衡？**
- 本数据集: label 0 占 69.1%, label 1 占 30.9%
- 不平衡比例: 2.23:1

**为什么会有问题？**
```
假设模型总是预测label 0（不相似）：
准确率 = 69.1%（看起来还不错？）

但实际上：
- label 1 的召回率 = 0%（完全失败！）
- 模型没有学到任何有用的模式
```

**解决方法详解**:

**方法1: 调整类别权重（Class Weight）**
```python
# 计算类别权重
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(labels),
    y=labels
)

# 在损失函数中使用
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
```
- **原理**: 给少数类更大的权重，增加其损失值
- **优点**: 简单易用，不改变数据
- **缺点**: 可能导致过拟合少数类

**方法2: Focal Loss**
```python
# Focal Loss: 专注于难分类的样本
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 难度权重
```
- **原理**: 降低易分类样本的权重，专注于难样本
- **优点**: 同时处理不平衡和难样本
- **应用**: 目标检测、不平衡分类

**方法3: 重采样（Resampling）**
- **过采样（Over-sampling）**: 复制少数类样本
- **欠采样（Under-sampling）**: 减少多数类样本
- **SMOTE**: 合成新的少数类样本

---

### 4. 探索性数据分析（EDA）的意义

**EDA是什么？**
- Exploratory Data Analysis：在建模前对数据的全面探索
- 目标：理解数据特征、发现问题、指导建模

**为什么EDA很重要？**
> "Garbage In, Garbage Out" - 数据质量决定模型上限

**EDA的核心步骤**:

1. **数据概览**
   - 样本数量、特征数量
   - 缺失值、重复值
   - 数据类型

2. **统计分析**
   - 数值特征：均值、中位数、分位数
   - 文本特征：长度分布、词频统计

3. **可视化**
   - 分布图（直方图、密度图）
   - 关系图（散点图、热力图）

4. **异常检测**
   - 离群值（Outliers）
   - 数据质量问题

**本项目EDA的关键发现**（实际运行结果）:
1. **文本极短**: 95%样本总长度≤46字符，平均仅13.4字符 → **max_length=64** (不是128！)
2. **类别严重不平衡**: 2.23:1 (label 0: 69.1%, label 1: 30.9%) → 必须使用类别权重或Focal Loss
3. **长度无区分性**: 两个标签的文本长度分布完全重叠 → 必须用深度模型学习语义
4. **数据质量优秀**: 无缺失值、无重复、无异常 → 可直接用于训练

---

### 5. 文本长度分析与模型设计

**为什么要分析文本长度？**
- **max_length参数**: 决定模型能处理的最大输入长度
- **计算效率**: 更长的序列 = 更多计算 = 更慢的训练
- **显存占用**: 序列长度影响显存需求

**如何选择max_length？**
```python
# 查看分位数
df['total_len'].quantile([0.90, 0.95, 0.99])

# 结果示例：
# 90% -> 100 字符
# 95% -> 120 字符  ← 选择这个
# 99% -> 180 字符
```

**权衡**:
- **太小**: 截断重要信息，影响准确率
- **太大**: 浪费计算资源，显存不够
- **建议**: 覆盖95-99%的样本

**Padding与Truncation**:
```python
# BERT的输入处理
tokenizer(
    text,
    max_length=128,      # 最大长度
    padding='max_length', # 不足则填充
    truncation=True      # 超出则截断
)
```

---

## 🛠️ 实用技巧与方法论

### 技巧1: 数据加载优化

**使用生成器节省内存**:
```python
def load_jsonl_generator(file_path):
    """逐行生成数据，不一次性加载全部"""
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line.strip())

# 使用
for item in load_jsonl_generator('train.jsonl'):
    process(item)
```

---

## 🎓 面试常见问题

**Q1: 如何处理类别不平衡？**
A: 三种方法：
1. 重采样（过采样/欠采样）
2. 类别权重（调整损失函数）
3. 特殊损失函数（Focal Loss）

**Q2: 为什么要划分验证集？**
A:
- 避免在测试集上调参（信息泄露）
- 监控过拟合
- 早停（Early Stopping）

**Q3: 如何选择max_length？**
A:
- 分析文本长度分布
- 选择95-99分位数
- 权衡准确率和计算效率

**Q4: JSONL vs JSON的区别？**
A:
- JSONL可逐行读取，内存友好
- JSON需要完整解析
- JSONL适合大规模数据

---
### 阶段二预告：MacBERT Baseline

完成环境安装后，我们将开始：
1. 理解 Transformer 架构基础
2. 学习 BERT/MacBERT 工作原理
3. 实现文本对匹配模型
4. 掌握显存优化技巧（AMP、梯度累积）

**预期目标**: 准确率 78-83%

---

## 🎯 对阶段二的具体指导（基于EDA发现）

### 📝 完整的模型配置建议（RTX 4060 8GB）

```python
# ============================================
# MacBERT 训练配置（基于实际数据特征优化）
# ============================================

# 模型选择
MODEL_NAME = 'hfl/chinese-macbert-base'

# ========== 数据处理配置 ==========
MAX_LENGTH = 64  # ❗关键修正：实际95%分位=46，64已足够
# 原计划128太大，浪费显存和计算

BATCH_SIZE = 32  # 因为序列短，可以用更大的batch
VAL_RATIO = 0.1
RANDOM_SEED = 42

# ========== 显存优化配置 ==========
USE_AMP = True            # 自动混合精度训练 (FP16)
GRADIENT_ACCUMULATION = 2  # 梯度累积步数
# 实际有效batch = 32 * 2 = 64

# ========== 类别不平衡处理 ==========
# 方案1：类别权重（阶段二使用）
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(
    'balanced',
    classes=np.array([0, 1]),
    y=train_labels  # shape: (28800,)
)
# 预期结果：[0.72, 1.62]

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights, dtype=torch.float32)
)

# ========== 训练超参数 ==========
LEARNING_RATE = 2e-5  # MacBERT推荐学习率
NUM_EPOCHS = 3        # 预训练模型通常3-5轮即可
WARMUP_RATIO = 0.1    # 10%的训练步数用于warmup

# ========== 评估指标（不只看准确率！） ==========
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

metrics = {
    'accuracy': accuracy_score,
    'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
    'f1_weighted': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
    'precision_macro': lambda y_true, y_pred: precision_score(y_true, y_pred, average='macro'),
    'recall_macro': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro'),
}

# 对于不平衡数据，f1_macro 和 f1_weighted 比 accuracy 更重要！
```

---

### 💡 为什么这样配置？（参数选择的依据）

| 参数 | 推荐值 | 依据 | 如果用错了会怎样？ |
|------|--------|------|------------------|
| **max_length** | **64** | 95%分位=46，64足够覆盖 | 128会浪费40%显存和计算 |
| **batch_size** | **32** | 序列短，8GB显存足够 | 16太小，训练慢；64可能OOM |
| **class_weight** | **必须** | label比例2.23:1 | 不用会导致模型偏向label 0 |
| **AMP (FP16)** | **必须** | 节省显存和加速 | 不用会OOM或训练慢 |
| **gradient_accumulation** | **2** | 有效batch=64，足够大 | 太大(>4)会慢，太小(<2)效果差 |
| **评估指标** | **f1_macro** | 不平衡数据不能只看acc | 只看accuracy会被误导 |

---

### 🎯 三个优先级建议

#### P0（必须做）- 不做会严重影响性能
```python
1. ✅ max_length = 64  (不是128)
2. ✅ 使用 class_weight 处理不平衡
3. ✅ 评估指标：f1_macro > accuracy
```

#### P1（强烈推荐）- 能显著提升效率
```python
1. ✅ 使用 AMP (自动混合精度)
2. ✅ gradient_accumulation = 2
3. ✅ batch_size = 32 (因为序列短)
```

#### P2（锦上添花）- 阶段三再尝试
```python
1. Focal Loss 替代 CrossEntropyLoss
2. 对抗训练 (FGM)
3. 学习率 warmup + linear decay
```

---

### 📊 预期性能指标（阶段二目标）

基于以上配置，预期在验证集上：

```
✅ 最低目标（Baseline）:
   - Accuracy:    75-78%
   - F1-macro:    70-73%

✅ 良好目标（优化后）:
   - Accuracy:    80-83%
   - F1-macro:    76-79%

✅ 优秀目标（阶段三）:
   - Accuracy:    85-88%
   - F1-macro:    82-85%
```

**关键提醒**：
- 如果只看 accuracy，可能被高估（因为不平衡）
- **F1-macro** 是更可靠的指标（两个类别平等对待）
- label 1 的召回率 > 70% 才算合格

---

### 🔧 预计训练时间与资源占用

**硬件**: RTX 4060 (8GB)

```
配置方案A（推荐）：
  - max_length=64, batch_size=32, AMP=True
  - 单epoch训练时间: ~8-10分钟
  - 显存占用: ~4-5 GB
  - 3个epoch总时间: ~30分钟

配置方案B（如果显存不够）：
  - max_length=64, batch_size=16, gradient_accumulation=4
  - 单epoch训练时间: ~12-15分钟
  - 显存占用: ~3-4 GB
  - 3个epoch总时间: ~40分钟

配置方案C（错误示范 - 不要用）：
  - max_length=128, batch_size=16, AMP=False
  - 单epoch训练时间: ~20-25分钟
  - 显存占用: ~6-7 GB (可能OOM)
  - 训练慢且浪费资源
```

---

### 📋 阶段二检查清单

在开始编写代码前，确认以下事项：

- [ ] 理解了为什么 max_length=64 而不是 128
- [ ] 理解了类别不平衡的危害和解决方法
- [ ] 理解了为什么不能只看 accuracy
- [ ] 知道如何计算 class_weight
- [ ] 了解 AMP 和梯度累积的作用
- [ ] 准备好使用 f1_macro 作为主要评估指标

---

## 💡 关键要点回顾

1. ✅ **EDA是建模的基础** - 理解数据才能设计好模型
2. ✅ **文本极短，影响参数选择** - 实际95%分位=46字符 → **max_length=64** (不是128)
3. ✅ **类别严重不平衡** - 2.23:1，必须用class_weight，不能只看accuracy
4. ✅ **长度无区分性** - 证实必须用深度模型学习语义，不能用简单规则
5. ✅ **分层采样保证分布一致** - 训练集和验证集的标签比例相同
6. ✅ **规范的项目结构是好习惯** - 可维护、可协作、符合工业标准
7. ✅ **评估指标选择** - F1-macro 比 accuracy 更可靠（针对不平衡数据）

---

准备好后，告诉我你已完成环境安装，我们将进入**阶段二：MacBERT Baseline模型**！🚀
