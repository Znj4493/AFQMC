"""
配置文件 - MacBERT训练配置

基于阶段一EDA的关键发现优化:
1. 文本极短(95%分位=46) → max_length=64
2. 类别不平衡(2.23:1) → 使用class_weight
3. 硬件约束(RTX 4060 8GB) → AMP + 梯度累积

我们要把这本习题集学 3 遍 (Epochs)。
每次只拿 32 题 (Batch Size) 放在桌上，但我们要攒两波做完 64 题 (梯度累积) 再总结一次。
刚开始 10% 的时间 (Warmup) 慢慢学，后面按 0.00002 的步长 (Learning Rate) 稳步前进。
每句话最长只看 64 个字 (Max Length)。
最后的考试评分标准是 F1-Macro，谁偏科谁分低！
"""

import os
import torch

# ============================================
# 路径配置
# ============================================
# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据路径
DATA_DIR = os.path.join(PROJECT_ROOT, 'dataset')
TRAIN_FILE = os.path.join(DATA_DIR, 'train.jsonl')
TEST_FILE = os.path.join(DATA_DIR, 'test.jsonl')

# 模型保存路径
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints', 'macbert')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 结果保存路径
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================
# 模型配置
# ============================================
# 预训练模型名称
MODEL_NAME = os.path.join(PROJECT_ROOT, 'checkpoints', 'chinese-macbert-base')
# 如果下载慢,可以使用本地路径或国内镜像
# MODEL_NAME = './pretrained_models/chinese-macbert-base'

# ============================================
# 数据处理配置
# ============================================
# ❗关键参数: 基于EDA发现,95%分位=46,设置64足够
# 原计划128会浪费40%显存和计算
MAX_LENGTH = 64

# 验证集比例
VAL_RATIO = 0.1

# 随机种子(保证可复现)
RANDOM_SEED = 42

# ============================================
# 训练超参数配置
# ============================================
# 批次大小(因为序列短,可以用更大的batch)
BATCH_SIZE = 32

# 学习率(MacBERT推荐范围: 1e-5 到 5e-5)
LEARNING_RATE = 2e-5

# 训练轮数(预训练模型通常3-5轮即可,过多会过拟合)
NUM_EPOCHS = 5

# Warmup比例(前10%的训练步数用于学习率预热)
WARMUP_RATIO = 0.1

# 权重衰减(L2正则化)
WEIGHT_DECAY = 0.01

# 最大梯度裁剪(防止梯度爆炸)
MAX_GRAD_NORM = 1.0

# ============================================
# 显存优化配置(针对RTX 4060 8GB)
# ============================================
# 自动混合精度训练(FP16,节省显存并加速)
USE_AMP = True

# 梯度累积步数
# 实际有效batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
# 例如: 32 * 2 = 64
GRADIENT_ACCUMULATION_STEPS = 2

# ============================================
# 类别不平衡处理
# ============================================
# 是否使用类别权重(强烈推荐,本数据集2.23:1不平衡)
USE_CLASS_WEIGHT = True

# 权重将在训练脚本中根据实际标签分布自动计算
# class_weights = compute_class_weight('balanced', classes=[0, 1], y=train_labels)

# ============================================
# 对抗训练配置（阶段三）
# ============================================
# 是否使用 FGM 对抗训练
USE_ADVERSARIAL = True

# 对抗扰动强度（epsilon）
# - 越大，对抗性越强，但可能影响正常样本学习
# - 推荐范围：0.5 ~ 2.0
# - 默认值：1.0
ADV_EPSILON = 1.0

# ============================================
# 评估与保存配置
# ============================================
# 多少步评估一次验证集
# 设置为None则每个epoch评估一次
EVAL_STEPS = None

# 多少步保存一次checkpoint
# 设置为None则每个epoch保存一次
SAVE_STEPS = None

# 最多保存几个checkpoint(节省磁盘空间)
SAVE_TOTAL_LIMIT = 3

# 根据什么指标保存最佳模型
# 选项: 'loss', 'accuracy', 'f1_macro', 'f1_weighted'
# 对于不平衡数据,推荐使用 f1_macro
METRIC_FOR_BEST_MODEL = 'f1_macro'

# 是否越大越好(loss是False,其他是True)
GREATER_IS_BETTER = True

# 早停参数(连续多少次评估没有提升就停止训练)
EARLY_STOPPING_PATIENCE = 3

# ============================================
# 设备配置
# ============================================
# 自动检测GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 是否使用多GPU
USE_MULTI_GPU = False

# ============================================
# 日志配置
# ============================================
# 多少步打印一次日志
LOGGING_STEPS = 50

# 是否使用tensorboard
USE_TENSORBOARD = False

# ============================================
# 其他配置
# ============================================
# DataLoader的线程数
NUM_WORKERS = 0  # Windows上建议设为0,避免多进程问题

# 是否使用快速tokenizer
USE_FAST_TOKENIZER = True


# ============================================
# 配置验证与打印
# ============================================
def print_config():
    """打印关键配置信息"""
    print("=" * 60)
    print("MacBERT训练配置")
    print("=" * 60)
    print(f"模型: {MODEL_NAME}")
    print(f"设备: {DEVICE}")
    print(f"最大序列长度: {MAX_LENGTH}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"有效批次大小: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"学习率: {LEARNING_RATE}")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"使用AMP: {USE_AMP}")
    print(f"使用类别权重: {USE_CLASS_WEIGHT}")
    print(f"评估指标: {METRIC_FOR_BEST_MODEL}")
    print("=" * 60)


if __name__ == '__main__':
    # 测试配置
    print_config()
    print(f"\n训练数据路径: {TRAIN_FILE}")
    print(f"测试数据路径: {TEST_FILE}")
    print(f"Checkpoint保存路径: {CHECKPOINT_DIR}")
    print(f"结果保存路径: {RESULTS_DIR}")
