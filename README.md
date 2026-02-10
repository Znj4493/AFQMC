# AFQMC 蚂蚁金服语义相似度匹配项目

## 项目简介

本项目是参加"蚂蚁金服语义相似度匹配（AFQMC）"比赛的完整实现，采用了**MacBERT + LoRA微调大模型**的双技术路线。

## 技术栈

- **基础框架**: PyTorch 2.0+
- **Baseline模型**: hfl/chinese-macbert-base
- **LLM微调**: Qwen-1.5 + LoRA (PEFT)
- **显存优化**: AMP、梯度累积、4-bit量化

## 项目结构

```
AFQMC/
├── dataset/                    # 数据集
├── src/                        # 源代码
│   ├── models/                 # 模型定义
│   │   ├── macbert.py         # MacBERT模型
│   │   ├── llm.py             # LLM模型
│   │   └── lora_config.py     # LoRA配置
│   ├── data/                   # 数据加载
│   │   ├── data_loader.py     # BERT数据加载
│   │   ├── data_loader_llm.py # LLM数据加载
│   │   └── data_augmentation.py # 数据增强
│   ├── training/               # 训练相关
│   │   ├── bert_train.py      # BERT训练脚本
│   │   ├── train_lora.py      # LoRA训练脚本
│   │   ├── loss.py            # 损失函数
│   │   └── adversarial.py     # 对抗训练
│   ├── inference/              # 推理相关
│   │   ├── inference_macbert.py # MacBERT推理
│   │   └── inference_api.py   # API推理
│   └── utils/                  # 工具函数
│       └── utils.py           # 通用工具
├── scripts/                    # 独立脚本
│   ├── download_models.py     # 下载模型
│   ├── calibrate_threshold.py # 阈值校准
│   ├── compare_results.py     # 结果对比
│   ├── analyze_disagreements.py # 分歧分析
│   └── generate_report.py     # 生成报告
├── config/                     # 配置文件
├── checkpoints/                # 模型保存
├── results/                    # 结果输出
└── notebooks/                  # 数据探索
    └── eda.py                 # 探索性数据分析
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据探索

```bash
cd notebooks
python eda.py
```

### 3. 训练模型

```bash
# MacBERT baseline
cd src/training
python bert_train.py

# LoRA微调
cd src/training
python train_lora.py
```

### 4. 推理

```bash
# MacBERT推理
cd src/inference
python inference_macbert.py

# API推理
cd src/inference
python inference_api.py
```

## 数据集信息

- **训练集**: 32,000条
- **测试集**: 5,000条
- **标签分布**: 不平衡（69.1% label=0, 30.9% label=1）
- **领域**: 蚂蚁金服金融产品问答

## 学习资源

详见项目计划文档: [.claude/project_plan.md](.claude/project_plan.md)