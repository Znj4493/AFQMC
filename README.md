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
├── dataset/              # 数据集
├── src/                  # 源代码
│   ├── data_loader.py   # 数据加载模块
│   ├── model_macbert.py # MacBERT模型（待开发）
│   └── model_llm.py     # LLM模型（待开发）
├── config/               # 配置文件
├── checkpoints/          # 模型保存
├── results/              # 结果输出
└── notebooks/            # 数据探索
    └── eda.py           # 探索性数据分析

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

### 3. 训练模型（开发中）

```bash
# MacBERT baseline（阶段二）
python src/train.py

# LoRA微调（阶段四）
python src/train_lora.py
```

## 当前进度

- [x] 阶段一：环境搭建与数据探索
- [ ] 阶段二：MacBERT Baseline模型
- [ ] 阶段三：模型优化与进阶技巧
- [ ] 阶段四：LLM微调（LoRA）
- [ ] 阶段五：推理与提交
- [ ] 阶段六：总结与面试准备

## 数据集信息

- **训练集**: 32,000条
- **测试集**: 5,000条
- **标签分布**: 不平衡（69.1% label=0, 30.9% label=1）
- **领域**: 蚂蚁金服金融产品问答

## 学习资源

详见项目计划文档: [.claude/project_plan.md](.claude/project_plan.md)
