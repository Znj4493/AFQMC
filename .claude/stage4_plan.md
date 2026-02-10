# 阶段四：大语言模型（LLM）微调 - LoRA实战

**开始时间**：2026年2月6日

---

## 🎯 目标

- 掌握 LoRA 高效微调技术
- 使用 Qwen-1.5-1.8B 完成语义匹配任务
- 对比 LLM 与 MacBERT 的效果差异
- 预期性能：F1-Macro 85-90%

---

## 📋 技术选型

- **模型**：Qwen-1.5-1.8B
- **微调方法**：LoRA (rank=8-16)
- **量化方案**：4-bit (QLoRA)
- **输入格式**：带思维链的 Prompt
- **显存优化**：量化 + 梯度累积 + 梯度检查点

---

## 🚀 实施路线图

### 阶段一：环境准备与依赖安装 [20分钟]
- [ ] 1. 更新 requirements.txt（添加 PEFT、bitsandbytes）
- [ ] 2. 安装依赖并验证
- [ ] 3. 测试 Qwen-1.8B 模型下载和加载

### 阶段二：数据准备 [30分钟]
- [ ] 1. 创建 data_loader_llm.py（带思维链的 prompt 模板）
- [ ] 2. 创建 lora_config.py（配置参数）
- [ ] 3. 测试数据格式输出

### 阶段三：模型搭建 [40分钟]
- [ ] 1. 创建 model_llm.py
  - 4-bit 量化加载
  - LoRA 配置注入
  - 梯度检查点设置
- [ ] 2. 测试模型加载和显存占用

### 阶段四：训练流程 [50分钟]
- [ ] 1. 创建 train_lora.py
  - 训练循环主体
  - AMP + 梯度累积
  - 验证和保存逻辑
- [ ] 2. 首次训练测试（1个epoch）

### 阶段五：对比实验 [后续]
- [ ] 1. 完整训练（多个epoch）
- [ ] 2. 创建 compare_models.py
- [ ] 3. 性能对比分析
- [ ] 4. 撰写实验报告

---

## 🎯 成功标准

- Qwen-1.8B + LoRA 模型训练成功
- F1-Macro ≥ 85%（相比 MacBERT 的 71.37%）
- 显存占用 < 8GB
- 完成 LLM vs BERT 对比分析

---

## 📦 产出文件

```
src/
├── model_llm.py          # LLM模型定义
├── train_lora.py         # LoRA微调脚本
├── data_loader_llm.py    # LLM数据加载器
└── compare_models.py     # 模型对比脚本

config/
└── lora_config.py        # LoRA配置参数

checkpoints/
└── qwen-lora/            # 模型保存目录

results/
└── stage4_comparison.md  # LLM vs BERT对比报告
```
