# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the **AFQMC (蚂蚁金服语义相似度)** project - a Chinese semantic text similarity matching competition using MacBERT fine-tuning. The task is binary classification: determine if two financial product Q&A sentences are semantically similar.

**Key Constraints:**
- **Data imbalance**: 69.1% label=0 (not similar), 30.9% label=1 (similar) - ratio 2.23:1
- **Hardware**: RTX 4060 8GB VRAM (requires AMP + gradient accumulation)
- **Short text**: 95th percentile = 46 tokens, so MAX_LENGTH=64 (not 128)
- **Primary metric**: F1-macro (not Accuracy, due to class imbalance)

## Environment Setup

```bash
# Create conda environment
conda create -n afqmc python=3.10
conda activate afqmc

# Install dependencies
pip install -r requirements.txt

# Download pretrained model (if not already cached)
python src/download_models.py
```

**Note**: The project expects MacBERT model at `checkpoints/chinese-macbert-base/`. If using HuggingFace, it will auto-download to this location.

## Training Commands

### MacBERT Baseline (Stage 2)
```bash
# Full training with default config
python src/train.py

# Training resumes automatically if checkpoint exists in:
# checkpoints/macbert/best_model.pt
```

**Expected Performance (Stage 2 target):**
- Accuracy: 74%
- F1-macro: 70%
- Training time: ~3-4 minutes for 5 epochs on RTX 4060

### Data Exploration
```bash
# Run EDA analysis
python notebooks/eda.py

# Interactive notebook (if needed)
jupyter notebook notebooks/eda_analysis.ipynb
```

### Testing Report Format
```bash
# Test the enhanced evaluation report output
python test_report.py
```

## Core Architecture

### Data Pipeline Flow
```
JSONL files → data_loader.py → split_train_val() → AFQMCDataset → DataLoader → train_epoch()
                ↓                                        ↓
                pandas DataFrame                     Tokenization (on-the-fly)
```

**Key Design Decisions:**
1. **Lazy tokenization**: Tokenization happens in `AFQMCDataset.__getitem__()`, not during data loading
2. **Stratified split**: `split_train_val()` maintains class ratio in train/val split
3. **Class weights**: Auto-calculated via `compute_class_weights()` to handle imbalance

### Training Loop Architecture

**Single Epoch Flow:**
```python
train_epoch():
    for batch in train_loader:
        # 1. Forward pass with AMP
        with torch.cuda.amp.autocast():
            outputs = model(input_ids, attention_mask, labels)
            loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS  # Important: divide by accumulation steps

        # 2. Backward pass
        scaler.scale(loss).backward()

        # 3. Update every N batches (gradient accumulation)
        if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
```

**Key Components:**
- `train_epoch()`: Training loop with AMP + gradient accumulation
- `evaluate()`: Validation with `torch.no_grad()`, returns loss + metrics + predictions
- `compute_metrics()`: Calculates F1-macro, accuracy, confusion matrix (in `utils.py`)
- `save_checkpoint()` / `load_checkpoint()`: Model persistence with optimizer state

### Model Architecture
```
Input: [CLS] sentence1 [SEP] sentence2 [SEP] [PAD]...
    ↓
MacBERT Encoder (12 layers, 768 hidden)
    ↓
Take [CLS] token output → Dropout → Linear(768 → 2)
    ↓
Logits (2 classes) → CrossEntropyLoss (with class weights)
```

## Configuration System (`config/config.py`)

**Critical Parameters (tuned based on EDA):**
- `MAX_LENGTH = 64` (not 128 - based on 95th percentile analysis)
- `BATCH_SIZE = 32`
- `GRADIENT_ACCUMULATION_STEPS = 2` (effective batch = 64)
- `USE_AMP = True` (required for 8GB VRAM)
- `USE_CLASS_WEIGHT = True` (handles 2.23:1 imbalance)
- `METRIC_FOR_BEST_MODEL = 'f1_macro'` (not accuracy)
- `EARLY_STOPPING_PATIENCE = 3` (stop if no improvement)

**To modify training:**
1. Edit `config/config.py` directly (centralized configuration)
2. No need to pass command-line args - all config is in this file

## Memory Optimization Strategy

**Problem**: MacBERT-base (102M params) + batch_size=64 exceeds 8GB VRAM

**Solution (implemented in `train.py`):**
1. **AMP (Automatic Mixed Precision)**: FP16 forward pass, saves ~40% memory
2. **Gradient Accumulation**: Process 2 mini-batches (32 each) before updating, simulates batch_size=64
3. **Short sequences**: MAX_LENGTH=64 instead of 128 (saves 50% sequence memory)

**Memory breakdown:**
- Model: ~400MB (FP32 params)
- Gradients: ~400MB
- Optimizer (AdamW): ~800MB
- Activations (batch=32, len=64, FP16): ~750MB
- **Total: ~2.3GB** (fits in 8GB with headroom)

## Key Technical Decisions

### Why MacBERT (not BERT)?
- **MacBERT** uses **similar word replacement** instead of `[MASK]` during pretraining
- Reduces pretrain-finetune gap (no `[MASK]` in real text)
- **Whole Word Masking (WWM)** for Chinese (masks entire words, not characters)
- ~0.5-1.5% better F1 on Chinese NLP tasks vs original BERT

### Why F1-macro (not Accuracy)?
- Class imbalance (69:31 ratio)
- Accuracy can be 69% by always predicting class 0
- F1-macro gives equal weight to both classes: `(F1_class0 + F1_class1) / 2`
- Better reflects model's ability on minority class (similar sentences)

### Why Gradient Accumulation?
```python
# Mathematically equivalent to batch_size=64, but uses memory of batch_size=32
loss = loss / GRADIENT_ACCUMULATION_STEPS  # Key: divide loss to match gradient scale
```

## File Structure

```
src/
├── train.py              # Main training script (contains AFQMCDataset, train loop)
├── data_loader.py        # JSONL loading, train/val split
├── model_macbert.py      # Model initialization (create_model, create_tokenizer)
├── utils.py              # Metrics, checkpointing, random seed
└── download_models.py    # Download pretrained weights

config/
└── config.py             # All hyperparameters (centralized)

checkpoints/
├── chinese-macbert-base/ # Pretrained MacBERT weights
└── macbert/              # Training checkpoints (best_model.pt)

.claude/
├── CLAUDE.md             # This file
├── Stage2_MacBERT_Baseline_Review.md  # Stage 2 technical review
└── project_plan.md       # Project roadmap
```

## Common Issues & Solutions

### Issue: `RuntimeError: CUDA out of memory`
**Solution**: Already optimized for 8GB. If still failing:
1. Reduce `BATCH_SIZE` from 32 to 16
2. Increase `GRADIENT_ACCUMULATION_STEPS` from 2 to 4

### Issue: Loss not decreasing
**Check:**
1. Labels are `torch.long` (not strings)
2. Learning rate is reasonable (default `2e-5`)
3. Class weights are loaded correctly (should be `[0.72, 1.62]`)

### Issue: Model predicts all class 0
**Solution**: Increase class weight for class 1 in `compute_class_weights()` or manually set `class_weights = torch.tensor([0.5, 2.5])`

### Issue: `weights_only=True` error when loading checkpoint
**Fix already applied**: `torch.load(..., weights_only=False)` in `utils.py:430`
- This is safe for self-trained models (PyTorch 2.6+ security update)

## Current Project Status

**Stage 2 (MacBERT Baseline) - COMPLETED:**
- ✅ Training pipeline implemented with AMP + gradient accumulation
- ✅ Class imbalance handling (class weights)
- ✅ Early Stopping (patience=3)
- ✅ Enhanced evaluation report with confusion matrix analysis
- ✅ Current performance: 74.12% Accuracy, 70.19% F1-macro

**Known Issues:**
- Class 1 (similar) F1=59.37%, significantly lower than Class 0 F1=81.02%
- Overfitting after epoch 3 (val_loss increases while train_loss decreases)
- 6-9 percentage points gap to Stage 2 target (76-79% F1-macro)

**Next Steps (Stage 3 - Adversarial Training):**
- Implement FGM (Fast Gradient Method) to improve generalization
- Address overfitting and improve minority class performance
- Target: 74-76% F1-macro

## Important Notes for Future Claude Instances

1. **Do NOT change MAX_LENGTH to 128**: It's 64 based on EDA data analysis (95th percentile=46)
2. **Always use F1-macro** for evaluation, not accuracy (class imbalance)
3. **Loss must be divided** by `GRADIENT_ACCUMULATION_STEPS` in training loop
4. **Class weights are auto-calculated** from training data in `main()` function
5. **Tokenization is lazy** (in `Dataset.__getitem__`), not pre-computed
6. **Model input format**: `[CLS] text1 [SEP] text2 [SEP]` (sentence pair)
7. **Training resumes automatically** if `best_model.pt` exists in checkpoint directory
8. **Windows compatibility**: `num_workers=0` in DataLoader (non-zero causes hanging)

## Performance Expectations

**Stage 2 Baseline (current):**
- Train time: ~3-4 minutes (5 epochs, RTX 4060)
- Memory usage: ~2.3GB / 8GB VRAM (28.8%)
- Speed: ~10.5 it/s with AMP
- F1-macro: 70.19% (target: 76-79%)

**Metrics to monitor:**
- Primary: `f1_macro` (saved in checkpoint)
- Secondary: `accuracy`, `f1_weighted`
- Diagnostic: Confusion matrix (FP vs FN analysis)
