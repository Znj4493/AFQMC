"""
测试 Qwen-1.8B 模型加载和 4-bit 量化
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import os

def test_qwen_loading():
    print("=" * 60)
    print("测试 Qwen-1.8B-Chat 模型加载")
    print("=" * 60)

    # 0. 设置本地模型路径
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "Qwen1.5-1.8B-Chat")
    print(f"\n[0/5] 本地模型路径: {model_path}")

    if not os.path.exists(model_path):
        print(f"✗ 模型路径不存在: {model_path}")
        return False
    print("✓ 模型路径存在")

    # 1. 配置 4-bit 量化
    print("\n[1/5] 配置 4-bit 量化...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,                      # 启用 4-bit 量化
        bnb_4bit_quant_type="nf4",              # 使用 NF4 量化类型
        bnb_4bit_compute_dtype=torch.float16,   # 计算时使用 FP16
        bnb_4bit_use_double_quant=True,         # 双重量化（进一步节省显存）
    )
    print("✓ 量化配置完成")
    print(f"   - 量化类型: NF4 (4-bit)")
    print(f"   - 计算精度: FP16")
    print(f"   - 双重量化: 开启")

    # 2. 加载模型
    print(f"\n[2/5] 从本地加载模型...")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",                   # 自动分配设备
            trust_remote_code=True,              # Qwen 需要信任远程代码
        )
        print("✓ 模型加载成功")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. 加载 Tokenizer
    print("\n[3/5] 加载 Tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        print("✓ Tokenizer 加载成功")
    except Exception as e:
        print(f"✗ Tokenizer 加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 显存占用情况
    print("\n[4/5] 显存占用情况:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   已分配显存: {allocated:.2f} GB")
        print(f"   已预留显存: {reserved:.2f} GB")
        print(f"   显存总量:   {total:.2f} GB")
        print(f"   剩余显存:   {total - reserved:.2f} GB")

        if reserved > 7.5:
            print("   ⚠️  显存占用接近上限，建议降低 batch_size")
        else:
            print("   ✓ 显存占用健康")

    # 5. 简单推理测试
    print("\n[5/5] 简单推理测试")
    print("=" * 60)

    test_prompt = "判断以下两个句子是否语义相似：\n句子1：花呗如何还款\n句子2：花呗怎么还钱\n回答："

    print(f"\n输入提示词:\n{test_prompt}")

    try:
        inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

        print("\n生成中...")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                top_p=0.8,
                repetition_penalty=1.1,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 只显示模型生成的部分（去掉输入的 prompt）
        generated_text = response[len(test_prompt):].strip()

        print(f"\n模型生成:\n{generated_text}")
        print("\n✓ 推理测试成功")
    except Exception as e:
        print(f"✗ 推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！")
    print("=" * 60)
    print("\n关键指标:")
    print(f"  - 模型加载: 成功 (4-bit 量化)")
    print(f"  - 显存占用: {reserved:.2f} GB / {total:.2f} GB")
    print(f"  - 推理功能: 正常")

    return True

if __name__ == "__main__":
    success = test_qwen_loading()
    exit(0 if success else 1)
