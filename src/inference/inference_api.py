import json
import re
import random
import threading
import time
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from tqdm import tqdm

# 修复 Windows 终端中文编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ==================== 平台配置 ====================
PLATFORMS = {
    "nvidia": {
        "type": "openai",  # 使用 OpenAI 兼容接口
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key": "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",
    },
    "zhipu": {
        "type": "zhipu",  # 使用智谱原生 SDK
        "api_key": "5672204330d24299977d56cf8edb1f2f.mq8J3aAgMRswVzXu",
    },
    "zhipu_zai": {
        "type": "zai",  # 使用 zai SDK (glm-4.6 等新模型)
        "api_key": "5672204330d24299977d56cf8edb1f2f.mq8J3aAgMRswVzXu",
    },
    "custom": {
        "type": "openai",  # 使用 OpenAI 兼容接口
        "base_url": "http://106.53.8.125:21207/v1",
        "api_key": "sk-1180c2b72d4b17c9885ab6ec7df404a2",  # 请替换为你的实际 API Key
    },
    # 新增平台只需在这里添加
}

# ==================== 模型配置 ====================
MODEL_CONFIGS = {
    # --- NVIDIA 平台 ---
    "qwen3": {
        "platform": "nvidia",
        "model_id": "qwen/qwen3-next-80b-a3b-instruct",
        "max_tokens": 512,
        "extra_body": None,
        "max_workers": 5,
    },
    "glm4.7": {
        "platform": "nvidia",
        "model_id": "z-ai/glm4.7",
        "max_tokens": 512,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True}},
        "max_workers": 1,
    },
    "glm4.7-nothink": {
        "platform": "nvidia",
        "model_id": "z-ai/glm4.7",
        "max_tokens": 512,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        "max_workers": 1,
    },
    "deepseek-v3.2": {
        "platform": "nvidia",
        "model_id": "deepseek-ai/deepseek-v3.2",
        "max_tokens": 512,
        "extra_body": None,
        "max_workers": 5,
    },
    # --- 智谱平台 ---
    "glm4.7-flash": {
        "platform": "zhipu",
        "model_id": "glm-4.7-flash",
        "max_tokens": 4096,
        "thinking": False,  # 显式禁用 thinking
        "max_workers": 1,
    },
    "glm4.7-flash-think": {
        "platform": "zhipu",
        "model_id": "glm-4.7-flash",
        "max_tokens": 65536,
        "thinking": True,
        "max_workers": 1,
    },
    "glm4.6": {
        "platform": "zhipu_zai",
        "model_id": "glm-4.6",
        "max_tokens": 4096,
        "thinking": False,
        "max_workers": 3,
    },
    "glm4.6-think": {
        "platform": "zhipu_zai",
        "model_id": "glm-4.6",
        "max_tokens": 65536,
        "thinking": True,
        "max_workers": 3,
    },
    # --- Custom 平台 ---
    "gpt5.2": {
        "platform": "custom",
        "model_id": "gpt-5.2",
        "max_tokens": 4096,
        "extra_body": None,
        "max_workers": 10,  
        "params": {  # 自定义参数
            "temperature": 0.1,
            "top_p": 0.95,
            "reasoning_effort": "low",  
        },
    },
    # 新增模型只需在这里添加配置即可
}

# ==================== 客户端管理 ====================
_clients = {}

def get_client(platform_name):
    """获取或创建平台对应的客户端"""
    if platform_name in _clients:
        return _clients[platform_name]

    platform = PLATFORMS[platform_name]

    if platform["type"] == "openai":
        client = OpenAI(base_url=platform["base_url"], api_key=platform["api_key"])
    elif platform["type"] == "zhipu":
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key=platform["api_key"])
    elif platform["type"] == "zai":
        from zai import ZhipuAiClient
        client = ZhipuAiClient(api_key=platform["api_key"])
    else:
        raise ValueError(f"未知平台类型: {platform['type']}")

    _clients[platform_name] = client
    return client

# ==================== Prompt ====================
PROMPT_TEMPLATE = """你是一个中文语义相似度判断专家，专注于金融领域问答的语义理解。

【任务】判断两个句子是否表达相同或相近的意思。

【判断标准】
- 相似(1)：两句话询问同一个问题，或表达相同的意图，即使用词、句式不同也算相似
- 不相似(0)：询问不同的问题，或表达不同/相反的意图

【示例】
句子1：花呗可以用于购买黄金吗
句子2：在支付宝买黄金支持花呗付款吗
分析：两句都在问能否用花呗买黄金，意图相同
答案：1

句子1：如何在使用花呗购物后更新收货地址
句子2：移除花呗账户中已保存的收货地址
分析：一个问更新地址，一个问删除地址，意图不同
答案：0

句子1：借呗第二次借款的还款日期会改变吗
句子2：借呗每月还款日能改吗
分析：都在问还款日期是否可变，意图相同
答案：1

句子1：花呗过去可用于支付日常开销费用
句子2：生活缴费功能目前无法通过花呗完成
分析：一个说可以用，一个说不能用，意图相反
答案：0

【现在请判断】
句子1：{text1}
句子2：{text2}

请先简要分析两句话的核心意图，然后给出答案。
答案格式：只输出数字 1 或 0，不要有其他内容。"""


def judge_similarity(text1, text2, config, show_detail=False):
    """调用 API 判断两个文本是否语义相似"""
    client = get_client(config["platform"])
    platform = PLATFORMS[config["platform"]]
    prompt = PROMPT_TEMPLATE.format(text1=text1, text2=text2)

    # 根据平台类型构造请求参数
    kwargs = {
        "model": config["model_id"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config["max_tokens"],
    }

    # 添加默认参数
    default_params = {
        "temperature": 0.1,
    }

    # 如果模型配置中有自定义参数，使用自定义参数覆盖默认值
    if "params" in config:
        default_params.update(config["params"])

    # 将参数添加到 kwargs
    kwargs.update(default_params)

    if platform["type"] == "openai":
        kwargs["stream"] = False
        if config.get("extra_body"):
            kwargs["extra_body"] = config["extra_body"]
    elif platform["type"] == "zhipu":
        if config.get("thinking"):
            kwargs["thinking"] = {"type": "enabled"}
        else:
            kwargs["thinking"] = {"type": "disabled"}
    elif platform["type"] == "zai":
        if config.get("thinking"):
            kwargs["thinking"] = {"type": "enabled"}
        else:
            kwargs["thinking"] = {"type": "disabled"}

    response = client.chat.completions.create(**kwargs)

    msg = response.choices[0].message
    content = (msg.content or "").strip()
    reasoning = getattr(msg, 'reasoning_content', None) or ""

    if content:
        raw_output = content
        source = "content"
    else:
        raw_output = reasoning.strip()
        source = "reasoning"

    if show_detail:
        print(f"\n{'='*60}")
        print(f"句子1: {text1}")
        print(f"句子2: {text2}")
        if reasoning and content:
            print(f"思考过程（节选）:\n{reasoning[:200]}...")
        print(f"最终输出 [{source}]:\n{raw_output}")
        print(f"{'='*60}\n")

    # 提取最终答案
    result = "0"

    if source == "content":
        matches = re.findall(r'(?<!\d)[01](?!\d)', raw_output)
        if matches:
            result = matches[-1]
        else:
            if '不相似' in raw_output or '不一致' in raw_output or '不同' in raw_output:
                result = "0"
            elif '相似' in raw_output or '一致' in raw_output or '相同' in raw_output:
                result = "1"
    else:
        last_lines = raw_output.split('\n')[-3:]
        last_text = '\n'.join(last_lines)
        matches = re.findall(r'(?<!\d)[01](?!\d)', last_text)
        if matches:
            result = matches[-1]
        else:
            if '不相似' in raw_output or '不一致' in raw_output or '不同' in raw_output:
                result = "0"
            elif '相似' in raw_output or '一致' in raw_output or '相同' in raw_output:
                result = "1"

    return result, raw_output


def main():
    parser = argparse.ArgumentParser(description="AFQMC 语义相似度推理")
    parser.add_argument("--model", type=str, default="qwen3",
                        choices=list(MODEL_CONFIGS.keys()),
                        help=f"选择模型: {list(MODEL_CONFIGS.keys())}")
    parser.add_argument("--workers", type=int, default=None,
                        help="并发线程数（默认使用模型配置值）")
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]
    max_workers = args.workers or config["max_workers"]
    model_name = args.model
    output_path = os.path.join("results", f"result_{model_name}.jsonl")

    print(f"平台: {config['platform']}")
    print(f"模型: {config['model_id']} ({model_name})")
    print(f"并发数: {max_workers}")
    print(f"输出文件: {output_path}")

    test_data = []
    with open('dataset/test.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))

    sample_indices = set(random.sample(range(len(test_data)), min(50, len(test_data) // 100)))
    sample_indices.update({0, 1, 2})

    print(f"共 {len(test_data)} 条数据，随机显示 {len(sample_indices)} 条详细输出\n")

    results = ["0"] * len(test_data)
    completed = [0]
    errors = [0]
    lock = threading.Lock()
    pbar = tqdm(total=len(test_data), desc="推理进度", unit="条",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    def save_results():
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(json.dumps({"response": r}, ensure_ascii=False) + '\n')

    def process_item(idx, item):
        text1 = item['text1']
        text2 = item['text2']
        show_detail = idx in sample_indices

        try:
            result, _ = judge_similarity(text1, text2, config, show_detail)
            results[idx] = result
            with lock:
                completed[0] += 1
                pbar.update(1)
                if show_detail:
                    tqdm.write(f"[{completed[0]}/{len(test_data)}] #{idx+1} 结果: {result}")
                if completed[0] % 500 == 0:
                    save_results()
                    tqdm.write(f"💾 已自动存盘 ({completed[0]} 条)")

        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "too many" in error_msg.lower():
                for retry in range(3):
                    wait = (retry + 1) * 5
                    with lock:
                        tqdm.write(f"⏳ #{idx+1} 限流，{wait}秒后第{retry+1}次重试...")
                    time.sleep(wait)
                    try:
                        result, _ = judge_similarity(text1, text2, config, show_detail)
                        results[idx] = result
                        with lock:
                            completed[0] += 1
                            pbar.update(1)
                        return
                    except Exception:
                        continue

            with lock:
                completed[0] += 1
                errors[0] += 1
                pbar.update(1)
                tqdm.write(f"\n❌ 第 {idx+1} 条出错: {e}")
                tqdm.write(f"   句子1: {text1}")
                tqdm.write(f"   句子2: {text2}\n")
            results[idx] = "0"

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_item, i, item) for i, item in enumerate(test_data)]
            for future in as_completed(futures):
                future.result()
    except KeyboardInterrupt:
        print(f"\n⚠️ 收到中断信号，正在保存已完成的 {completed[0]} 条结果...")

    pbar.close()
    save_results()
    print(f"\n推理完成！已保存 {completed[0]}/{len(test_data)} 条结果到 {output_path}")
    if errors[0] > 0:
        print(f"其中 {errors[0]} 条出错，默认标记为 0")


if __name__ == "__main__":
    main()
