import os
# 关键设置：设置环境变量使用镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

def download_model(repo_id, local_dir):
    print(f"🚀 正在下载: {repo_id} ...")
    print(f"📂 保存路径: {local_dir}")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False, # Windows下必须设为False，否则会报错
            resume_download=True,         # 支持断点续传
            max_workers=8                 # 多线程下载，加速
        )
        print(f"✅ {repo_id} 下载完成！\n")
    except Exception as e:
        print(f"❌ {repo_id} 下载失败: {e}")

if __name__ == "__main__":
    # 1. 下载 MacBERT (Base版本)
    download_model(
        repo_id="hfl/chinese-macbert-base", 
        local_dir="./pretrained_models/chinese-macbert-base"
    )

    # 2. 下载 Qwen-1.5-1.8B-Chat (LLM)
    # 注意：这里下载的是 Chat 版本，适合对话和指令微调
    download_model(
        repo_id="Qwen/Qwen1.5-1.8B-Chat", 
        local_dir="./pretrained_models/Qwen1.5-1.8B-Chat"
    )
