import os
import argparse
from huggingface_hub import snapshot_download, HfApi, HfFolder
from huggingface_hub.utils import RepositoryNotFoundError, LocalEntryNotFoundError
from huggingface_hub.hf_api import GatedRepoError


def download_model(model_name, save_path=None, mirror_url=None):
    """
    从 Hugging Face 下载模型到指定路径

    参数:
        model_name (str): Hugging Face 模型标识符
        save_path (str): 本地保存路径 (默认: 当前目录下的 models 文件夹)
        mirror_url (str): Hugging Face 镜像源 URL
    """
    # 设置默认保存路径
    if save_path is None:
        save_path = os.path.join(os.getcwd(), "models")

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    # 检查模型是否存在
    api = HfApi(endpoint=mirror_url if mirror_url else None)
    try:
        model_info = api.model_info(model_name)
    except RepositoryNotFoundError:
        print(f"❌ 错误: 模型 '{model_name}' 不存在")
        return
    except GatedRepoError:
        print(f"🔒 错误: 模型 '{model_name}' 是受保护的，请先登录并接受许可协议")
        print(f"请访问: https://huggingface.co/{model_name}")
        return

    # 设置下载参数
    download_kwargs = {
        "repo_id": model_name,
        "local_dir": os.path.join(save_path, model_name.replace("/", "_")),
        "local_dir_use_symlinks": False,
        "resume_download": True,
        "token": HfFolder.get_token(),
    }

    # 设置镜像源（如果提供）
    if mirror_url:
        download_kwargs["endpoint"] = mirror_url

    try:
        print(f"⬇️ 开始下载模型: {model_name}")
        print(f"📁 保存路径: {download_kwargs['local_dir']}")
        if mirror_url:
            print(f"🌐 使用镜像源: {mirror_url}")

        # 执行下载
        snapshot_download(**download_kwargs)

        print(f"✅ 模型 '{model_name}' 下载完成!")
        print(f"文件保存在: {download_kwargs['local_dir']}")

    except LocalEntryNotFoundError:
        print(f"❌ 错误: 模型 '{model_name}' 文件不存在或访问受限")
    except Exception as e:
        print(f"❌ 下载失败: {str(e)}")
        print("💡 建议: 尝试使用不同的镜像源或检查网络连接")


def main():
    parser = argparse.ArgumentParser(description="从 Hugging Face 下载模型")
    parser.add_argument(
        "model_name", help="Hugging Face 模型标识符 (例如: 'bert-base-uncased')"
    )
    parser.add_argument(
        "-p",
        "--path",
        default=os.path.join(os.getcwd(), "models"),
        help="保存路径 (默认: ./models)",
    )
    parser.add_argument(
        "-m",
        "--mirror",
        default=None,
        help="Hugging Face 镜像源 URL (例如: https://hf-mirror.com)",
    )

    args = parser.parse_args()

    download_model(
        model_name=args.model_name, save_path=args.path, mirror_url=args.mirror
    )


if __name__ == "__main__":
    main()
