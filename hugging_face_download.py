import os
import sys

import requests
from tqdm import tqdm

MODEL_URLS = {
    "best_ann_model.h5": "https://huggingface.co/Minh-Nguyen01/ann-model-cat-dog/resolve/main/ann_model.h5",
    "vgg_model.h5": "https://huggingface.co/Minh-Nguyen01/ann-model-cat-dog/resolve/main/vgg_model.h5",
    "inceptionv3.h5": "https://huggingface.co/Minh-Nguyen01/ann-model-cat-dog/resolve/main/inception_model.h5",
}

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def download_with_progress(url, output_path):
    response = requests.get(url, stream=True)
    total = int(response.headers.get("Content-Length", 0))

    if response.status_code != 200 or total < 5000:
        print(
            f"Không thể tải {os.path.basename(output_path)} – URL lỗi hoặc file không tồn tại.",
            flush=True,
        )
        return

    with open(output_path, "wb") as f:
        downloaded = 0
        chunk_size = 1024
        last_percent = -1
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                percent = int(downloaded * 100 / total)
                if percent % 5 == 0 and percent != last_percent:
                    tqdm.write(f"{os.path.basename(output_path)}: {percent}%")
                    sys.stdout.flush()
                    last_percent = percent
        print(f"Tải xong: {os.path.basename(output_path)}", flush=True)


for filename, url in MODEL_URLS.items():
    file_path = os.path.join(CURRENT_DIR, filename)

    if os.path.exists(file_path):
        print(f"{filename} đã tồn tại. Bỏ qua.")
    else:
        print(f"Đang tải: {filename}")
        download_with_progress(url, file_path)
