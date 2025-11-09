# setup_models.py
import os
import urllib.request

os.makedirs("models", exist_ok=True)

urls = {
    "openocr_det_model.onnx": "https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_det_model.onnx",
    "openocr_rec_model.onnx": "https://github.com/Topdu/OpenOCR/releases/download/develop0.0.1/openocr_rec_model.onnx"
}

for filename, url in urls.items():
    dest = os.path.join("models", filename)
    if not os.path.exists(dest):
        print(f"下載 {filename} ...")
        urllib.request.urlretrieve(url, dest)
    else:
        print(f" {filename} 已存在，跳過下載。")
