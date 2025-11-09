# # === 中央區域顏色分析（比例切 + 內縮） ===
# CENTER_RATIO = 0.6  # 取短邊的 50% 當中心方塊；可試 0.40~0.60
# MARGIN_RATIO = 0.06  # 先把整個裁切圖四邊內縮 6%，避免邊緣背景
#
# h, w = cropped.shape[:2]
# # 先做「內縮框」以避開邊緣雜訊
# mx = int(w * MARGIN_RATIO)
# my = int(h * MARGIN_RATIO)
# ix1, iy1 = mx, my
# ix2, iy2 = max(w - mx, ix1 + 1), max(h - my, iy1 + 1)
# inner = cropped[iy1:iy2, ix1:ix2].copy()
#
# # 在「內縮框」內以比例切中心方塊
# ih, iw = inner.shape[:2]
# side = max(1, int(min(iw, ih) * CENTER_RATIO))
# cx, cy = iw // 2, ih // 2
# x1 = max(cx - side // 2, 0)
# y1 = max(cy - side // 2, 0)
# x2 = min(cx + side // 2, iw)
# y2 = min(cy + side // 2, ih)
# cropped2 = inner[y1:y2, x1:x2].copy()
#
# # --- debug 圖 ---
# def _img_to_b64(img_rgb):
#     img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
#     ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
#     return f"data:image/jpeg;base64,{base64.b64encode(buf).decode('utf-8')}" if ok else None
#
# # 疊框：先畫內縮框，再畫中心方塊（兩層座標）
# overlay = cropped.copy()
# # 內縮框（藍色）
# cv2.rectangle(overlay, (ix1, iy1), (ix2, iy2), (255, 0, 0), 2)
# # 中心方塊（綠色），要加回內縮偏移量
# cv2.rectangle(overlay, (ix1 + x1, iy1 + y1), (ix1 + x2, iy1 + y2), (0, 255, 0), 2)
#
# center_b64 = _img_to_b64(cropped2)
# overlay_b64 = _img_to_b64(overlay)
# cropped_b64 = _img_to_b64(cropped)  # 完整裁切
#
# # --- 取得主色（4 值：RGB/HEX/HSV/佔比）---
# rgb_colors, hex_colors, hsv_values, ratios = _get_colors_ex(cropped2, k=3, min_ratio=0.3)
# pill_detection.py  — 無 rembg 版本（YOLO → 顏色/外型 → OCR）

import os
import base64
import logging
import cv2
import torch
import numpy as np
from ultralytics import YOLO

from app.utils.image_io import read_image_safely
from openocr import OpenOCR
from app.utils.ocr_utils import recognize_with_openocr
from app.utils.shape_color_utils import (
    rotate_image_by_angle,
    enhance_contrast,
    desaturate_image,
    enhance_for_blur,
    get_basic_color_name,

    get_dominant_colors,
    increase_brightness,

    detect_shape_from_image
)

# ====== 輕量化設定 ======
# Render 的 CPU 只有 1 核，避免 PyTorch/NumPy 開太多執行緒
torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))

logging.getLogger("openrec").setLevel(logging.ERROR)
# ocr_engine = OpenOCR(backend='onnx', device='cpu')

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

_ocr_engine = None


def get_ocr_engine():
    global _ocr_engine
    if _ocr_engine is None:
        print("[OCR] loading OpenOCR (onnx, cpu)…")
        _ocr_engine = OpenOCR(backend='onnx', device='cpu')
    return _ocr_engine


_det_model = None


def get_det_model():
    """Lazy-load YOLO 權重，只初始化一次"""
    global _det_model
    if _det_model is None:
        print("[DET] loading YOLO model…")
        m = YOLO("models/best.pt")

        try:
            m.fuse()
        except Exception:
            pass
        _det_model = m
        print("[DET] model ready")
    return _det_model


def generate_image_versions(base_img):
    """產生多個影像增強版本供 OCR 嘗試"""
    # v1 = enhance_contrast(base_img, 1.5, 1.5, -0.5)
    # 減少判斷
    # v2 = desaturate_image(v1)
    # v3 = enhance_contrast(base_img, 5.5, 2.0, -1.0)
    # v4 = desaturate_image(v3)
    # v5 = enhance_for_blur(base_img)
    # 減少判斷
    # return [
    #     (base_img, "原圖"),
    #     (v1, "增強1"),
    #     (v2, "去飽和1"),
    #     (v3, "增強2"),
    #     (v4, "去飽和2"),
    #     (v5, "模糊優化"),
    # ]
    # return [
    #     (base_img, "原圖"),
    #     (v1, "增強去飽和"),
    # ]

    return [
        (base_img, "原圖"),

    ]


def get_best_ocr_texts(
        image_versions,
        angles=(0, 45, 90, 135, 180, 225, 270, 315), ocr_engine=None,
        # angles=(0, 90, 180, 270), ocr_engine=None,
):
    version_results = {}
    score_dict = {}

    for img_v, version_name in image_versions:
        for angle in angles:
            rotated = rotate_image_by_angle(img_v, angle)
            full_name = f"{version_name}_旋轉{angle}"
            texts, score = recognize_with_openocr(
                rotated, ocr_engine=ocr_engine, name=full_name, min_score=0.8
            )
            version_results[full_name] = texts
            score_dict[full_name] = score

    score_combined = {
        k: (sum(len(txt) for txt in version_results[k]) * score_dict[k])
        for k in version_results
    }
    best_name = max(score_combined, key=score_combined.get)
    return version_results[best_name], best_name, score_dict[best_name]


# Don't use this function, it will consume a lot CPU.
# Although it will make Pill Detection accu to 100%, only under few cases will need fallback.

def _fallback_rembg_crop(input_img):
    """
    Fallback crop by removing background with rembg, then take the largest blob's bbox.
    input_img: np.ndarray in BGR (as read by OpenCV)
    return: cropped np.ndarray (BGR) or None if failed
    """
    try:
        from rembg import remove
    except Exception as e:
        print(f"[REMBG] rembg not available: {e}")
        return None

    try:
        # 1) rembg returns RGBA (with alpha); keep original resolution
        rgba = remove(input_img)  # input can be np.ndarray (BGR/RGB); rembg handles internally
        if rgba is None:
            print("[REMBG] remove() returned None")
            return None

        # Ensure we have 4 channels (RGBA). If bytes returned, try decode.
        if isinstance(rgba, bytes):
            rgba = cv2.imdecode(np.frombuffer(rgba, np.uint8), cv2.IMREAD_UNCHANGED)

        if rgba is None or rgba.ndim < 3 or rgba.shape[2] < 4:
            print("[REMBG] unexpected output shape")
            return None

        # 2) alpha mask → binary
        alpha = rgba[:, :, 3]
        # Heuristic binarization: Otsu + small opening/closing to clean noise
        _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphology to remove tiny speckles and fill small holes
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 3) find largest contour
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            print("[REMBG] no contours found on alpha mask")
            return None

        largest = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        # sanity check: discard absurdly tiny boxes
        H, W = mask.shape[:2]
        if w * h < 0.001 * (W * H):
            print("[REMBG] contour too small; likely noise")
            return None

        # 4) crop from original BGR image (not RGBA)
        x0 = max(0, x - 5)  # small padding
        y0 = max(0, y - 5)
        x1 = min(W, x + w + 5)
        y1 = min(H, y + h + 5)

        cropped = input_img[y0:y1, x0:x1].copy()
        if cropped is None or cropped.size == 0:
            print("[REMBG] crop is empty")
            return None

        # print(f"[REMBG] crop success: ({x0},{y0})-({x1},{y1})")
        return cropped

    except Exception as e:
        print(f"[REMBG] fallback error: {e}")
        return None


def _pick_crop_from_boxes(input_img, boxes):
    """從 YOLO boxes 選最佳框並回傳裁切圖（不再去背）"""
    xyxy = boxes.xyxy.cpu().numpy()  # [N,4]
    conf = boxes.conf.squeeze().cpu().numpy()
    conf = conf if conf.ndim else conf[None]

    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    score = conf * (areas / (areas.max() + 1e-6))  # 面積加權，避免挑到超小框
    best_idx = score.argmax()
    x1, y1, x2, y2 = map(int, xyxy[best_idx])

    pad = int(0.08 * max(x2 - x1, y2 - y1))  # 少量 padding
    h, w = input_img.shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)

    cropped = input_img[y1:y2, x1:x2]
    return cropped


import time  # 確保你有加上這行


def process_image(img_path: str):
    """
    單張藥品圖片辨識流程：
    圖片路徑 → 讀取 → YOLO → 裁切 → 顏色/外型 → 多版本 OCR → 回傳
    """
    # print(f"[PROC] start process_image: {img_path}")
    # t0 = time.perf_counter()
    debug_start = time.perf_counter()

    # === 讀圖（BGR）===
    image_bgr = read_image_safely(img_path)  # ✅ BGR 格式，OpenCV/YOLO 用
    if image_bgr is None:
        return {"error": "圖片讀取失敗"}

    # === 分出 RGB 給顏色分析用 ===
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)  # ✅ RGB 給顏色分析


    # === 用 BGR 做 YOLO 偵測 ===
    input_img = image_bgr.copy()
    # t1 = time.perf_counter()
    # print(f"⏱️ Pillow RGB → OpenCV BGR：{(t1 - t0)*1000:.1f} ms")
    # print(f"⏱️ 讀取圖片：{(t1 - t0)*1000:.1f} ms")

    # === 讀取模型（已快取）===
    det_model = get_det_model()
    # t2 = time.perf_counter()
    # print(f"⏱️ 讀取模型：{(t2 - t1)*1000:.1f} ms")

    # === Debug 標記：預測來源 ===
    det_src = "unknown"

    # === YOLO 預測時間 ===
    # print("🔍 YOLO 開始預測")
    # yolo_t0 = time.perf_counter()
    res = det_model.predict(
        source=input_img,
        imgsz=640,
        conf=0.25,
        iou=0.7,
        device=DEVICE,
        verbose=False
    )[0]
    # yolo_t1 = time.perf_counter()
    # print("✅ YOLO 結束預測")

    # === 裁切時間（含 fallback）===
    # crop_t0 = time.perf_counter()
    boxes = res.boxes
    if boxes is not None and boxes.xyxy.shape[0] > 0:
        cropped_bgr = _pick_crop_from_boxes(input_img, boxes)  # 給 OCR/encode
        cropped_rgb = _pick_crop_from_boxes(image_rgb, boxes)  # 給顏色分析
        det_src = "yolo_conf_0.25"
        # print("YOLO 0.25")
    else:
        res_lo = det_model.predict(
            source=input_img,
            imgsz=640,
            conf=0.10,
            iou=0.7,
            device=DEVICE,
            verbose=False
        )[0]
        boxes_lo = res_lo.boxes
        if boxes_lo is not None and boxes_lo.xyxy.shape[0] > 0:
            cropped_bgr = _pick_crop_from_boxes(input_img, boxes_lo)
            cropped_rgb = _pick_crop_from_boxes(image_rgb, boxes_lo)
            det_src = "yolo_conf_0.10"
            # print("YOLO 0.10")
        else:
            # 🚫 不再使用 rembg，直接回傳失敗
            # print("🔴 YOLO 失敗 (0.25 / 0.10)，無法擷取藥品")
            return {"error": "藥品擷取失敗"}

    # crop_t1 = time.perf_counter()

    # t3 = crop_t1
    # print(f"⏱️ YOLO 預測時間：{(yolo_t1 - yolo_t0)*1000:.1f} ms")
    # print(f"⏱️ 裁切（選框）時間：{(crop_t1 - crop_t0)*1000:.1f} ms")
    # print(f"⏱️ YOLO 偵測+裁切：{(t3 - t1)*1000:.1f} ms")

    # === 外型、顏色分析 (直接用裁切圖, 不去背) ===
    shape, _ = detect_shape_from_image(cropped_bgr, cropped_bgr, expected_shape=None)
    # t4 = time.perf_counter()
    # print(f"⏱️ 外型分析：{(t4 - t3)*1000:.1f} ms")

    # === 多版本 OCR 辨識 ===
    # print("🔍 OCR 開始辨識")
    image_versions = generate_image_versions(cropped_bgr)
    best_texts, best_name, best_score = get_best_ocr_texts(
        image_versions, ocr_engine=get_ocr_engine()
    )

    # t5 = time.perf_counter()
    # print("✅ OCR 結束辨識")
    # print(f"⏱️ OCR 多版本辨識：{(t5 - t4)*1000:.1f} ms")

    # === 中央區域顏色分析（比例切 + 內縮） ===
    CENTER_RATIO = 0.6  # 取短邊的 60% 當中心方塊
    MARGIN_RATIO = 0.06  # 裁切圖四邊內縮 6%

    h, w = cropped_rgb.shape[:2]
    mx, my = int(w * MARGIN_RATIO), int(h * MARGIN_RATIO)
    ix1, iy1 = mx, my
    ix2, iy2 = max(w - mx, ix1 + 1), max(h - my, iy1 + 1)
    inner = cropped_rgb[iy1:iy2, ix1:ix2].copy()

    ih, iw = inner.shape[:2]
    side = max(1, int(min(iw, ih) * CENTER_RATIO))
    cx, cy = iw // 2, ih // 2
    x1, y1 = max(cx - side // 2, 0), max(cy - side // 2, 0)
    x2, y2 = min(cx + side // 2, iw), min(cy + side // 2, ih)
    cropped2 = inner[y1:y2, x1:x2].copy()

    # === 顏色分析（中央區域）===
    cropped2 = increase_brightness(cropped2, value=20)
    rgb_colors, hex_colors = get_dominant_colors(cropped2, k=3, min_ratio=0.35)
    rgb_colors_int = [tuple(map(int, c)) for c in rgb_colors]
    # t6 = time.perf_counter()
    # print(f"⏱️ 中央顏色分析：{(t6 - t5)*1000:.1f} ms")

    # === encode 成 base64 傳回前端 ===
    ok, buffer = cv2.imencode(".jpg", cropped_bgr)
    cropped_b64 = (
        f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"
        if ok else None
    )

    # === 色彩名稱判定（由 RGB 轉 HSV，再分類）===
    basic_names, hsv_values = [], []
    for rgb in rgb_colors_int:
        bgr = np.uint8([[rgb[::-1]]])
        h_raw, s, v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
        hsv_values.append((h_raw * 2, s, v))
        basic_names.append(get_basic_color_name(rgb))

    colors = list(dict.fromkeys(basic_names))
    # t7 = time.perf_counter()
    # print(f"⏱️ 顏色分類：{(t7 - t6)*1000:.1f} ms")

    # === 最終結果輸出 ===
    # print(f"[PROC] OCR={best_texts}, shape={shape}, colors={colors}, score={best_score:.3f}")
    # print(f"⏱️ 🔚 總耗時（內部統計）：{(t7 - t0)*1000:.1f} ms")

    # debug_end = time.perf_counter()
    # print(f"🟠 process_image() 實際耗時（外層觀察）：{(debug_end - debug_start)*1000:.1f} ms")

    return {
        "文字辨識": best_texts if best_texts else ["None"],
        "最佳版本": best_name,
        "信心分數": round(best_score, 3),
        "顏色": colors,
        "外型": shape,
        "cropped_image": cropped_b64,
        "debug": {
            "det_source": det_src,
        }
    }
