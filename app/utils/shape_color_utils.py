import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter


def rotate_image_by_angle(image, angle):
    """
    將圖片依指定角度旋轉。
    - image: 輸入圖片（OpenCV 格式）
    - angle: 順時針旋轉角度（例如 90, 180）
    - return: 旋轉後的圖片
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated


def enhance_contrast(img, clip_limit, alpha, beta):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhance_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhance_img, (5, 5), 3.0)
    return cv2.addWeighted(enhance_img, alpha, blurred, beta, 0)


def rgb_to_hex(color):
    """Convert RGB (0-255) to HEX string."""
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


def get_center_region(img, size=100):
    """
    擷取圖片的中央區域 (固定大小)。
    - img: 輸入圖片 (H, W, C)
    - size: 方形區域邊長 (像素)，預設 100
    - return: 中央裁切後的圖片
    """
    h, w = img.shape[:2]

    cx, cy = w // 2, h // 2  # 圖片中心點

    # 計算邊界
    x1 = max(cx - size // 2, 0)
    y1 = max(cy - size // 2, 0)
    x2 = min(cx + size // 2, w)
    y2 = min(cy + size // 2, h)

    return img[y1:y2, x1:x2]


def is_color_similar(hsv1, hsv2, h_thresh=20, s_thresh=70, v_thresh=70):
    """
    Check if two HSV colors are similar within thresholds.
    Hue in degrees (0-360), s & v in [0-255].
    """
    h1, s1, v1 = hsv1
    h2, s2, v2 = hsv2

    dh = min(abs(h1 - h2), 360 - abs(h1 - h2))
    ds = abs(s1 - s2)
    dv = abs(v1 - v2)

    # if both are very low saturation, compare only brightness
    if s1 < 30 and s2 < 30:
        return abs(v1 - v2) < 120

    return (dh <= h_thresh) and (ds <= s_thresh) and (dv <= v_thresh)


def get_basic_color_name(rgb):
    """Classify an RGB value into a basic color family (Chinese Traditional)."""
    # Convert RGB (0–255) to HSV
    bgr = np.uint8([[rgb[::-1]]])  # OpenCV expects BGR
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = hsv
    # print(f"HSV: {h}, {s}, {v}")
    h = int(h) * 2  # Convert to degrees (0-360)
    # print(f"h: {h}")
    r, g, b = rgb
    # print(f"RGB: {r}, {g}, {b}")
    # Handle black/white/gray
    if v < 30:  # instead of 40 → more tolerant to lighting
        return "黑色"
    if s < 55:  # or is it 140? for the greyish and v > 180
        return "白色"

    # Hue ranges
    if (h < 20 or h >= 330) and s > 70 and r > 50:
        return "紅色"
    elif (s < 35 and 35 < v < 220) or ((196 < h < 250) and s < 100 and v < 100):
        return "灰色"

        # return "灰色"
    elif h < 40 or (h > 195 and v < 100):
        # elif h < 40 and r < 240:
        return "棕色" if v < 80 else "橘色"

    elif h < 75:
        return "黃色"
    elif h < 195:  # green and blue only, need to add s and v
        return "綠色"
    elif h < 250:
        return "藍色"
    elif h < 300:  # candidate for pink (270–330)
        return "紫色"
    elif h < 360:  # candidate for pink (270–330)
        return "粉紅色"
    else:
        return "其他"


def get_dominant_colors(image, k=3, ignore_black=True, min_ratio=0.3):
    """Extract dominant colors using KMeans, ignore black and small/shadow clusters."""
    # resize for speed
    # img = cv2.resize(image, (600, 400))
    img_flat = image.reshape((-1, 3))

    # run kmeans
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(img_flat)

    counts = Counter(labels)
    centers = kmeans.cluster_centers_

    # order by frequency
    ordered = counts.most_common()
    ordered_colors = [centers[i] for i, _ in ordered]

    # filter out black if requested
    if ignore_black:
        def is_black(c, threshold=40):  # v < 40 means black
            bgr = np.uint8([[c[::-1]]])
            h, s, v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
            return v < threshold

        ordered_colors = [c for c in ordered_colors if not is_black(c)]

    # ---- merge similar colors ----
    merged_colors = []
    for idx, c in enumerate(ordered_colors):
        bgr = np.uint8([[c[::-1]]])
        h_raw, s, v = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0][0]
        h_deg = h_raw * 2
        hsv_c = (h_deg, int(s), int(v))

        merged = False
        for mc in merged_colors:
            if is_color_similar(hsv_c, mc["hsv"]):
                mc["count"] += counts[ordered[idx][0]]
                merged = True
                break

        if not merged:
            merged_colors.append({
                "rgb": tuple(map(int, c)),
                "hsv": hsv_c,
                "count": counts[ordered[idx][0]]
            })

    # ---- filter out small clusters (shadows, noise) ----
    total = sum(mc["count"] for mc in merged_colors)
    filtered_colors = [mc for mc in merged_colors if mc["count"] / total >= min_ratio]

    # safeguard: if all removed, keep the largest one
    if not filtered_colors and merged_colors:
        filtered_colors = [max(merged_colors, key=lambda mc: mc["count"])]

    hex_colors = [rgb_to_hex(mc["rgb"]) for mc in filtered_colors]

    return [mc["rgb"] for mc in filtered_colors], hex_colors


# === 可調參數 ===
CIRCLE_LO = 1
CIRCLE_HI = 1.2
ELLIPSE_HI = 3.8


def set_shape_thresholds(circle_lo: float, circle_hi: float, ellipse_hi: float):
    global CIRCLE_LO, CIRCLE_HI, ELLIPSE_HI
    CIRCLE_LO = circle_lo
    CIRCLE_HI = circle_hi
    ELLIPSE_HI = ellipse_hi


def detect_shape_three_classes(contour, expected_shape=None):
    set_shape_thresholds(CIRCLE_LO, CIRCLE_HI, ELLIPSE_HI)
    shape = "其他"
    try:
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            major, minor = axes
            if minor == 0:
                return shape

            ratio = max(major, minor) / min(major, minor)
            ratios_list.append(ratio)
            # print(f"Ellipse ratio: {ratio:.3f}")
            # === classify with global thresholds ===
            if CIRCLE_LO <= ratio <= CIRCLE_HI:
                shape = "圓形"
            elif ratio <= ELLIPSE_HI:
                shape = "橢圓形"
            else:
                shape = "其他"


    except Exception as e:
        print(f"❗ detect_shape_three_classes 發生錯誤：{e}")
    return shape


def increase_brightness(img, value=30):
    """Increase brightness of an RGB image by boosting V channel in HSV."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img_bright = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)
    return img_bright


# === 增強處理函式 ===

def desaturate_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s.fill(0)
    return cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)


def enhance_for_blur(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    contrast_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(contrast_img, (3, 3), 1.0)
    sharpened = cv2.addWeighted(contrast_img, 1.8, blurred, -0.8, 0)
    return cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)


def preprocess_with_shadow_correction(img_bgr):
    """改進的前處理，更好地分離藥物與背景"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 多尺度的背景估計
    blur1 = cv2.GaussianBlur(gray, (25, 25), 0)
    blur2 = cv2.GaussianBlur(gray, (75, 75), 0)

    # 雙重陰影校正
    corrected1 = cv2.divide(gray, blur1, scale=255)
    corrected2 = cv2.divide(gray, blur2, scale=255)
    corrected = cv2.addWeighted(corrected1, 0.5, corrected2, 0.5, 0)

    # 使用 OTSU 自動找最佳閾值
    _, otsu_thresh = cv2.threshold(corrected, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形態學操作去除雜訊
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(otsu_thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    return cleaned


def detect_shape_from_image(cropped_img, original_img=None, expected_shape=None):
    try:
        output = cropped_img.copy()
        thresh = preprocess_with_shadow_correction(output)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        shape = "其他"
        if not contours and original_img is not None:
            gray_fallback = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            _, thresh_fallback = cv2.threshold(gray_fallback, 127, 255, cv2.THRESH_BINARY)
            contours_fallback, _ = cv2.findContours(thresh_fallback, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours_fallback:
                main_contour = max(contours_fallback, key=cv2.contourArea)
                shape = detect_shape_three_classes(main_contour, expected_shape=expected_shape)
        elif contours:
            main_contour = max(contours, key=cv2.contourArea)
            shape = detect_shape_three_classes(main_contour, expected_shape=expected_shape)

        if expected_shape:
            result = "OK" if shape == expected_shape else "BAD"
            return shape, result
        return shape, None
    except Exception as e:
        print(f"發生錯誤：{e}")
        return "錯誤", None


ratios_list = []
