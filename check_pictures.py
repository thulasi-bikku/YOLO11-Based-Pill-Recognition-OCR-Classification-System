# app/check_pictures.py

import os
import pandas as pd
from pathlib import Path

# 路徑設定
EXCEL_PATH = Path("data/TESTData.xlsx")
PICTURE_ROOT = Path(r"C:\Users\92102\OneDrive - NTHU\桌面\大三下\畢業專題\APP_新版本\data\pictures")
REPORT_PATH = Path("reports/missing_pictures.xlsx")

# 支援的副檔名
VALID_EXTS = {".jpg"}


def check_pictures(excel_path: Path, image_root: Path):
    df = pd.read_excel(excel_path)

    missing = []
    not_jpg_only = []

    for _, row in df.iterrows():
        code = str(row.get("批價碼", "")).strip()
        name = str(row.get("學名", "")).strip()
        if not code:
            continue

        matches = list(image_root.glob(f"{code}.*"))
        matches = [p for p in matches if p.suffix in VALID_EXTS]

        if not matches:
            missing.append({"批價碼": code, "學名": name})
        else:
            # 有圖片但沒有 .jpg
            has_jpg = any(p.suffix == ".jpg" for p in matches)
            if not has_jpg:
                not_jpg_only.append({
                    "批價碼": code,
                    "學名": name,
                    "圖片副檔名": ", ".join(sorted(set(p.suffix for p in matches)))
                })

    # 輸出統計
    print("\n 沒有任何圖片的藥物：")
    for item in missing:
        print(f"  - {item['批價碼']} ➜ {item['學名']}")
    print(f"共 {len(missing)} 筆\n")

    print(" 只有非 .jpg 圖片的藥物：")
    for item in not_jpg_only:
        print(f"  - {item['批價碼']} ➜ {item['學名']}（{item['圖片副檔名']}）")
    print(f"共 {len(not_jpg_only)} 筆\n")

    # 輸出成 Excel
    os.makedirs(REPORT_PATH.parent, exist_ok=True)
    with pd.ExcelWriter(REPORT_PATH, engine="openpyxl") as writer:
        pd.DataFrame(missing).to_excel(writer, sheet_name="缺圖", index=False)
        pd.DataFrame(not_jpg_only).to_excel(writer, sheet_name="非JPG圖片", index=False)

    print(f"已輸出報表：{REPORT_PATH}")


if __name__ == "__main__":
    check_pictures(EXCEL_PATH, PICTURE_ROOT)
