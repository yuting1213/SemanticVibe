# scripts/clean_backgrounds.py
"""批次處理素材庫,把白底 PNG 都去背成透明。"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


STICKER_DIR = Path("assets")
WHITE_THRESHOLD = 240  # RGB 都 > 240 視為白底
ALPHA_THRESHOLD = 250  # 已經是透明的就跳過


def has_white_background(img: Image.Image) -> bool:
    """偵測四個角落是否為白色,判斷是否為白底圖。"""
    rgba = img.convert("RGBA")
    arr = np.array(rgba)
    h, w = arr.shape[:2]
    
    # 看四個角落 10x10 區域
    corners = [
        arr[:10, :10],   # 左上
        arr[:10, -10:],  # 右上
        arr[-10:, :10],  # 左下
        arr[-10:, -10:], # 右下
    ]
    
    white_count = 0
    for corner in corners:
        # alpha 要不透明 (代表沒去過背)
        if corner[..., 3].mean() < 250:
            continue
        # RGB 要接近白
        if corner[..., :3].mean() > WHITE_THRESHOLD:
            white_count += 1
    
    return white_count >= 3  # 至少 3 個角是白的


def remove_white_background(img: Image.Image, threshold=230, soft_edge=15):
    """把白色像素變透明,邊緣做 alpha 漸變避免鋸齒。"""
    rgba = img.convert("RGBA")
    arr = np.array(rgba).astype(np.int16)
    
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    
    # 計算「離白色的距離」(0=純白, 255=完全不白)
    distance_from_white = 255 - np.minimum(np.minimum(r, g), b)
    
    # 在 threshold ~ threshold-soft_edge 範圍內做漸變
    # 比 threshold 接近白 → 透明
    # 比 threshold-soft_edge 遠離白 → 不透明
    new_alpha = np.clip(
        (distance_from_white - (255 - threshold)) * (255 / soft_edge),
        0, 255
    ).astype(np.uint8)
    
    arr[..., 3] = np.minimum(arr[..., 3], new_alpha)
    
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA")


def main():
    pngs = list(STICKER_DIR.rglob("*.png"))
    print(f"檢查 {len(pngs)} 張 PNG")
    
    fixed = skipped = 0
    for png_path in tqdm(pngs, desc="Processing"):
        try:
            img = Image.open(png_path)
            if has_white_background(img):
                cleaned = remove_white_background(img)
                # 備份原檔
                backup = png_path.with_suffix(".png.bak")
                if not backup.exists():
                    png_path.rename(backup)
                cleaned.save(png_path, "PNG")
                fixed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"\n[ERR] {png_path.name}: {e}")
    
    print(f"\n修正 {fixed} 張白底 PNG,跳過 {skipped} 張(已透明)")
    print("原檔備份為 *.png.bak,確認效果好就可以刪掉備份")


if __name__ == "__main__":
    main()