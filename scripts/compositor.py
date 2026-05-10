# semanticvibe/compositor.py 的核心改動

from moviepy.editor import ImageClip, CompositeVideoClip
from semanticvibe.animations import compose_animation


def make_animated_clip(image, element, fps=30):
    """把一張 PIL image 包裝成有動畫的 MoviePy clip。"""
    
    duration = element["end_time"] - element["start_time"]
    
    # 取得動畫 lambdas
    anim = compose_animation(
        entry_name=element.get("entry_animation", "fade"),
        idle_name=element.get("idle_animation", "pulse"),
        entry_duration=0.4,
    )
    
    base_x, base_y = element["position"]
    base_size = element.get("size", 100)
    
    # 用 numpy array 創建 clip,持續 duration 秒
    import numpy as np
    np_img = np.array(image)
    clip = ImageClip(np_img, transparent=True, duration=duration)
    
    # 套用動畫
    # 注意:MoviePy 的 lambda t 是「from clip start」,
    # 跟我們的 element t 是同一個基準
    
    clip = clip.resize(lambda t: anim["scale"](t))
    clip = clip.rotate(lambda t: anim["rotation"](t))
    
    # 位置 = base + offset
    clip = clip.set_position(lambda t: (
        base_x + anim["offset_x"](t),
        base_y + anim["offset_y"](t),
    ))
    
    # 透明度比較複雜,用 mask
    clip = clip.set_opacity(lambda t: anim["opacity"](t))
    
    # 設定在主時間軸的位置
    clip = clip.set_start(element["start_time"])
    
    return clip