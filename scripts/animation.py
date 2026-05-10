# semanticvibe/animations.py
"""動畫系統:進場動畫 + 持續動畫,持續動畫是這檔案的重點。"""

import math
import random
from typing import Callable, Dict


# ============ Easing 函式 ============

def ease_out_back(t: float) -> float:
    """過彈再回來,適合 scale_pop。"""
    c1, c3 = 1.70158, 2.70158
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def ease_out_elastic(t: float) -> float:
    """彈性大幅震盪。"""
    c4 = (2 * math.pi) / 3
    if t == 0 or t == 1:
        return t
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * c4) + 1


def ease_out_bounce(t: float) -> float:
    """像球落地反彈。"""
    n1, d1 = 7.5625, 2.75
    if t < 1 / d1:
        return n1 * t * t
    elif t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    elif t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    else:
        t -= 2.625 / d1
        return n1 * t * t + 0.984375


# ============ 進場動畫 (Entry) ============
# 全部回傳 {scale, opacity, rotation, offset_x, offset_y} 的 lambda
# t 是「從元素 start_time 起算的秒數」

def entry_fade(duration=0.4):
    return {
        "opacity": lambda t: min(1.0, t / duration) if t < duration else 1.0,
        "scale":   lambda t: 1.0,
        "rotation": lambda t: 0,
        "offset_x": lambda t: 0,
        "offset_y": lambda t: 0,
    }


def entry_scale_pop(duration=0.4):
    return {
        "opacity": lambda t: min(1.0, t / 0.15),
        "scale": lambda t: ease_out_back(min(1.0, t / duration)) if t < duration else 1.0,
        "rotation": lambda t: 0,
        "offset_x": lambda t: 0,
        "offset_y": lambda t: 0,
    }


def entry_drop_in(duration=0.5, from_y=-200):
    def y_offset(t):
        if t >= duration:
            return 0
        progress = ease_out_bounce(t / duration)
        return from_y * (1 - progress)
    return {
        "opacity": lambda t: min(1.0, t / 0.2),
        "scale": lambda t: 1.0,
        "rotation": lambda t: 0,
        "offset_x": lambda t: 0,
        "offset_y": y_offset,
    }


def entry_slide_in(duration=0.4, from_dir="left"):
    sign = -1 if from_dir == "left" else 1
    from_x = sign * 300
    def x_offset(t):
        if t >= duration:
            return 0
        progress = 1 - (1 - t / duration) ** 3  # ease_out_cubic
        return from_x * (1 - progress)
    return {
        "opacity": lambda t: min(1.0, t / 0.2),
        "scale": lambda t: 1.0,
        "rotation": lambda t: 0,
        "offset_x": x_offset,
        "offset_y": lambda t: 0,
    }


def entry_stamp(duration=0.3):
    """蓋章感:從 1.5 倍縮到 1 倍 + 震動。"""
    def scale(t):
        if t >= duration:
            return 1.0
        progress = t / duration
        return 1.5 - 0.5 * ease_out_back(progress)
    def rotation(t):
        if t >= duration:
            return 0
        # 震動衰減
        return 5 * math.sin(t * 30) * (1 - t / duration)
    return {
        "opacity": lambda t: 1.0 if t > 0 else 0,
        "scale": scale,
        "rotation": rotation,
        "offset_x": lambda t: 0,
        "offset_y": lambda t: 0,
    }


def entry_wobble_in(duration=0.5):
    def rotation(t):
        if t >= duration:
            return 0
        progress = t / duration
        return 15 * math.cos(progress * math.pi * 3) * (1 - progress)
    return {
        "opacity": lambda t: min(1.0, t / 0.2),
        "scale": lambda t: min(1.0, t / 0.3),
        "rotation": rotation,
        "offset_x": lambda t: 0,
        "offset_y": lambda t: 0,
    }


def entry_spin_in(duration=0.6):
    def rotation(t):
        if t >= duration:
            return 0
        progress = 1 - (1 - t / duration) ** 3
        return 360 * (1 - progress)
    return {
        "opacity": lambda t: min(1.0, t / 0.2),
        "scale": lambda t: min(1.0, ease_out_back(t / duration)) if t < duration else 1.0,
        "rotation": rotation,
        "offset_x": lambda t: 0,
        "offset_y": lambda t: 0,
    }


# ============ 持續動畫 (Idle) — 這是你最缺的 ============
# 注意這些函式回傳的 lambda 是「持續用的」,不會結束

def idle_pulse(period=1.5, amplitude=0.06):
    """呼吸式縮放。amplitude=0.06 代表 ±6% 大小變化。"""
    return {
        "scale": lambda t: 1.0 + amplitude * math.sin(2 * math.pi * t / period),
        "rotation": lambda t: 0,
        "offset_x": lambda t: 0,
        "offset_y": lambda t: 0,
        "opacity": lambda t: 1.0,
    }


def idle_wiggle(freq=2.5, amplitude=8, rotation_amp=4):
    """高頻小抖動,模擬手繪不穩定感。"""
    return {
        "scale": lambda t: 1.0,
        "rotation": lambda t: rotation_amp * math.sin(2 * math.pi * freq * t),
        "offset_x": lambda t: amplitude * math.sin(2 * math.pi * freq * t),
        "offset_y": lambda t: amplitude * 0.6 * math.cos(2 * math.pi * freq * t * 1.3),
        "opacity": lambda t: 1.0,
    }


def idle_drift(distance=25, period=4, direction=0):
    """緩慢飄移。direction 是角度(0=往右,90=往下)。"""
    rad = math.radians(direction)
    dx, dy = math.cos(rad) * distance, math.sin(rad) * distance
    return {
        "scale": lambda t: 1.0,
        "rotation": lambda t: 0,
        "offset_x": lambda t: dx * math.sin(2 * math.pi * t / period),
        "offset_y": lambda t: dy * math.sin(2 * math.pi * t / period),
        "opacity": lambda t: 1.0,
    }


def idle_rotate_slow(speed=15):
    """持續慢速旋轉。speed 是度/秒。"""
    return {
        "scale": lambda t: 1.0,
        "rotation": lambda t: speed * t,
        "offset_x": lambda t: 0,
        "offset_y": lambda t: 0,
        "opacity": lambda t: 1.0,
    }


def idle_shimmer(period=1.0, opacity_range=(0.6, 1.0)):
    """透明度小幅變化,閃光感。"""
    lo, hi = opacity_range
    mid, amp = (lo + hi) / 2, (hi - lo) / 2
    return {
        "scale": lambda t: 1.0,
        "rotation": lambda t: 0,
        "offset_x": lambda t: 0,
        "offset_y": lambda t: 0,
        "opacity": lambda t: mid + amp * math.sin(2 * math.pi * t / period),
    }


def idle_breathe_rotate(period=3, scale_amp=0.08, rot_amp=3):
    """組合動畫:呼吸 + 微旋轉。最自然。"""
    return {
        "scale": lambda t: 1.0 + scale_amp * math.sin(2 * math.pi * t / period),
        "rotation": lambda t: rot_amp * math.sin(2 * math.pi * t / (period * 1.3)),
        "offset_x": lambda t: 0,
        "offset_y": lambda t: 0,
        "opacity": lambda t: 1.0,
    }


def idle_static():
    return {
        "scale": lambda t: 1.0,
        "rotation": lambda t: 0,
        "offset_x": lambda t: 0,
        "offset_y": lambda t: 0,
        "opacity": lambda t: 1.0,
    }


# ============ 註冊表 ============

ENTRY_ANIMATIONS = {
    "fade": entry_fade,
    "scale_pop": entry_scale_pop,
    "drop_in": entry_drop_in,
    "slide_in": entry_slide_in,
    "stamp": entry_stamp,
    "wobble_in": entry_wobble_in,
    "spin_in": entry_spin_in,
}

IDLE_ANIMATIONS = {
    "static": idle_static,
    "pulse": idle_pulse,
    "wiggle": idle_wiggle,
    "drift": idle_drift,
    "rotate_slow": idle_rotate_slow,
    "shimmer": idle_shimmer,
    "breathe_rotate": idle_breathe_rotate,
}


# ============ 組合器 (這是關鍵!) ============

def compose_animation(entry_name: str, idle_name: str, 
                      entry_duration: float = 0.4) -> Dict[str, Callable]:
    """
    把 entry + idle 組合成單一 lambda dict。
    渲染時 t < entry_duration 用 entry,之後用 idle。
    
    回傳的 lambdas 接受 t (從元素 start_time 起算)。
    """
    entry = ENTRY_ANIMATIONS[entry_name](duration=entry_duration) \
            if entry_name in ENTRY_ANIMATIONS else entry_fade()
    idle = IDLE_ANIMATIONS[idle_name]() \
           if idle_name in IDLE_ANIMATIONS else idle_static()
    
    def make_combined(prop):
        entry_fn = entry[prop]
        idle_fn = idle[prop]
        def combined(t):
            if t < entry_duration:
                return entry_fn(t)
            else:
                # idle 從 entry 結束後才開始,所以 t 要扣掉
                idle_t = t - entry_duration
                idle_val = idle_fn(idle_t)
                # opacity 特例:entry 結束時是 1,idle 在 1 附近抖動,直接用 idle
                if prop == "opacity":
                    return idle_val
                # scale 特例:entry 結束在 1.0,idle 在 1.0 附近,用 idle
                if prop == "scale":
                    return idle_val
                # rotation 和 offset:idle 在 0 附近抖動,直接用 idle
                return idle_val
        return combined
    
    return {prop: make_combined(prop) for prop in 
            ["scale", "rotation", "offset_x", "offset_y", "opacity"]}