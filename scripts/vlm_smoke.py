"""v13 feasibility smoke test — can gemma3:4b recognise dance gestures?

Workflow:
  1. Pull motion peak times from demo.mp4 via v12 motion_detector.
  2. Pick 5 spread-out peaks across intensity buckets.
  3. Save each frame as PNG.
  4. POST each frame to Ollama gemma3:4b with a closed-gesture-vocabulary
     prompt.
  5. Print VLM's answer next to the saved PNG path so the human can
     eyeball the frame vs the label.

Hit-rate decision (per the design discussion):
  ≥ 60% : invest in full v13
  30-60%: try qwen2.5-vl:7b
  < 30% : fall back to rules-based gesture classifier from pose landmarks
"""

from __future__ import annotations

import base64
import json
import urllib.request
from pathlib import Path

import cv2

from semanticvibe.motion_detector import detect_motion_peaks

import argparse

VIDEO_PATH = Path("data/test_videos/demo.mp4")
OUT_DIR = Path("outputs/vlm_smoke")
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma3:4b"

GESTURE_VOCAB = [
    "heart_hands",      # 比心
    "arms_raised",      # 舉手/手在頭頂
    "jump",             # 跳起來
    "peace_sign",       # 比 V
    "point_at_camera",  # 指鏡頭
    "spin",             # 轉身/側身
    "clap",             # 拍手
    "smile_close_up",   # 大笑/特寫笑
    "pose_static",      # 靜止擺姿勢
    "lean_or_sway",     # 傾身/搖晃
    "none",             # 看不出特定動作
]

PROMPT = f"""Look at this image of a young woman who is dancing or vlogging.
What single gesture / action is she most clearly doing right now?

Pick EXACTLY ONE label from this list:
{chr(10).join('- ' + g for g in GESTURE_VOCAB)}

Respond with ONLY the label, lowercase, nothing else. No explanation."""


def grab_frame(video_path: Path, t: float, out_path: Path) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"can't read frame at {t}s")
    # Resize to ~512px-wide so we don't blow the VLM's image token budget.
    h, w = frame.shape[:2]
    if w > 512:
        scale = 512 / w
        frame = cv2.resize(frame, (512, int(h * scale)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)


def ask_vlm(image_path: Path, model: str = DEFAULT_MODEL) -> str:
    b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    body = json.dumps({
        "model": model,
        "prompt": PROMPT,
        "images": [b64],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 32},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("response", "").strip().lower().split("\n")[0]


def main() -> int:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Ollama model tag (default gemma3:4b). For v13 "
                        "feasibility try qwen2.5vl:7b.")
    args = p.parse_args()

    print(f"[1/3] detecting motion peaks in {VIDEO_PATH}")
    info = detect_motion_peaks(str(VIDEO_PATH))
    print(f"      → {len(info['peak_times'])} peaks")

    # Pick 5 well-spread peaks across the full clip + intensity range.
    peaks = list(zip(info["peak_times"],
                     [info["peak_intensities"][t] for t in info["peak_times"]]))
    if len(peaks) < 5:
        print("⚠️  too few peaks; smoke test inconclusive")
        return 1
    # 1 from each thirds + ensure at least one "high".
    thirds = [peaks[i] for i in (
        0,
        len(peaks) // 4,
        len(peaks) // 2,
        len(peaks) * 3 // 4,
        len(peaks) - 1,
    )]
    # Make sure we have at least one 'high' if any exist.
    high_peaks = [p for p in peaks if p[1] == "high"]
    if high_peaks and not any(p[1] == "high" for p in thirds):
        thirds[2] = high_peaks[len(high_peaks) // 2]
    sample = thirds

    print(f"[2/3] saving 5 frame PNGs at: {[round(t, 2) for t, _ in sample]}")
    paths = []
    for t, intensity in sample:
        out = OUT_DIR / f"frame_t{t:.2f}_{intensity}.png"
        grab_frame(VIDEO_PATH, t, out)
        paths.append((t, intensity, out))
        print(f"      → {out}")

    print(f"[3/3] querying {args.model} on each frame…")
    print()
    print(f"{'t (s)':>6}  {'intensity':<8}  {'VLM answer':<28}  {'frame'}")
    print("-" * 88)
    import time
    t0 = time.time()
    for t, intensity, path in paths:
        ans = ask_vlm(path, model=args.model)
        ans_short = ans[:30]
        print(f"{t:>6.2f}  {intensity:<8}  {ans_short:<28}  {path}")
    print(f"\ntotal VLM time: {time.time() - t0:.1f}s for 5 frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
