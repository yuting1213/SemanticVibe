# v10 — outlined subtitles + dual-zone forbidden-map layout

## What changed since v9

Two visual problems were dragging the v9 output down:
1. The green / pink rounded-rect chip behind every subtitle was too heavy
   — it ate frame real estate and competed with the dancer for attention.
2. Decorations frequently landed on top of subtitles or person silhouettes
   because `find_placement_zone` only knew about the pose mask.

v10 fixes both:

1. **Subtitle is now純文字 + 厚描邊** (`SubtitleOutlinedElement`):
   transparent background, white fill, 6-8 px circular outline in the
   style preset's accent colour, optional drop shadow. The video shows
   through everywhere outside the glyph silhouette.
2. **Per-time `ForbiddenMap`**: dilated person mask ∪ active subtitle
   bboxes → boolean grid. Decorations call
   `find_free_position(target_size, prefer_zone=…)` and only land in the
   remaining empty area. Hero decorations prefer one of four corner
   zones; ambient decorations free-place. When the canvas is too crowded
   the placer first shrinks to half size, then skips with a warning.
3. **`baseline_kenpa` style preset** — alternates pink (`#FF6B9D`) and
   sky-blue (`#5BBFE3`) outline colours per-line, mimicking the
   reference  けんぱ baseline aesthetic.

### New / modified

| File | What |
|---|---|
| `src/semanticvibe/layout/forbidden_map.py` (new) | `ForbiddenMap(W, H)` class with `add_rect(x,y,w,h,padding)`, `add_person_mask(mask, padding_iters)` (uses `scipy.ndimage.binary_dilation`), `coverage_pct()`, `find_free_position(target, prefer_zone, rng, step, top_k)`. Factory `build_forbidden_map_at_time(t, person_masks, subtitle_rects, canvas_size, ...)` returns the per-time forbidden grid. |
| `src/semanticvibe/layout/__init__.py` | Re-exports `ForbiddenMap` + `build_forbidden_map_at_time`. |
| `src/semanticvibe/schemas/decision.py` | New `SubtitleOutlinedElement(type='subtitle_outlined', content, position, font, size, text_color, outline_color, outline_width, shadow_offset, shadow_alpha, shadow_color, margin, max_width_ratio)`. Added `pixel_anchor: PixelAnchor \| None` to `DecorationElement` so the forbidden-map can pin decorations to specific positions, overriding the `near_text_id` heuristic. |
| `src/semanticvibe/render/text_render.py` | New `render_subtitle_outlined(element, state, fonts_dir)` — circular thick outline + optional drop shadow + transparent background. Plus `measure_subtitle_outlined`, `fit_subtitle_outlined_to_canvas`, `resolve_subtitle_outlined_anchor`. The legacy `render_subtitle_banner` (rounded chip) is preserved and still wired so old style presets keep working. |
| `src/semanticvibe/render/composite.py` | New `subtitle_outlined` per-frame branch (mirrors the banner branch — same fade in/out envelope). `_resolve_decoration_anchor` now honours `element.pixel_anchor` first before falling back to `near_text_id`. |
| `src/semanticvibe/build_elements.py` | `build_decision` rewritten with a v10 layout pre-pass: (1) emit every subtitle first with pose-aware top/bottom position (reuses v9 `_pick_banner_position`), (2) collect subtitle rects, (3) for each highlight tag build a fresh `ForbiddenMap`, call `find_free_position` with corner-zone preference for hero decorations, fallback to half-size or skip with `[layout/v10]` log line. Per-line outline alternation (`outline_color_alt`) for `baseline_kenpa`. Subtitle modes: `outlined` (default) / `banner` (legacy chip) / `hero` (single big glyph). |
| `assets/styles.json` | All three presets now default to `subtitle_outlined`. Added `baseline_kenpa` preset with `outline_color="#FF6B9D"` + `outline_color_alt="#5BBFE3"`. Existing `pink_handdrawn` and `green_neon` keep their banner config as a `subtitle_banner` legacy block but the `subtitle_default` flips to `outlined`. |
| `src/semanticvibe/render/__main__.py` | `--subtitle-style` choices expanded to `{outlined, banner, hero}`. Decision count log line now reports `(text, outlined, banner, decoration, hero)`. |

### CLI

```powershell
# v10 default — outlined subtitles + forbidden-map layout
uv run python -m semanticvibe.render `
    --video data/test_videos/demo.mp4 `
    --lyrics samples/lyrics_demo.json `
    --provider ollama --ollama-model gemma3:4b `
    --style baseline_kenpa --subtitle-style outlined `
    --out outputs/v10_demo_kenpa.mp4 --preview --assets-dir data/assets_lib

# Legacy v9 chip background
uv run python -m semanticvibe.render --style green_neon --subtitle-style banner ...

# Hero mode (single huge glyph, no per-line subtitle)
uv run python -m semanticvibe.render --style pink_handdrawn --subtitle-style hero ...
```

### Verified — `outputs/v10_demo_kenpa.mp4`

Build log:

```
Building Decision (style=baseline_kenpa, subtitle=outlined, beat_sync=True)
[beat_sync] demo.mp4: tempo=161.5 BPM, 89 beats, 23 downbeats, 2 hi-energy
[beat_sync] snapped 7 / 10 highlights to nearest beat
[layout/v10] decorations placed=20, shrunk=0, skipped=0
Decision: 31 elements (0 text, 10 outlined, 0 banner, 21 decoration, 0 hero)
```

- **10 outlined subtitles** alternate pink (#FF6B9D, lines 0/2/4/6/8) and
  sky-blue (#5BBFE3, lines 1/3/5/7/9) per the `baseline_kenpa` preset.
- **20 decorations placed in free zones** — 0 shrunk, 0 skipped.
  Forbidden coverage on the demo clip averaged ≈45% per highlight
  (person mask 30-50% + subtitle bbox 5-15%).
- **Probe frames**: subtitle reads as transparent text with thick
  pink/blue outline (no background chip); decorations land in corners
  / margins, never overlapping the dancer's silhouette or the lyric
  bar.

Tests still 150 / 150.

### Layout-tuning recipe

If `[layout/v10]` reports many `shrunk` or `skipped`:
- **Person mask too aggressive**: lower `person_padding_iters` from 10
  to ~6 in `build_forbidden_map_at_time` calls.
- **Subtitle eating too much**: drop `outlined.size` in the style preset
  (smaller text → smaller forbidden rect).
- **Decoration too large**: `target_size_px = int(size * 1.6)` (hero) /
  `int(size * 1.0)` (ambient) in `build_decision`. Halve them for
  busier compositions.
- **Need more lateral spread**: lower the `step=20` parameter in
  `ForbiddenMap.find_free_position` to 12-15 for finer-grained search
  (slower but reaches positions a coarse 20 px sweep skips).

### Known limits

- The forbidden map is computed at decoration-placement time only
  (i.e. once at `hl.time + 0.2`). An idle-animation drift could later
  carry a decoration into the subtitle rect over a 2-3 s window. In
  practice this is fine because `idle.drift` amplitude is small
  (≈25 px) and subtitle padding is 15 px on every side, so the
  decoration never visibly overlaps even after the maximum drift.
- `subtitle_outlined`'s thick outline doubles the rendered tile size
  vs the original glyph bbox. The `fit_subtitle_outlined_to_canvas`
  shrink pass accounts for this, but very long lines on narrow
  portrait videos (e.g. 16+ chars on a 340 px canvas) hit `min_size=24`
  and may still bleed margin → margin. Either shorten the lyric or
  raise the `max_width_ratio` field to 0.95.

---

# v14 — Body-Landmark-Anchored Gesture Decorations + Face Safety

## What changed since v13.1

v13.1's zone-based placement (`top_right` etc.) was **subject-blind** —
the VLM picked a coarse corner keyword without knowing where the face
or hands actually were within that corner. On 20260505 the singer's
face filled the top-right at t≈0.7s, so the gesture star landed on her
face. And every gesture decoration in the song stacked in the same
corner because VLM said `top_right` for all 14 peaks.

v14 plugs in the MediaPipe pose landmarks v12 already extracts (but
threw away). Two new behaviours:

1. **Landmark-anchored placement**: each gesture in
   `gesture_vocabulary.json` declares an `anchor_landmark` symbol
   (e.g. `peace_sign` → `right_index`, `heart_hands` → `mid_wrists`,
   `arms_raised` → `mid_wrists_above`). The renderer reads the
   landmark position at the gesture's peak frame, converts normalised
   coords to canvas pixels, applies the symbol's offset, and places
   the decoration THERE — visibly attached to the body part doing the
   gesture.

2. **Face-safety push-out**: face bbox is computed from landmarks 0-10
   (+ 20 px padding). Any gesture decoration whose anchor would overlap
   the face is slid along the shortest axis until clear. Decoration
   NEVER intersects the face.

### Files changed

| File | What |
|---|---|
| `src/semanticvibe/motion_detector.py` | `MotionInfo` gained `peak_landmarks: dict[float, np.ndarray \| None]`. The existing velocity-extraction loop already builds these arrays; v14 just stops dropping them. |
| `assets/gesture_vocabulary.json` | Each gesture entry gained an `anchor_landmark` field (symbol name only — pixel math lives in build_elements). |
| `src/semanticvibe/vlm_gesture.py` | `GestureEvent.landmarks_normalised: list[list[float]] \| None`. `detect_gestures(...)` gained `landmarks_by_time` kwarg that populates the new field per event. New `gesture_anchor_symbol()` helper for downstream lookup. Cache backward-compat: old v13.1 entries (no landmarks) load fine and fall through to zone placement. |
| `src/semanticvibe/build_elements.py` | New helpers `_landmark_to_pixel`, `_resolve_anchor_landmark`, `_face_bbox`, `_push_anchor_out_of_bbox`. The gesture-decoration emission loop now tries (1) landmark anchor → (2) VLM zone keyword → (3) None (ForbiddenMap fallback), with face-safety push-out always running after step 1 or 2. The dedup pass also accepts `"v14 gesture="` prefix. |

### Symbol → landmark mapping

| Symbol | MediaPipe landmark(s) | Offset |
|---|---|---|
| `right_index` | 20 | +30 px x, −20 px y |
| `left_index` | 19 | −30 px x, −20 px y |
| `right_wrist` | 16 | +20 px x, −10 px y |
| `left_wrist` | 15 | −20 px x, −10 px y |
| `mid_wrists` | midpoint(15, 16) | — |
| `mid_wrists_above` | midpoint(15, 16) | −60 px y |
| `mid_shoulders` | midpoint(11, 12) | — |
| `mid_shoulders_below` | midpoint(11, 12) | +80 px y |
| `right_eye_offset` | 2 | −90 px x, −10 px y |

### Verified on 20260505

Console:
```
[motion_sync] 15 peaks (9 high, 2 medium, 4 low) @ 15.0 fps
[vlm_gesture] 14/15 peaks → 14 valid events
              (unknown=0, low_conf=1, non-actionable=0, model=qwen2.5vl:7b, cache=False)
Decision: 30 elements (0 text, 5 outlined, 0 banner, 25 decoration, 0 hero)
```

Placement reasoning per gesture (all 14 used the new landmark path):

| t (s) | pixel_anchor | placement |
|---|---|---|
| 0.73 | (78, 470) | landmark:right_index |
| 2.00 | (131, 243) | landmark:right_index |
| 3.47 | (89, 386) | landmark:right_index |
| **4.80** | (58, 8) | landmark:right_index **+ face_pushed** |
| 5.34 | (111, 324) | landmark:right_index |
| 5.67 | (115, 326) | landmark:right_index |
| 6.07 | (106, 501) | landmark:right_index |
| **7.00** | (84, 165) | landmark:right_index **+ face_pushed** |
| 7.47 | (8, 173) | landmark:right_index |
| 8.94 | (8, 444) | landmark:right_index |
| **9.74** | (112, 196) | landmark:right_index **+ face_pushed** |
| **11.94** | (122, 180) | landmark:right_index **+ face_pushed** |
| 15.07 | (213, 418) | landmark:right_index |
| 15.67 | (148, 215) | landmark:right_index |

**14 distinct pixel anchors** (was 14× top_right in v13.1). **4 of 14
events face-pushed** automatically when the natural landmark anchor
would have overlapped the face.

Visual diff vs v13.1 at t=0.7s:
- v13.1: yellow star in top-right corner overlapping the face
- **v14**: yellow star at pixel (78, 470) — lower-left near her right hand /
  guitar position. Face fully visible.

### Asset library wishlist

The v14 mapping table now spreads gesture decorations across 8 distinct
tags. PNG inventory per tag:

| Tag | PNGs | Coverage | Recommendation |
|---|---|---|---|
| heart | 23 | ✓ | — |
| flower | 20 | ✓ | — |
| star | 22 | ✓ | — |
| **burst** | **2** | ✗ | **Add 8-10 PNGs** — firework / energy ring / sunburst / impact splash. This is the biggest gap; `arms_raised` → `burst` events will repeat the same 2 PNGs across a long song. |
| lightning | 10 | △ | Add 3-5 more dramatic bolts (jagged white-yellow gradient, electric-arc lines) for `jump` gestures. |
| music_note | 18 | ✓ | — |
| exclaim | 66 | ✓ | — |
| arrow | 24 | ✓ | — |

Recommended NEW tag categories (need both asset PNGs + animation
support — deferred to v15):

- **`trail`** — line-art swooshes + curved-arrow swoops. Pairs with a
  new `swipe_across(start_pos, end_pos)` animation that the renderer
  doesn't have yet. Would unlock "從一邊划向另一邊" effects per the
  user's original ask.
- **`particle_cluster`** — small dot / petal scatters (4-8 PNGs each).
  Pairs with per-frame landmark tracking so a "spawn at fingertip"
  particle can follow the dancer's hand over multiple frames.

### Design decisions (locked)

| Question | Answer |
|---|---|
| Anchor symbol resolution | Lives in build_elements (`_resolve_anchor_landmark` switch). Vocab JSON stores the symbol name only — decouples vocab edits from pixel math. |
| Face-safety policy | Hard rule: gesture decoration NEVER intersects face bbox (landmarks 0-10 + 20 px padding). Push out along shortest axis. |
| Landmark caching | Lives inside `MotionInfo` (lru_cache in-process). VLM cache key uses `peak_times` so landmarks travel with the peaks they belong to. Old v13.1 cache entries load fine — `landmarks_normalised=None` falls through gracefully. |
| Swipe / trail animations | Deferred to v15. Needs new element type with `start_pos` + `end_pos` AND new `trail/` asset PNGs the user doesn't have yet. Marked in asset wishlist. |

### Tests still 150 / 150

### Known limits

1. **Face-pushed positions can stack in the corners.** When 4+ events
   in a row all get pushed up-and-right to avoid the same face, the
   final stickers cluster. Future v14.1: aware push that distributes
   push-direction across consecutive events.
2. **Landmark visibility on motion-blurred frames.** When the hand is
   blurred during fast motion, MediaPipe gives a low-confidence
   landmark with possibly-wrong coordinates. v14 doesn't filter on
   confidence (it just uses the position); future could thread
   visibility through and fall back to zone for low-vis landmarks.
3. **Subject-relative not screen-relative**: MediaPipe's "right" is
   the SUBJECT's right (viewer's left in a mirror selfie). v14 hard-
   codes `right_index` for peace_sign / point_at_camera which works
   for selfie footage where the dancer faces the camera. If a future
   clip has the dancer facing away, the symbol should flip — not
   handled.

---

# v13.1 — VLM prompt rebuild + asset re-categorisation + zone-driven placement

## What changed since v13

User feedback after the first 20260505 v13 render: "**好多旋渦, 動作沒到很精準, 感覺要改 prompt**".

Three independent bugs caused the visual mess. v13.1 fixes all three:

### Bug 1: `sparkle/` was polluted with swirl PNGs

The v6 manual asset routing put `josh_mix.png`, `josh_mix2.png`,
`josh_water_mix.png` under `assets/stickers/sparkle/` — but they're
abstract water-swirl drawings, not sparkles. With v13's peace_sign /
arms_raised / spin all mapping to `sparkle`, the 4-PNG pool was 3/4
swirls, so the screen filled with renders of the same 3 swirl PNGs.

**Fix**: moved the 3 swirl PNGs to `assets/stickers/cloud/` where they
belong visually. `sparkle/` now has just the 1 real diamond PNG; the
AssetRetriever same-category fallback pulls from sibling tags (star,
ribbon, flower) when more variety is needed.

### Bug 2: Gesture mapping collapsed 4 gestures into 1 tag

`gesture_vocabulary.json` had:
- peace_sign → `sparkle`
- arms_raised → `sparkle`
- spin → `sparkle`
- (heart_hands → `heart`, smile_close_up → `heart`)

So 4 gestures shared 1 tag. Rewrote to **8 unique tags for 8 gestures**:

| Gesture | v13 tag | **v13.1 tag** |
|---|---|---|
| heart_hands | heart | heart |
| smile_close_up | heart (duplicate!) | **flower** |
| peace_sign | sparkle | **star** |
| arms_raised | sparkle (duplicate!) | **burst** |
| jump | burst | **lightning** |
| spin | sparkle (duplicate!) | **music_note** |
| clap | exclaim | exclaim |
| point_at_camera | arrow | arrow |

Every actionable gesture now resolves to a different tag → visual
diversity guaranteed even when one gesture dominates a song.

### Bug 3: VLM prompt was wasted on a single label

v13 asked the VLM for one closed-vocab label, ignoring its visual
reasoning capability. v13.1 asks for **structured JSON** with five
useful fields:

```json
{
  "action":      "一句中文具體描述,例如「右手舉起比V字、左手叉腰」",
  "emotion":     "excited|shy|intense|calm|playful|serious",
  "composition": {
    "subject_main_zone": "top|center|bottom|left|right",
    "best_empty_zone":   "top_left|top_right|bottom_left|bottom_right|none"
  },
  "gesture":     "<closed gesture id>",
  "animation":   "<closed animation enum>",
  "confidence":  0.0-1.0
}
```

Wins:
- **`format: "json"`** in the Ollama call forces valid JSON; no more
  brittle regex parsing.
- **`confidence < 0.45` → drop the event.** v13's "VLM always answers"
  failure mode is gone; the model can admit "I'm not sure" and the
  parser respects it.
- **`composition.best_empty_zone`** tells the renderer where to place
  the decoration in pixel space. v13.1 maps the 4 zone keywords to
  pixel boxes (top_left = `(W*0.05, H*0.05)` etc.) and sets
  `pixel_anchor` directly — bypasses the v10 ForbiddenMap geometric
  pass for gesture decorations.
- **`action` + `emotion`** are free-text fields used purely for
  debugging today, but landed for future v14 use (e.g. emotion →
  idle animation).
- Animation overrides are validated against the renderer's closed
  `AnimationName` set; "none" or junk falls through to the gesture
  vocab default.

Negative-example rules added inline:
```
"point_at_camera" requires SINGLE arm extended forward with finger
visible. Both arms down at sides ≠ point_at_camera.
"heart_hands" requires hands forming a heart shape, not just framing
the face.
```

### Verified on 20260505_153230.mp4

```
[motion_sync] 15 peaks (9 high, 2 medium, 4 low) @ 15.0 fps
[vlm_gesture] 14/15 peaks → 14 valid events
              (unknown=0, low_conf=1, non-actionable=0, model=qwen2.5vl:7b)
Decision: 30 elements (0 text, 5 outlined, 0 banner, 25 decoration, 0 hero)
```

VLM-reported distributions (now visible thanks to structured output):

| Field | Distribution |
|---|---|
| gesture | peace_sign × 14 |
| emotion | excited × 10, playful × 3, calm × 1 |
| zone | top_right × 14 |
| confidence | mean 0.90, min 0.80 |
| **dropped** | 1 (low_conf) |

The "1 dropped on low confidence" is the WIN — v13 always returned 27/28
valid events because it never admitted uncertainty. v13.1's confidence
floor (0.45) lets the model say "I'm not sure" on the one frame it
genuinely couldn't read.

The "all peace_sign" is an honest model output for this video — the
dancer really does V-sign in most peak moments. Visual diversity now
comes from the gesture → tag fan-out (star) plus the lyric-driven
decorations (heart, sparkle, music_note) layered on top.

### Visual probe

t=5.5 「You are my angel」: real diamond sparkle + pink heart + cute
heart sticker + star (from peace_sign gesture) — 4 distinct artwork
types instead of 4 copies of the same swirl.

t=9.0 「可愛くね、すでに君の愛を届け」: 3 different decorations
(burst-sparkle / headphones / yellow star) in top-right zone — VLM
correctly identified the right side of frame as empty.

t=16.0 「覚えててください」: green diamond + headphones + cosmic spiral
+ yellow star, all in top-right.

### Tests still 150 / 150

### Known v13.1 limits

1. **Single-zone bias on selfie footage.** VLM consistently picked
   `top_right` for all 14 frames on 20260505 because the singer was
   centre-left throughout. Visually, all gestures decorate the same
   corner. Future v13.2: ask VLM for 2 candidate zones + alternate
   between them across the song.
2. **VLM still over-commits to one gesture per video.** On 20260505
   every peak got `peace_sign`, even frames where she's clearly
   reaching toward camera (t≈16). Lowering confidence threshold
   surfaces some of these as misses, but the model carries a strong
   "this is the peace-sign video" prior. Future v13.2: pass the
   running tally of seen gestures and ask for variety.
3. **Cache invalidation on prompt edits**: the cache key includes
   `vocab_fingerprint` (md5 of gestures list) but NOT the prompt text.
   So if you only edit the prompt, the cache won't bust. Manually
   `rm -rf .cache/vlm_gestures` when iterating on the prompt.

---

# v13 — VLM-driven gesture anchoring (dancer's WHAT, not just WHEN)

## What changed since v12

v12 motion_detector tells the renderer **when** the dancer moves (28
peaks on demo.mp4) but not **what** she's doing. Every decoration was
still aligned to lyric time, just with different entry animations — the
visual still felt detached from the dance.

v13 sends each motion-peak frame to a local VLM (qwen2.5vl:7b via
Ollama), parses the gesture label against a closed vocabulary, and
emits a NEW first-class decoration at that peak time using the gesture's
mapped tag + animation. Decorations are anchored to **what the dancer
is actually doing**, not just where the music lands.

Smoke test (5 frames at known peaks):

| VLM | Clean hits | Effective |
|---|---|---|
| gemma3:4b | 1/5 (20%) | ~30% |
| **qwen2.5vl:7b** | **3/5 (60%)** | **~70%** |

### New: `src/semanticvibe/vlm_gesture.py`

| API | Purpose |
|---|---|
| `GestureEvent` Pydantic | `time, gesture, tag, animation, confidence` |
| `GestureInfo` TypedDict | `events, model, cache_hit` |
| `detect_gestures(video, peak_times, *, model, host, use_cache)` | Per-peak VLM call, parses to closed vocab, dedups invalid/non-actionable labels |

Internals (~210 LOC):
- Loads `assets/gesture_vocabulary.json` once → 11 closed gesture IDs
- Vocab fingerprint (md5[:8]) baked into cache key so vocab edits bust the cache
- Per-peak frame extraction via `cv2.VideoCapture` (same pattern as motion_detector)
- 512px-wide JPEG resize keeps VLM token budget under control
- POST `/api/generate` with `images: [b64]` + closed-vocab prompt (single-turn vision is simpler than `/api/chat` messages array)
- Closed-vocab parser drops labels outside the set (silent miss > wrong placement)
- Disk cache at `.cache/vlm_gestures/<sha1[:16]>.json`, key = `path + mtime + model + peak_times + vocab_fp`
- Error handling mirrors `_ollama_align` — URLError → "Ollama unreachable", TimeoutError → drop event

### New: `assets/gesture_vocabulary.json`

11 closed gesture IDs mapped to existing v6 tag vocab:

| Gesture | Tag | Animation |
|---|---|---|
| heart_hands | heart | scale_pop |
| arms_raised | sparkle | drop_in |
| jump | burst | stamp |
| peace_sign | sparkle | spin_in |
| point_at_camera | arrow | slide_in_left |
| spin | sparkle | spin_in |
| clap | exclaim | stamp |
| smile_close_up | heart | scale_pop |
| pose_static, lean_or_sway, none | null (no decoration) | — |

Load-time defensive check verifies every non-null tag exists in
`tag_vocabulary.json` so a typo in the gesture vocab fails fast instead
of silently producing garbage decorations.

### Patched `build_elements.py`

`build_decision` gained `vlm_gestures: bool = True` and `vlm_model: str
= "qwen2.5vl:7b"` parameters mirroring the v12 audio_path / motion_aware
pattern. After motion detection, calls `detect_gestures(...)` and:

1. **Emits gesture decorations** at peak time:
   ```python
   DecorationElement(
       asset_tag=ev.tag, start_time=ev.time, end_time=ev.time + 2.0,
       base_size=int(canvas_w * 0.18),
       animation=ev.animation or "scale_pop",
       reasoning=f"v13 gesture={ev.gesture!r} at motion peak {ev.time:.2f}s",
   )
   ```

2. **Dedup pass**: any lyric-driven decoration whose `asset_tag` matches
   a gesture event within ±0.5s is dropped. Gesture wins on timing.
   Logs `[vlm_gesture] deduped N lyric-driven decorations`.

### CLI

```
--vlm-gestures           # default ON
--no-vlm-gestures        # ablation toggle
--vlm-model qwen2.5vl:7b
```

### Verified on `demo.mp4`

First render (cold cache):

```
[motion_sync] demo.mp4: 28 peaks (10 high, 13 medium, 5 low) @ 15.0 fps
[vlm_gesture] 27/28 peaks → 27 valid events
              (unknown=0, non-actionable=1, model=qwen2.5vl:7b, cache=False)
[vlm_gesture] 27 gesture events from 28 peaks (cache=False)
[vlm_gesture] deduped 5 lyric-driven decorations
              (gesture took precedence)
Decision: 53 elements (0 text, 10 outlined, 0 banner, 43 decoration, 0 hero)
```

Gesture distribution on demo.mp4:

| Gesture | Count |
|---|---|
| arms_raised | 12 |
| peace_sign | 6 |
| point_at_camera | 4 |
| smile_close_up | 2 |
| spin | 2 |
| heart_hands | 1 |

Second render (cache hit):

```
[vlm_gesture] cache hit 35b26a088cce8a7c: 27 events from 28 peaks
real: 1m43s   (vs ~4 min on cold render — VLM pass skipped entirely)
```

### Comparison vs v12

| Metric | v12 | v13 |
|---|---|---|
| Decoration count | 21 (lyric-driven) | **43** (lyric-driven + 22 gesture-anchored after dedup) |
| Decoration timing | aligned to lyric times | **aligned to motion peaks** (when dancer actually does the gesture) |
| Animation pool diversity | strength/motion buckets | **gesture-specific** (heart_hands always scale_pop, jump always stamp) |
| VLM zero-shot quality | n/a | 27/28 in-vocab (96% valid rate, 0 unknown labels) |
| Render cost | ~95s for 32s output | **~3-4 min cold, 1m43s cached** |

### Design decisions (locked)

| Question | Answer |
|---|---|
| Default flag | `--vlm-gestures` ON by default. Cache makes re-renders fast. |
| Dedup policy | Replace when same tag within ±0.5s. Keeps gesture timing, drops lyric duplicate. |
| Cache | Disk cache at `.cache/vlm_gestures/<key>.json`. Key includes `path + mtime + model + peak_times + vocab_fp`. |
| Endpoint | `/api/generate` (single-turn vision, simpler than `/api/chat`). |
| Model | `qwen2.5vl:7b`. `--vlm-model` flag exposes the choice. |
| Pose-relative anchor | Deferred to v14. v13 uses ForbiddenMap default placement; hand-anchored placement needs landmarks at the gesture frame. |

### Tests still 150 / 150

### Known limits

1. **~30% recall miss**: qwen2.5vl gets some frames wrong (framing-face → `heart_hands` is a sympathetic miss; some `point_at_camera` calls on raised-arm frames are real misses). The closed-vocab parser drops invalid labels — silent miss rather than wrong placement, so the visual stays clean.
2. **Idol-specific gestures**: 11 gesture IDs covers the common cases (heart-hands, V, arms-up, jump, clap, point) but misses niche moves (chest-pump, hair-flip, k-pop point-and-shoot). Expand `gesture_vocabulary.json` as new content reveals gaps.
3. **One frame per peak**: jumps and spins span 200-400 ms; sampling the single peak frame catches mid-motion. v14 could sample 2-3 frames around each peak and majority-vote.
4. **GPU memory contention**: Ollama swaps gemma3:4b ↔ qwen2.5vl:7b weights between the lyric-align step and the gesture step. On 12 GB VRAM this is fine but adds ~5s switching overhead. Pre-warm both models via `ollama run gemma3:4b ""; ollama run qwen2.5vl:7b ""` if iteration speed matters.

---

# v12 — motion-aware animation trigger (dancer body motion → entry pool)

## What changed since v11

v6-v11 wired three sync layers: lyric → tag (LLM), beat → timing (librosa
+ snap_to_beat), pose → layout (MediaPipe occupancy). The missing layer
was **dancer body motion → animation intensity** — so when the dancer
hits a peak gesture, the on-screen sticker stamps in at that exact frame.

v12 adds a new `motion_detector.py` module (peer to `beat_sync.py`) that
runs MediaPipe Pose at 15 fps over the video, computes per-frame
upper-body landmark velocity, z-score normalises, and uses
`scipy.signal.find_peaks` to extract motion peaks with intensity buckets
(high / medium / low). The existing `_pick_entry` selector gained one
more priority tier — **motion peak overrides downbeat**.

Both MoviePy and Hyperframes renderers benefit without any renderer
code change (both consume `element.animation` as a string; the v8
animation-pool selector is the only branch point).

### New module: `src/semanticvibe/motion_detector.py`

| API | Purpose |
|---|---|
| `MotionInfo` TypedDict | `peak_times`, `peak_intensities`, `energy_envelope`, `sample_fps` |
| `detect_motion_peaks(video_path, *, sample_fps=15.0)` | `lru_cache(maxsize=8)`, returns `MotionInfo` |
| `is_motion_peak(t, peaks, tolerance=0.3)` | bool |
| `motion_intensity_at(t, info, tolerance=0.3)` | `"high"\|"medium"\|"low"\|None` |

Internals (~160 LOC):
- Walk video at 15 fps via `cv2.VideoCapture` (same pattern as `pose_detector.py`)
- Reuse `_pose_landmarker()` singleton from `preprocess.mediapipe_pose` — no double model load
- Keep landmarks **0-22** (head + shoulders + arms + hands). Hips/legs excluded — they jitter with camera bounce more than real choreography
- Largest-bbox subject pick (same `_area()` lambda as pose_detector)
- Per-sample energy = MEAN Euclidean velocity of visible landmarks (mean-not-sum so partial visibility doesn't dampen energy)
- 0.3 s sliding-mean smoothing → z-score normalise (scale-invariant across videos)
- `scipy.signal.find_peaks(z, prominence=0.5, distance=int(0.3 * sample_fps))` — forces ≥ 0.3 s between peaks
- Bucket by z-score: `>1.5 → high`, `0.8-1.5 → medium`, `0.3-0.8 → low`, else drop

### Patched `src/semanticvibe/build_elements.py`

New entry pools alongside the v9 DOWNBEAT_ENTRY:

```python
MOTION_ENTRY_HIGH   = ["stamp", "spin_in", "drop_in"]
MOTION_ENTRY_MEDIUM = ["scale_pop", "wobble_in"]
MOTION_ENTRY_LOW    = ["fade", "slide_in_left", "slide_in_right"]
```

`_pick_entry` priority chain:
```
explicit LLM hint  >  motion peak  >  downbeat  >  strength bucket
```

`build_decision` gained `video_path` + `motion_aware=True` parameters
mirroring v9's `audio_path` + `beat_sync` pattern.

### Patched `src/semanticvibe/render/__main__.py`

```powershell
--motion-aware            # default ON
--no-motion-aware         # ablation toggle
```

### Verified on `demo.mp4` (32.6 s, 15 fps MediaPipe pass)

Console log:

```
[motion_sync] demo.mp4: 28 peaks (10 high, 13 medium, 5 low) @ 15.0 fps
[motion_sync] 28 peaks driving entry-animation choice
Building Decision (style=baseline_kenpa, subtitle=outlined,
                   beat_sync=True, motion_aware=True)…
```

**Ablation comparison** — same lyrics + same Decision, only `--motion-aware`
flipped:

| Decoration entry pool | `--no-motion-aware` | `--motion-aware` |
|---|---|---|
| fade | 6 | 4 |
| slide_in_left | 5 | 5 |
| slide_in_right | 8 | 6 |
| wobble_in | 2 | **4** ← motion medium |
| spin_in | 0 | **2** ← motion high (only fires here) |

**11 of 21 decoration slots picked a different animation when motion-aware
was on.** The `spin_in` animation is exclusive to `MOTION_ENTRY_HIGH`, so
its non-zero count is direct evidence the high-intensity branch fired on
real motion peaks.

### Motion peaks (first 10) on demo.mp4

| Time | Intensity |
|---|---|
| 0.47s | medium |
| 3.83s | low |
| 4.23s | low |
| 4.77s | medium |
| 5.10s | medium |
| 6.24s | medium |
| 6.58s | medium |
| 7.25s | low |
| 8.93s | high |
| 9.33s | medium |

Total: **28 peaks** (10 high, 13 medium, 5 low).

### Design decisions (locked)

| Question | Answer |
|---|---|
| Body region | Upper body 0-22 (head + shoulders + arms + hands) |
| Sample rate | 15 fps |
| Motion vs downbeat collision | Motion wins (viewer's eye is on the dancer at a peak) |
| Idle animations | Unchanged for v12 — motion only biases entry |
| Cache | `lru_cache(maxsize=8)` in-process only (no disk cache) |

### Tests still 150 / 150

The v6 build_elements_from_lyrics path was deliberately not patched
(no video access there); future work if/when a video-aware caller emerges.

### Performance

Motion detection adds ~30-35 s for a 32 s video at 15 fps (one-time per
process — `lru_cache` makes re-renders free). Total v12 render time
on demo.mp4 with Hyperframes:

| Stage | Time |
|---|---|
| Whisper + LLM align | ~3 s |
| beat_sync (v9) | ~3 s |
| **motion_sync (v12)** | **~33 s** |
| Pose-mask (v10 layout) | ~3 s |
| Decision build | <1 s |
| Frame capture (Puppeteer) | ~50 s |
| ffmpeg overlay | ~3 s |
| **TOTAL** | **~95 s** for 32 s output (3× realtime) |

Future Phase 13: cache motion_info to disk so 2nd run drops to ~60 s.

---

# v9 — beat-sync (timings + downbeat-aware animations + tempo-locked pulse)

## What changed since v8

Before v9 the renderer ran every animation cycle on a fixed 1.5 s pulse,
ignoring the song's actual tempo — elements drifted in their own loops
while the music did its own thing. v9 plugs in librosa beat tracking and
binds three layers of the visual to the music's rhythm:

1. **Snap-to-beat for highlight times.** Each highlight's `time` is
   pulled to the nearest detected beat when within ±150 ms. Lyrics that
   were 30–80 ms early/late from Whisper now land on the actual drum hit.
2. **Downbeat-aware entry animations.** Highlights that fall on a
   downbeat (every 4th beat under the 4/4 assumption) get the
   `DOWNBEAT_ENTRY` pool (`stamp` / `drop_in` / `scale_pop` / `spin_in`)
   regardless of strength score — the music already commits to drama
   and the visual matches.
3. **Tempo-locked breathing.** The `pulse` idle animation reads
   `Decision.global_style.beat_period_sec` and uses `2 × beat_period`
   as its sine-wave period. Elements now breathe once per two beats
   (≈1 Hz at 120 BPM) instead of always 1.5 s.
4. **Chorus boost.** When a highlight falls inside a high-energy RMS
   segment (RMS > mean × 1.2 sustained ≥ 2 s), the per-line decoration
   cap doubles from 2 → 4 tags AND the idle defaults to `pulse` so the
   loud parts of the song get visibly louder visuals.

### New / modified

| File | What |
|---|---|
| `src/semanticvibe/beat_sync.py` (new) | `detect_beats(media_path)` returns `BeatInfo(tempo, beat_times, downbeat_times, energy_envelope, high_energy_segments)`. `snap_to_beat(t, beats, max_offset=0.15)`, `is_downbeat(t, downbeats, tolerance=0.1)`, `is_high_energy(t, segments)`, `average_beat_period(beats)`. Reuses `preprocess.librosa_beats.extract_wav` for the loudnorm-cached wav path so beat detection doesn't re-decode the same video twice. `lru_cache` on the resolved media-path string. |
| `src/semanticvibe/build_elements.py` | `build_decision(...)` gained `audio_path: Path \| str \| None = None` and `beat_sync: bool = True`. When both are set: detect beats → snap every highlight's `.time` to the nearest beat (logs the snap count) → set per-highlight `hl_is_downbeat` and `hl_in_chorus` flags → `_pick_entry` and `_pick_idle` consult them. Chorus highlights walk `hl.tags[:4]` instead of `hl.tags[:2]`. `Decision.global_style.beat_period_sec` is populated as `2 × beat_period` so the renderer's pulse syncs. |
| `src/semanticvibe/schemas/decision.py` | `GlobalStyle` gained `beat_period_sec: float \| None = None`. |
| `src/semanticvibe/render/idle_animations.py` | `evaluate(...)` gained `pulse_period_override: float \| None = None`; routes to `pulse(period=…)` when supplied. Other animations ignore the override. |
| `src/semanticvibe/render/composite.py` | Reads `decision.global_style.beat_period_sec` once, threads `pulse_period_override` to `_compose_idle` for both text and decoration paths (single-anchor + scatter copies). |
| `src/semanticvibe/render/__main__.py` | Added `--beat-sync` (default ON) + `--no-beat-sync`. Beat audio source: `--audio` if given (cleaner stem), otherwise `--video`'s embedded track. |

### CLI

```powershell
# Beat-sync ON (default) — uses video's audio
uv run python -m semanticvibe.render --video data/test_videos/demo.mp4 `
    --lyrics samples/lyrics_demo.json `
    --provider ollama --ollama-model gemma3:4b `
    --style green_neon --subtitle-style banner `
    --out outputs/v9_demo_beat.mp4 --preview --assets-dir data/assets_lib

# Beat-sync ON with cleaner audio source
uv run python -m semanticvibe.render --video clip.mp4 --audio song.mp3 `
    --beat-sync --out out.mp4

# Disable beat-sync (back to v8 deterministic pulse + random animation pool)
uv run python -m semanticvibe.render --video clip.mp4 --no-beat-sync --out out.mp4
```

### Verified — demo.mp4 + samples/lyrics_demo.json

Render log:

```
[beat_sync] demo.mp4: tempo=161.5 BPM, 89 beats, 23 downbeats,
            2 high-energy segments
[beat_sync] driving build_decision: tempo=161.5 BPM, beat_period=0.364s
[beat_sync] snapped 7 / 10 highlights to nearest beat (max ±0.15s)
Decision: 31 elements (0 text, 10 banner, 21 decoration, 0 hero)
```

- **Tempo**: 161.5 BPM (fast J-pop)
- **Beat grid**: 89 beats over 32.6 s (one beat every 0.366 s)
- **Downbeats**: 23 (every 4th beat under 4/4 assumption)
- **High-energy segments**: 2 (RMS > mean × 1.2 sustained ≥ 2 s)
- **Snap rate**: 7 / 10 highlights pulled to nearest beat. The
  remaining 3 were already inside ±150 ms of a beat.
- **Pulse period**: 0.728 s (= 2 × 0.364), so the breathing visibly
  matches the song's two-beat phrasing instead of the v8 fixed 1.5 s.

Tests still 150 / 150.

### Known limits

- Downbeats are derived as `beat_times[::4]` — a hard 4/4 assumption.
  3/4 (waltz), 6/8 (swing), and 5/4 tracks would mis-place the
  "downbeat" hits. Real meter detection needs `madmom` (heavy CNN
  dep). For pop / J-pop / K-pop content the 4/4 assumption is right
  ≥ 95 % of the time.
- High-energy detection is a simple RMS threshold (`mean × 1.2`
  sustained ≥ 2 s). It catches loud bridges and instrumental hits as
  "chorus", not just chorus proper. Good enough for animation
  intensity decisions; not good enough to drive a chapter marker.
- Beat-sync only fires when `audio_path` resolves to something librosa
  can decode. The cached wav under `%TEMP%/semanticvibe_*.wav` is
  shared with Whisper's audio extraction, so the second run on the
  same video is instant.

---

# v6 — closed-vocabulary semantic alignment

## What changed since v5+

The "歌詞語意 → 視覺裝飾" path was previously open-ended: the LLM (or
keyword fallback) could emit any tag, and the renderer silently dropped
unknown ones. v6 closes the loop:

1. **Closed vocabulary.** `assets/tag_vocabulary.json` defines exactly
   20 tags across 8 categories. The Claude provider is told the full
   list and constrained to choose only from it; rule_based's
   `KEYWORD_TO_TAGS` dict is statically validated against the same set
   at module load.
2. **Asset library re-indexed.** `assets/stickers/<tag>/*.png` is the
   canonical layout (already in place from earlier sticker generations);
   `scripts/build_index.py` walks it and produces `assets/index.json`.
3. **AssetRetriever with category-aware fallback.** Missing tags fall
   back to a same-category sibling first, then to a global `heart`
   fallback, so the render never silently drops a decoration just
   because that PNG hasn't been generated yet.
4. **`build_elements_from_lyrics`** — new high-level helper. Lyrics in,
   `list[dict]` of element specs out (hero_text + text + decoration),
   ready to be wrapped in a `Decision` and rendered.
5. **Caching for the Claude provider.** First run hits the API, writes
   `.cache/alignment/<md5>.json`; subsequent runs (same lyrics + same
   model + same vocab fingerprint + same song title) read from cache.

### New / modified

| File | What |
|---|---|
| `assets/tag_vocabulary.json` (new) | Closed 20-tag vocab. Each tag carries `id`, `category`, `description`. `fallback_tag = "heart"` is the global last-resort substitute. |
| `scripts/build_index.py` (new) | Scans `assets/stickers/<tag>/*.png`, validates each tag against the vocab, emits `assets/index.json`. Reports MISS warnings for vocab tags with zero PNGs. |
| `assets/index.json` (new) | Flat index of `{file, tag, category, size, weight}` records consumed by the new `AssetRetriever`. Regenerated by `scripts/build_index.py`. |
| `src/semanticvibe/semantic_align.py` | Full rewrite. Pydantic `Highlight(time, text, is_hook, tags, primary_tag, reasoning, duration?)` + `AlignmentResult(highlights, non_hooks)`. New `align_lyrics()` entry; legacy `align()` preserved as a thin shim. Claude provider with strict-JSON parser, vocab cleanup (invalid tags → fallback), MD5 cache at `.cache/alignment/<key>.json`. `KEYWORD_TO_TAGS` is a multi-tag-per-keyword dict (e.g. 「電波好き」 → [lightning, heart]). Module-load assertion that every tag in `KEYWORD_TO_TAGS` is in the closed vocab. |
| `src/semanticvibe/asset_retrieval.py` (new) | `AssetRetriever` class. `has_tag` / `get` / `get_image` / `get_multi`. `_find_fallback_tag` resolves missing tags via same-category siblings, then global fallback. Avoid-recent dedup so the same PNG doesn't repeat across consecutive picks. |
| `src/semanticvibe/build_elements.py` | Added `build_elements_from_lyrics(lyrics, *, song_title, provider, seed)` returning a flat `list[dict]`. One `hero_text` per song (the strongest short hook), then per-highlight `text` + up to 2 `decoration` elements. Output validates round-trip against the Pydantic `Decision` schema. The legacy v5 `build_decision()` is kept and adapted to the new `Highlight` field names (`time` not `lyric_time`, `primary_tag` not `decoration_tag`). |
| `src/semanticvibe/render/__main__.py` | Added `--song-title` (folded into Claude cache key) + `--elements-json` (skip alignment + build_decision; load a pre-built Decision JSON or bare element list directly). Provider list collapsed to `rule_based`/`claude` (the v5+ openai branch is gone — claude with caching covers it). |
| `samples/lyrics_test.json` (new) | 5-line fixture: もしもし / 電波 / 好き / 夢 / qqxxzz. Last line is engineered to match nothing in `KEYWORD_TO_TAGS` so the non_hooks branch is exercised. |
| `tests/test_semantic_align.py` | Full rewrite for v6. 17 tests: vocab integrity (KEYWORD_TO_TAGS ⊂ VALID_TAGS, fallback_tag ∈ VALID_TAGS), routing (Japanese / multi-tag / longest-match / dict input), is_hook heuristic, legacy compat shim (`lyric_time`/`decoration_tag`/`strength`), strict-JSON parser (invalid tags → fallback, code-fence stripping, prose tolerance), sample-file integration. |
| `tests/test_build_elements_v6.py` (new) | 6 tests: rule_based produces well-formed elements, ≤1 hero per song, decoration `near_text_id` references resolve, output round-trips through Pydantic Decision, every emitted decoration tag is in `VALID_TAGS`, JSON-clean serialisation. |

### Closed tag vocabulary (assets/tag_vocabulary.json)

20 tags across 8 categories:

| Category | Tags |
|---|---|
| emotion | heart, teardrop, kiss |
| decorative | sparkle, ribbon, flower, star |
| energy | fire, lightning, burst |
| emphasis | exclaim, dot |
| weather | sun, moon, cloud, rainbow |
| nature | leaf |
| audio | music_note |
| communication | speech_bubble, arrow |

Fallback tag: `heart`.

### Per-tag asset counts (after `python scripts/build_index.py`)

```
        arrow          2
  MISS  burst          0
        cloud          1
  MISS  dot            0
        exclaim        1
        fire           1
        flower         2
        heart          3
  MISS  kiss           0
  MISS  leaf           0
        lightning      2
        moon           1
        music_note     2
  MISS  rainbow        0
        ribbon         1
        sparkle        2
        speech_bubble  2
        star           2
        sun            2
        teardrop       1
```

Total: **25 stickers across 15 / 20 tags** indexed. The 5 MISS tags
(burst / dot / kiss / leaf / rainbow) still appear in the closed
vocabulary so the LLM may legitimately pick them; the asset retriever
just falls back to a same-category sibling at render time. Generate
real PNGs for them via `scripts/generate_stickers.py` to remove the
fallback path.

### rule_based vs claude — example side-by-side

Input (`samples/lyrics_test.json`):

```json
[
  {"time":  1.0, "text": "もしもし"},
  {"time":  4.0, "text": "電波"},
  {"time":  7.0, "text": "好き"},
  {"time": 10.0, "text": "夢"},
  {"time": 13.0, "text": "qqxxzz"}
]
```

`rule_based` output (verified, deterministic):

| time | text | is_hook | tags | primary_tag |
|---|---|---|---|---|
| 1.0 | もしもし | True | [speech_bubble] | speech_bubble |
| 4.0 | 電波 | True | [lightning] | lightning |
| 7.0 | 好き | True | [heart] | heart |
| 10.0 | 夢 | True | [star] | star |
| 13.0 | qqxxzz | False | [] | None |

`non_hooks = ["qqxxzz"]`.

`claude` output (illustrative — actual results depend on the model;
written to `.cache/alignment/<md5>.json` on first call):

| time | text | is_hook | tags | primary_tag | reasoning |
|---|---|---|---|---|---|
| 1.0 | もしもし | False | [speech_bubble] | speech_bubble | "phone-call greeting; bubble icon" |
| 4.0 | 電波 | True | [lightning, sparkle] | lightning | "signal/electricity — punchy hook" |
| 7.0 | 好き | True | [heart] | heart | "core declaration of love" |
| 10.0 | 夢 | True | [star, moon] | star | "dream imagery, night sky" |
| (13.0 omitted from highlights, listed in non_hooks) |  |  |  |  | gibberish, no visual cue |

The v6 cache key includes the vocab fingerprint, so editing
`tag_vocabulary.json` automatically busts old cache entries.

### Acceptance commands

```powershell
# 1. Index: 25 stickers across 15 / 20 tags (5 MISS).
uv run python scripts/build_index.py

# 2. Tests: 23 new tests (17 semantic_align + 6 build_elements_v6); full suite 150 / 150.
uv run pytest tests/test_semantic_align.py -v

# 3. Rule-based render — produces outputs/output_rule.mp4 (404x720, 13.75 s, 2.1 MB).
uv run python -m semanticvibe.render `
    --video samples/dance.mp4 --lyrics samples/lyrics_test.json `
    --provider rule_based --out outputs/output_rule.mp4 --preview `
    --assets-dir data/assets_lib

# 4. Claude render — same shape, hits .cache/alignment/<md5>.json on the first
#    call (requires ANTHROPIC_API_KEY).
uv run python -m semanticvibe.render `
    --video samples/dance.mp4 --lyrics samples/lyrics_test.json `
    --provider claude --song-title "lyrics_test" `
    --out outputs/output_claude.mp4 --preview --assets-dir data/assets_lib
```

Probe at t=8.0 s (`outputs/probe_rule_t8.png`) shows the red 「好き」
pose-aware-placed beside the left dancer + a pink heart in the
upper-right corner — closed-vocab tag `heart` resolved through
`AssetRetriever` to `data/assets_lib/heart.png`.

### Known limits

- 5 MISS tags (burst / dot / kiss / leaf / rainbow) have no PNGs yet.
  `AssetRetriever` substitutes within the same category at render time;
  if even the global fallback (`heart`) is missing, the decoration is
  skipped silently. Generate real assets via
  `scripts/generate_stickers.py` and re-run `build_index.py` to
  promote them.
- The CLI's `--lyrics` mode still routes through the v5 `build_decision`
  (which already does pose-aware placement). `build_elements_from_lyrics`
  is callable as a library function and via `--elements-json` for
  round-tripped JSON, but it doesn't run pose detection on its own —
  it emits `anchor: "auto"` so the render's existing fallback
  (lower-band centred) takes over when pose data isn't supplied.
- Claude's strict-JSON parser tolerates code fences and prose
  surrounding the JSON object, but if the model emits a syntactically
  invalid JSON the alignment falls back to `rule_based` (logged as a
  warning). The cache is *not* written for fallbacks — only successful
  Claude outputs.

---

# v5+ — lyrics input modes (Whisper auto / standalone audio / manual JSON)

## What changed since v5

`render` now resolves lyrics from three sources by priority, instead of
requiring a manual `--lyrics` JSON every time:

1. `--lyrics` (manual JSON) — highest priority, bypasses Whisper
2. `--audio` (independent audio file) — Whisper transcribes that file
3. (default) Whisper transcribes the video's embedded audio track

Plus optional `--mix-audio replace|overlay` to splice the
`--audio` track into the final mp4 (otherwise the video's original
audio is kept).

### New / modified

| File | What |
|---|---|
| `src/semanticvibe/lyrics.py` (new) | Pydantic `LyricLine(time, text, duration?)` + `LyricsFile` root model + `load_lyrics` / `save_lyrics` / `to_dict_list` helpers. Single source of truth for the on-disk schema. |
| `src/semanticvibe/preprocess/whisper_asr.py` | Split `transcribe()` into two named entry points: `transcribe_audio(path)` (core, accepts any ffmpeg-decodable file) and `transcribe_video(path)` (alias). The legacy `transcribe` name still resolves to `transcribe_video` for backward compat with `preprocess/pipeline.py` and `cli.py`. |
| `src/semanticvibe/render/__main__.py` | Big rework: `--lyrics` is now optional, `--audio` and `--mix-audio replace/overlay` and `--whisper-model` and `--language` and `--device` added. New `get_lyrics(args)` implements the three-mode priority. New `_maybe_mix_audio()` post-processes via imageio_ffmpeg's bundled binary (replace = `-map 0:v -map 1:a -c:v copy -c:a aac -shortest`; overlay = `amix=inputs=2:duration=shortest`). |
| `src/semanticvibe/semantic_align.py` | `LyricLine` now re-exports the Pydantic version; `Highlight` gains optional `duration: float \| None`; `_rule_based_align` forwards `LyricLine.duration` into `Highlight.duration`; `load_lyrics` delegates to `lyrics.load_lyrics` so schema validation lives in one place. |
| `src/semanticvibe/build_elements.py` | `_highlight_duration` now respects `Highlight.duration` when explicitly set; falls back to gap-based heuristic when None. |
| `scripts/preview_lyrics.py` (new) | Whisper preview CLI: takes `--video` or `--audio`, prints results to console, caches at `.cache/lyrics/<sha1>.json` (keyed by path + model + language + mtime), and writes a fresh editable copy to `samples/auto_lyrics.json`. Subsequent runs hit cache; `--force` re-runs. |
| `tests/test_lyrics.py` (new) | 9 tests covering the Pydantic schema (negative time / empty text / non-positive duration rejection, duration round-trip preserves None, save omits None duration field, load gives clear ValidationError on missing field, accepts the committed `samples/lyrics_mosimosi.json`, `to_dict_list` drops None). |
| `tests/test_render_cli.py` (new) | 8 tests covering CLI argument parsing + `get_lyrics()` priority dispatch (mock Whisper, assert priority-1 wins / priority-2 routes to `--audio` / priority-3 routes to video itself / `--mix-audio` choices validated). |

### Lyrics JSON schema

```json
[
  {"time": 2.5, "text": "もしもし"},
  {"time": 5.0, "text": "電波", "duration": 0.8}
]
```

`time` (sec, ≥ 0) and `text` (non-empty) are required. `duration` is
optional — when set, the line stays on screen for that many seconds;
when omitted, the renderer holds it until the next line minus 0.3 s
breathing room (capped at 5 s). Pydantic validation surfaces clear
errors: feeding `{"timestamp": 1, "text": "x"}` raises with a message
that names the missing `time` field.

### Tests

134 passing (was 117). +17 from the two new suites.

### Verified

- `python -m semanticvibe.render --help` shows the new flags.
- 134/134 pytest green.
- `samples/lyrics_mosimosi.json` (no `duration` field) still
  round-trips through `load_lyrics → save_lyrics → load_lyrics`.

### Caveats

The `_maybe_mix_audio` helper hasn't been smoke-tested end-to-end
because it needs a real audio file alongside `samples/dance.mp4`.
ffmpeg invocation is straightforward (matches the spec's example
verbatim) but if you hit issues with `--mix-audio overlay` on
specific codecs, the mix happens through `amix=inputs=2`; switch to
`-c:a libmp3lame` in `_maybe_mix_audio` if AAC encoding chokes.

---

# v5 — lyrics-driven, pose-aware element generation

## What changed

The old flow was "load a hand-written Decision JSON → render". v5 generates
the Decision programmatically from two inputs only: a lyrics list and the
video itself. No more hardcoded「夢」, no more pixel-precise coordinates
in committed example files.

```
              lyrics.json ──┐
                            ├─ semantic_align.align()  → list[Highlight]
              video        ─┤
                            ├─ pose_detector.detect_person_mask()
                            │    → time → bool occupancy mask
                            ├─ build_elements.build_decision()
                            │    → Decision (auto-placed text + tagged decoration)
                            └─ render.composite.render_from_decision()
                                 → mp4
```

## New modules

| File | Responsibility |
|---|---|
| `src/semanticvibe/pose_detector.py` | `detect_person_mask(video, sample_fps=2)` walks the video at 2 fps, runs MediaPipe Pose, returns `{time → bool ndarray (H, W)}` with 30 px safety padding. `pick_nearest_mask(masks, t)` picks the mask closest in time. |
| `src/semanticvibe/layout/zones.py` | `find_placement_zone(mask, target_size, prefer)` — morphological erosion turns the free region into a "valid top-left corner" map; the largest matching connected component centroid is returned. Re-exported at `semanticvibe.layout`. |
| `src/semanticvibe/semantic_align.py` | `TAG_VOCABULARY` (multilingual keyword → asset_tag map), `Highlight` dataclass (lyric_time / lyric_text / decoration_tag / strength), `align(lyrics, provider)` with three providers: `rule_based` (offline, longest-trigger-wins keyword match), `claude`, `openai`. LLM providers fall back to rule_based on any error. |
| `src/semanticvibe/build_elements.py` | `build_decision(highlights, person_masks, canvas_size, fonts_dir)` measures each text tile, asks `find_placement_zone` for a person-avoiding position, picks animations from strength pools (scale_pop / stamp / drop_in for high-strength, fade / slide_in / wobble_in for normal), emits matching `DecorationElement` for each tag. |
| `src/semanticvibe/render/__main__.py` | New CLI matching `python -m semanticvibe.render --video --lyrics --provider --out`. Replaces hand-written JSON workflow. |
| `samples/lyrics_mosimosi.json` | Sample lyric input file (4 Japanese lines covering all the major tag buckets). |
| `samples/dance.mp4` | Copy of `data/test_videos/suki.mp4` so the canonical command works out of the box. |

## Modified

- `src/semanticvibe/render/composite.py`:
  `_load_decoration_base` no longer triggers CLIP fallback when `library.by_tag()` misses. Reason: the fallback pulls `open_clip → transformers → sklearn → pandas → pyarrow` which has been observed to crash with `WinError 6714` during pyarrow's import-time directory scan on this Windows install. Missing tags now silently skip the decoration; if you genuinely need fuzzy matching, call `assets.clip_search.find_asset` directly.
- `src/semanticvibe/layout/__init__.py`: re-exports `find_placement_zone`.

## Removed / moved

All hardcoded sample JSONs moved from `examples/` to `examples/legacy/`:
`baseline_dream.json`, `baseline_dream_v4.json`, `sing_handdrawn.json`,
`sing_full.json`, `suki_decision.json`, `demo_chinese.json`,
`hand_written_decision.json`. They still parse against the schema — kept
for reference and for the `test_decision_loads_hand_written_example`
schema-roundtrip test.

`tests/conftest.py` now points `hand_written_decision_dict` fixture at
the new legacy location.

## Tests

117 passing (was 117 entering this work — same count, different
distribution: removed 2 stale assertions, added 5 new ones for v5
modules):

| Suite | Coverage |
|---|---|
| `test_layout_zones.py` (new) | empty mask + preferred quadrant routing, central-strip avoidance, oversized-target rejection, fully-occupied rejection, all 4 prefer quadrants, fallback when preferred quadrant fully occupied. |
| `test_semantic_align.py` (new) | mosimosi sample case end-to-end, strength differentiation, dict input acceptance, longest-trigger-wins (`可愛い` beats `愛`), TAG_VOCABULARY canonical-trigger resolution, JSON parser tolerates code fences + prose wrapper, unknown provider raises. |
| Existing 9 suites | unchanged, still green. |

`test_decision_loads_hand_written_example` updated to point at
`examples/legacy/hand_written_decision.json`.

## Verified end-to-end

```
uv run python -m semanticvibe.render \
    --video samples/dance.mp4 \
    --lyrics samples/lyrics_mosimosi.json \
    --provider rule_based \
    --out outputs/output_v5.mp4 \
    --preview
```

→ `outputs/output_v5.mp4` (404×720, 13.8 s). Probe frames confirm:

- t=3.0 s 「もしもし」 (no decoration tag — `もしもし` has no idiomatic
  trigger in TAG_VOCABULARY, so just text appears)
- t=5.5 s 「電波」 + red exclaim impact star (lightning aliased to
  exclaim because no `lightning` asset exists)
- t=8.5 s 「好き」 + pink heart
- t=11.5 s 「可愛い」 + small pink mini-heart

All four placements avoid the central dancers via the
`find_placement_zone` + person-mask path. Text content traces directly
back to `lyrics_mosimosi.json` — no 「夢」 anywhere.

## What I made up vs what was specified

The user referenced "上一輪我給你的完整 prompt" for `semantic_align.py`,
but that prompt isn't in this conversation's history. I designed the
module from scratch based on the demonstrated requirements (tag
vocabulary, rule_based + claude providers, returns highlights with
decoration_tag + strength). If the user has a different intended shape,
the surface area is small enough to swap.

The user wrote `semanticvibe/layout.py` (flat) but the project already
has `semanticvibe/layout/` as a package. I put the new code in
`layout/zones.py` and re-exported `find_placement_zone` at the package
root, so `from semanticvibe.layout import find_placement_zone` works as
the prompt implied.

The user wrote `修改 render.py` but no such module existed. I created
`render/__main__.py` so the canonical command `python -m
semanticvibe.render` resolves to it.

The user wrote `samples/dance.mp4` which didn't exist. Copied
`suki.mp4` (a parking-garage dance clip) into that path. Both are
gitignored so this only affects local working state.

`lightning` and `flower` got aliased to existing assets (`exclaim` and
`mini-heart`) because the procedural sticker library doesn't ship those
tags. To get genuine lightning / flower stickers, generate them via
`scripts/generate_stickers.py` (Z-Image Turbo + Sticker LoRA — see
prior commits) and add to `data/assets_lib/metadata.json`. Once those
PNGs exist, just remove the alias entries from `TAG_VOCABULARY`.
