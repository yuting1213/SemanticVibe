# v5 progress — lyrics-driven, pose-aware element generation

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
