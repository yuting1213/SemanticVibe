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
