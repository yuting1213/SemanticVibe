# HyperFrames Validation Report

Render: `playground/renders/test_validate.mp4` (1920×1080, 30 fps, 5.0 s, 722 KB)
Spec: `playground/test-hyperframes/index.html`
Render time: **13.1 s** for 150 frames (5 parallel Chrome workers + GPU, HW-accel WebGL)

## Validation questions

### (a) Can I precisely control when an element appears (the second)?

**YES — sub-frame precision.** Each `.clip` element carries `data-start` and
`data-duration` (seconds, accepts decimals). The GSAP timeline is built with
absolute time offsets via `tl.from("#id", { ... }, 1.0)` — the `1.0` is the
wall-clock second the animation begins, not a relative delay.

In the test:
- `#denpa` set `data-start="1.0"` + `tl.from("#denpa", ..., 1.0)` → frame 30 (= 1.0 s × 30 fps) shows the start of the scale_pop. Probe at t=1.2 s confirms 「電波」 is in.
- `#heart1` set `data-start="2.5"` → frame 75 shows it stamping in.
- `#suki_group` set `data-start="4.0"` → frame 120 shows it appearing.

Probes at t=0.5 (before 電波), t=1.2 (電波 visible), t=2.7 (電波 + 心 both visible), t=4.2 (好き ♥ in), t=4.7 (好き ♥ pulsing) all matched the specified schedule.

### (b) Can I control position with pixel (x, y) coordinates?

**YES — pure CSS pixel coordinates.** The composition declares its canvas
size via `<meta name="viewport" content="width=1920, height=1080">` and
`#root` is sized at exactly that. From there everything is plain CSS —
`left: 100px; top: 200px;` works exactly like a browser.

In the test `#denpa` uses `left: 100px; top: 200px;` (exact spec match);
`#heart1` uses `left: 600px; top: 150px;`. `#suki_group` mixes percent +
`translate(-50%, -50%)` for the "centre-lower" anchor — but you can use
straight pixel coords if you prefer.

Anchoring choices supported per-element:
- pixel: `left/top` in px
- percent: `left: 50%; top: 70%` for fractional placement
- viewport-relative: not generally useful (`vw/vh` collapses to fixed sizes)
- flex/grid layouts: works (used here for `#suki_group`)
- transform-based: `transform: translate(...)` works for sub-pixel offsets

### (c) Can I feed external PNGs? Format?

**YES — any web-standard format Chrome can decode.** PNG / JPEG / WebP / AVIF
/ SVG all work. Just reference them with relative URLs:

```html
<img src="./heart.png" alt="heart" />
```

The test copies one tofu heart PNG (`assets/stickers/heart/tofu_00008_.png`,
512×512, transparent) into the project folder and references it twice
(once at t=2.5 as `#heart1`, once at t=4.0 as `#suki_heart`). Both render
crisp at any scale.

Sizes are controlled via CSS (`width: 80px` for the small one,
`width: 120px` for the group heart); the source PNG's native resolution is
respected for sharpness but the displayed pixel size is whatever CSS says.

External media (videos, audio, web fonts) also supported — see hyperframes
docs for `data-track-index` ordering of video clips.

### (d) Can idle animations (continuous breathing) be set?

**YES — GSAP `repeat: -1, yoyo: true` is exactly this.** The renderer is
agnostic to what GSAP does; whatever animation runs in the browser gets
captured frame-by-frame.

In the test, `#suki_text` and `#suki_heart` both have:

```js
tl.to("#suki_text", {
    scale: 1.08,
    duration: 0.5, repeat: -1, yoyo: true, ease: "sine.inOut",
}, 4.5);
```

`repeat: -1` = forever, `yoyo: true` = bounce back, `ease: "sine.inOut"` =
smooth breathing curve. Comparing probes:
- t=4.2 (just after entry, scale 1.0)
- t=4.7 (mid-pulse) the heart is visibly different size from t=4.2,
  confirming the loop runs.

This is more flexible than a fixed pulse period — GSAP supports
arbitrary easing (back / elastic / steps / custom CubicBezier) and you
can layer multiple repeats with different phases per element.

### (e) Can multiple elements have independent animations simultaneously?

**YES — every `tl.from(...)` / `tl.to(...)` call is independent.** GSAP's
timeline composes tweens that act on separate selectors; they don't
interfere unless they target the same property of the same element.

The test proves this 3 ways:

1. `#denpa` (scale_pop entry, no idle) is animating between t=1.0 and 1.5
   while `#heart1` is invisible. They never overlap timeline-wise.
2. At t≈4.5 s, `#denpa` is gone, `#heart1` is fading out (its 2 s window
   ended at 4.5), AND `#suki_group` is entering (started at 4.0) — three
   distinct lifecycles running, none colliding.
3. `#suki_text` pulses at `duration: 0.5 s` and `#suki_heart` pulses at
   `duration: 0.4 s` with `rotation: -8°` — both children of the same
   group, two different idle loops, completely independent phases. The
   t=4.7 probe shows the text and heart at different scales mid-pulse.

The composition pattern is simply: declare each element with its lifetime
in HTML, then add one or more `tl.from/to(...)` calls per element at the
right offset. No coordinator needed.

## Animation quality (subjective 1-5)

**4.5 / 5.**

What works:
- GSAP eases (`back.out`, `elastic`, `bounce`, custom cubic-bezier) feel
  high-quality and snappy. `back.out(2.2)` for the scale_pop on 電波
  reads as bouncy without overshooting wildly.
- Sub-pixel sub-frame precision — no aliasing or stair-stepping visible
  at 30 fps. Chrome WebGL renders the text crisply.
- Custom outlines via `-webkit-text-stroke` give the pink-outline-white-
  fill look exactly matching v10 `SubtitleOutlinedElement` rendering.
- Independent timelines compose cleanly. Easy to read in source.

What's mid:
- `-webkit-text-stroke` outline does not anti-alias as smoothly as our
  Pillow "circular thick-outline" approach (multi-direction draws within
  radius). Aliased pink ring is slightly visible at low resolutions.
  Workaround: use SVG text + `stroke-width` for production-grade.
- No built-in concept of "track avoiding the dancer's face" — placement
  is purely declared by CSS. To replicate semanticvibe's pose-aware
  layout you'd need to either generate the CSS at build time or use
  hyperframes' programmatic API to compute positions.

## Learning curve

**~3 min** to scaffold + read the docs (`hyperframes init` + `--help`).
**~3 min** writing the spec from scratch (I'm familiar with GSAP — for
someone new, expect ~30 min on the first try).
**~7 min** total spec-writing including reading our heart PNG / bg setup +
testing the render.

If you already know HTML/CSS + GSAP, hyperframes adds essentially zero
mental tax — just a few `data-start/duration` attributes and you're done.
If you don't know GSAP, the docs example is enough to start; the API is
small (`timeline`, `.from`, `.to`, `.add`, `.from(target, vars, position)`
covers ~95% of use cases).

## Verdict for the SemanticVibe use case

Hyperframes is a **better fit than our current Pillow + moviepy pipeline
for intro/outro cards, branded title sequences, and CSS-friendly UI
animation**. It's a **weaker fit for lyric-overlay onto existing dance
video** because:

1. No pose-aware layout (would need custom layout-generator that emits
   CSS coords)
2. No semantic LLM alignment built in (you'd run our `semanticvibe.semantic_align` separately and inject coords into HTML)
3. Doesn't ingest existing MP4s as the base — you'd have to put the
   dance video as a `<video>` element inside the composition. Doable but
   loses our cv2/MediaPipe pose-mask flow.

**Hybrid approach**: use hyperframes to render the **overlay layer only**
(transparent WebM via `--format webm`), then composite over the dance
video with ffmpeg. The CSS animation quality + GSAP eases would
significantly upgrade the title/decoration look while we keep our
pose-detection + LLM alignment from semanticvibe.
