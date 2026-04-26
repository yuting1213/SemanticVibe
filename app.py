"""Streamlit UI for SemanticVibe.

    uv run streamlit run app.py

Lets a user upload a video, pick a style preset, optionally swap LLM provider,
and download the rendered overlay.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import streamlit as st

from semanticvibe.config import COST_MODES, STYLE_PRESETS, get_settings
from semanticvibe.pipeline import render_from_intermediate, run

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")


def _settings_sidebar() -> dict:
    settings = get_settings()
    st.sidebar.header("Settings")
    provider = st.sidebar.selectbox(
        "LLM provider",
        options=["claude", "openai"],
        index=["claude", "openai"].index(settings.llm_provider),
    )
    cost_mode = st.sidebar.selectbox(
        "Cost mode",
        options=["dev", "prod"],
        index=["dev", "prod"].index(settings.cost_mode),
        help="dev = Haiku/4o-mini (~1/20 prod cost). prod = Sonnet/4o.",
    )
    style = st.sidebar.selectbox(
        "Style preset", options=sorted(STYLE_PRESETS.keys()), index=0
    )
    preview = st.sidebar.checkbox("Preview mode (720p)", value=True)
    has_key = (
        (provider == "claude" and settings.anthropic_api_key)
        or (provider == "openai" and settings.openai_api_key)
    )
    if not has_key:
        st.sidebar.warning(
            f"No API key for {provider}. The pipeline will fall back to the "
            "deterministic heuristic Decision generator."
        )
    st.sidebar.caption(
        f"Resolved model: `{COST_MODES[cost_mode][provider]}`"
    )
    return {
        "provider": provider,
        "cost_mode": cost_mode,
        "style": style,
        "preview": preview,
    }


def main() -> None:
    st.set_page_config(page_title="SemanticVibe", layout="wide")
    st.title("SemanticVibe")
    st.caption(
        "Whisper + LLM + CLIP + MoviePy. Upload a video, pick a vibe, "
        "get an overlay."
    )

    cfg = _settings_sidebar()

    uploaded = st.file_uploader(
        "Video", type=["mp4", "mov", "mkv"], accept_multiple_files=False
    )

    workdir_root = Path(tempfile.gettempdir()) / "semanticvibe_streamlit"
    workdir_root.mkdir(parents=True, exist_ok=True)

    col1, col2 = st.columns([1, 1])
    run_btn = col1.button("Render", type="primary", disabled=uploaded is None)
    rerender_btn = col2.button(
        "Re-render from last Decision",
        help="Skips Stages 1–4 and re-runs Stage 5 only. Faster iteration on "
        "the same Decision JSON.",
        disabled=uploaded is None,
    )

    if uploaded and (run_btn or rerender_btn):
        run_dir = workdir_root / f"run_{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=True)
        video_path = run_dir / uploaded.name
        video_path.write_bytes(uploaded.getbuffer())
        output_path = run_dir / "output.mp4"

        with st.spinner("Rendering…"):
            try:
                if rerender_btn:
                    # Look for the most recent intermediates dir.
                    candidates = sorted(workdir_root.glob("run_*"), reverse=True)
                    decision_json = None
                    for c in candidates:
                        cand = c / "decision_resolved.json"
                        if cand.exists():
                            decision_json = cand
                            break
                    if decision_json is None:
                        st.error(
                            "No prior Decision found. Run a full render first."
                        )
                        return
                    out = render_from_intermediate(
                        video_path,
                        decision_json,
                        output_path,
                        preview=cfg["preview"],
                    )
                else:
                    out = run(
                        video_path,
                        output_path,
                        style_preset=cfg["style"],
                        provider=cfg["provider"],
                        preview=cfg["preview"],
                        intermediate_dir=run_dir,
                    )
            except Exception as exc:  # noqa: BLE001 — surface anything, debug-friendly
                st.exception(exc)
                return

        st.success(f"Done. {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
        st.video(str(out))
        st.download_button(
            "Download mp4",
            data=out.read_bytes(),
            file_name=out.name,
            mime="video/mp4",
        )

        intermediates = run_dir
        with st.expander("Intermediate artefacts (FeatureSummary, Decision JSON)"):
            for name in (
                "feature_summary.json",
                "decision.json",
                "decision_resolved.json",
            ):
                p = intermediates / name
                if p.exists():
                    st.subheader(name)
                    st.json(p.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
