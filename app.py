"""Streamlit demo UI — implemented in Week 5 (spec §8.5).

Kept as a placeholder so the entry-point path in pyproject.toml resolves.
"""

from __future__ import annotations


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="SemanticVibe", layout="wide")
    st.title("SemanticVibe")
    st.info(
        "Streamlit UI lands in Week 5. Until then, use `uv run python -m semanticvibe.render_demo`."
    )


if __name__ == "__main__":
    main()
