"""
demo_app.py — Minimal example using theme.py
============================================================
Run this to see every UI piece from the original app, with
placeholder content instead of the RAG/CHI-paper logic.

    streamlit run demo_app.py

Use this as a reference for how to call each theme.py helper —
copy the patterns into your real app and swap in your own data.
"""

import streamlit as st
from theme import (
    set_page_config, inject_css, COLORS,
    hero, divider, stat_box, status_pill, success_bar, error_box,
    empty_state, answer_box_label, answer_body, card_entry, chip,
    sidebar_brand, sidebar_section_label, sidebar_kv_row, sidebar_hr,
)

# ── page setup — do this first, exactly once ───────────────────────────────
set_page_config(page_title="My New Project", page_icon="✨")
inject_css()

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    sidebar_brand(eyebrow="MY PROJECT", title_html="Project<br>Name")

    sidebar_section_label("Overview")
    sidebar_kv_row("Items", "1,204")
    sidebar_kv_row("Source", "internal-db")
    sidebar_kv_row("Engine", "your-model-here")
    sidebar_hr()

    sidebar_section_label("System Status")
    status_pill("Service Online", status="ok")
    status_pill("Index Ready", status="ok")
    status_pill("Cache Warming", status="pending")
    sidebar_hr()

    with st.expander("⚙️ Settings", expanded=False):
        st.text_input("Some setting", placeholder="value")
        st.button("Save")

# ══════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════
hero(
    badge_text="MY DATA",
    title="Your Project Title Here",
    subtitle="A short description of what this tool does goes here, "
             "matching the original's centered muted subtitle style.",
)

# ══════════════════════════════════════════════════════════════════════════
# SEARCH / INPUT FORM (same pattern as the original: form + columns)
# ══════════════════════════════════════════════════════════════════════════
with st.form(key="demo_form", clear_on_submit=False):
    c1, c2 = st.columns([5, 1])
    with c1:
        st.text_input("q", key="_demo_q", placeholder="Type something…",
                      label_visibility="collapsed")
    with c2:
        submitted = st.form_submit_button("Search", use_container_width=True)

st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

# ── suggestion-style buttons (uses the default .stButton CSS) ─────────────
if not submitted:
    divider("Curated Starting Points")
    sc1, sc2 = st.columns(2)
    suggestions = [
        "First example suggestion text",
        "Second example suggestion text",
        "Third example suggestion text",
        "Fourth example suggestion text",
    ]
    for i, s in enumerate(suggestions):
        col = sc1 if i % 2 == 0 else sc2
        with col:
            st.button(s, key=f"sug_{i}", use_container_width=True)

st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# RESULTS STATE — demonstrates success_bar / answer box / cards / stats
# ══════════════════════════════════════════════════════════════════════════
if submitted:
    success_bar(
        message="Done. Found <strong style='color:#047857;'>3</strong> matching results.",
        right_label="your-model-here",
        right_sublabel="1,204 tokens processed",
    )

    answer_box_label(icon="✧", label="Executive Summary")
    answer_body("This is where your **generated answer** or summary text goes. "
                "Regular markdown like *italics*, lists, and `code` all render normally here.")

    divider("Results")
    for i in range(1, 4):
        card_entry(
            rank=i,
            title=f"Example Result Title {i}",
            badges=[("2024", "year"), (f"Relevance: 0.{95 - i * 5}", "score")],
            quote="a short illustrative excerpt would go here",
            progress_pct=95 - i * 15,
        )

    with st.expander("🔬 Technical Diagnostics", expanded=False):
        cols = st.columns(4)
        for col, (label, value) in zip(cols, [
            ("Base Retrieval", 120), ("Unique Entities", 87),
            ("Post Filter", 42), ("Context Pool", 3),
        ]):
            with col:
                stat_box(value, label)

        st.markdown("<div style='margin-top:1rem;'></div>", unsafe_allow_html=True)
        chip("Q1: example sub-query one")
        chip("Q2: example sub-query two")
        chip("⚠ example warning chip", warning=True)

else:
    empty_state(
        eyebrow="Engine Standing By",
        message="Enter a query above and press Enter to get started.",
    )

# ── error state example (commented out — uncomment to preview) ───────────
# error_box(
#     title="System Exception",
#     message="Traceback (most recent call last):\n  ...\nValueError: example error",
#     footnote="If you hit a rate limit, check ⚙️ Settings in the sidebar.",
# )
