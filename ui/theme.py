"""
theme.py — Reusable Streamlit UI Theme
============================================================
Extracted from the original app's design system. Same fonts,
same colors, same component shapes — none of the original
business logic (no RAG pipeline, no Gemini, no API key stuff).

USAGE
-----
    import streamlit as st
    from theme import inject_css, badge, divider, stat_box, success_bar, error_box

    st.set_page_config(page_title="My App", page_icon="🔬", layout="wide")
    inject_css()

    badge("MY DATA")
    st.markdown("<div class='chi-title'>My Page Title</div>", unsafe_allow_html=True)

Every helper function below just returns/writes HTML using the CSS
classes defined in inject_css(). If you need a layout this file
doesn't have a helper for yet, copy the class names from inject_css()
and write your own st.markdown(..., unsafe_allow_html=True) block —
that's exactly how the original file did it too.
"""

import streamlit as st


# ══════════════════════════════════════════════════════════════════════════════
# 1. PAGE CONFIG HELPER (optional convenience — you can call st.set_page_config
#    yourself instead, this just keeps the icon/layout defaults in one place)
# ══════════════════════════════════════════════════════════════════════════════
def set_page_config(page_title: str, page_icon: str = "🔬"):
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ══════════════════════════════════════════════════════════════════════════════
# 2. CORE CSS INJECTION — call this once, right after set_page_config()
# ══════════════════════════════════════════════════════════════════════════════
def inject_css():
    """Injects every CSS rule from the original design system, unchanged:
    fonts, colors, background grid, floating card layout, sidebar, inputs,
    buttons, expanders, scrollbar, and all component classes
    (badge / title / divider / answer-box / paper-entry / stat-box /
    status-pill / success-bar)."""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,500;0,600;0,700;1,400&family=Lora:ital,wght@0,500;0,600;0,700;1,500&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── base / background grid ──────────────────────────────────────────── */
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background-color: #f8fafc !important;
        background-image: radial-gradient(#cbd5e1 1px, transparent 1px) !important;
        background-size: 32px 32px !important;
    }

    html, body, [data-testid="stAppViewContainer"], p, span, li, label, .stMarkdown * {
        font-family: 'Inter', -apple-system, sans-serif;
        color: #1e293b;
    }
    
    /* Preserve Streamlit Material Icons */
    .stIcon, .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
    }

    /* ── floating document layout ────────────────────────────────────────── */
    .main .block-container, [data-testid="stMainBlockContainer"] {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(12px) !important;
        border-radius: 16px !important;
        box-shadow: 0 20px 40px -15px rgba(15, 23, 42, 0.08), 0 0 0 1px rgba(226, 232, 240, 0.8) !important;
        padding: 3.5rem 4.5rem !important;
        margin-top: 3rem !important;
        margin-bottom: 4rem !important;
        max-width: 960px !important;
    }

    /* ── sidebar ─────────────────────────────────────────────────────────── */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background: #ffffff !important;
        border-right: 1px solid rgba(226, 232, 240, 0.8) !important;
        box-shadow: 4px 0 24px rgba(15, 23, 42, 0.02) !important;
    }
    [data-testid="stSidebar"] * { color: #475569 !important; }
    [data-testid="stSidebar"] input { background: #f8fafc !important; border-radius: 8px !important; border: 1px solid #e2e8f0 !important; }

    /* ── hide chrome ─────────────────────────────────────────────────────── */
    #MainMenu, footer, [data-testid="stToolbar"], [data-testid="stHeader"], header { visibility: hidden !important; height: 0 !important; }
    .stDeployButton { display: none !important; }


    /* ── text input ──────────────────────────────────────────────────────── */
    .stTextInput input, [data-baseweb="input"] input, input[type="text"] {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 12px !important;
        color: #0f172a !important;
        font-size: 1.05rem !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.9rem 1.4rem !important;
        caret-color: #2563eb !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.04), inset 0 2px 4px rgba(255,255,255,0.5) !important;
    }
    .stTextInput input:focus, [data-baseweb="input"] input:focus {
        border-color: #2563eb !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15), 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
        outline: none !important;
    }
    .stTextInput input::placeholder { color: #94a3b8 !important; font-weight: 400 !important; }
    [data-baseweb="input"], .stTextInput > div > div { background: transparent !important; border: none !important; }

    /* ── form ────────────────────────────────────────────────────────────── */
    [data-testid="stForm"], [data-testid="stForm"] > div { background: transparent !important; border: none !important; padding: 0 !important; }

    /* ── primary button (form submit) ───────────────────────────────────── */
    [data-testid="stFormSubmitButton"] > button, .stFormSubmitButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: #ffffff !important;
        border: 1px solid #1e40af !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        width: 100% !important;
        padding: 0.9rem 1rem !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.25), inset 0 1px 0 rgba(255,255,255,0.2) !important;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
    }
    [data-testid="stFormSubmitButton"] > button *, .stFormSubmitButton > button * {
        color: #ffffff !important;
    }
    [data-testid="stFormSubmitButton"] > button:hover, .stFormSubmitButton > button:hover {
        box-shadow: 0 6px 14px rgba(37, 99, 235, 0.35), inset 0 1px 0 rgba(255,255,255,0.2) !important;
        transform: translateY(-2px) !important;
    }

    /* ── secondary buttons (suggestion / card style) ────────────────────── */
    .stButton > button {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        color: #334155 !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        padding: 1rem 1.2rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 4px rgba(15, 23, 42, 0.02) !important;
        display: flex !important;
        justify-content: flex-start !important;
        text-align: left !important;
        height: auto !important;
        line-height: 1.5 !important;
    }
    .stButton > button:hover {
        border-color: #93c5fd !important;
        background: #f0fdfa !important;
        color: #0369a1 !important;
        box-shadow: 0 8px 16px rgba(15, 23, 42, 0.06) !important;
        transform: translateY(-2px) !important;
    }

    /* ── expander ────────────────────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02) !important;
    }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p {
        color: #334155 !important; font-size: 0.85rem !important; font-weight: 600 !important;
    }

    /* ── scrollbar ───────────────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 99px; }

    /* ─── component classes ─────────────────────────────────────────────── */

    /* Badge (small pill label, e.g. above a page title) */
    .chi-badge {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 99px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #1d4ed8;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 4px rgba(37,99,235,0.05);
    }
    /* Big serif page/hero title */
    .chi-title {
        font-family: 'Lora', serif !important;
        font-size: clamp(2.4rem, 5vw, 3.8rem);
        font-weight: 600;
        letter-spacing: -0.03em;
        line-height: 1.1;
        color: #0f172a;
        margin-bottom: 0.8rem;
    }
    /* Subtitle under the hero title */
    .chi-subtitle {
        font-size: 1.05rem;
        color: #64748b;
        line-height: 1.6;
        max-width: 580px;
        margin: 0 auto;
        font-weight: 400;
    }

    /* Section dividers ("──── LABEL ────") */
    .fancy-divider {
        display: flex;
        align-items: center;
        text-align: center;
        margin: 2.5rem 0 1.5rem;
    }
    .fancy-divider::before, .fancy-divider::after {
        content: '';
        flex: 1;
        border-bottom: 1px solid #e2e8f0;
    }
    .fancy-divider:not(:empty)::before { margin-right: 1em; }
    .fancy-divider:not(:empty)::after { margin-left: 1em; }
    .fancy-divider span {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #94a3b8;
    }

    /* Answer / content box (left blue accent border) */
    .answer-box {
        background: linear-gradient(180deg, #ffffff 0%, #fafaf9 100%);
        border: 1px solid #e2e8f0;
        border-left: 4px solid #2563eb;
        border-radius: 12px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
        font-size: 0.98rem;
        line-height: 1.7;
        color: #334155;
    }
    .answer-box-label {
        font-family: 'Lora', serif;
        font-size: 1.15rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    /* Body text rendered just below an answer-box (left rule, no box) */
    .generated-answer-container {
        font-family: 'Inter', sans-serif;
        font-size: 1.05rem;
        line-height: 1.75;
        color: #334155;
        padding: 0 1rem 0 1.5rem;
        border-left: 3px solid #cbd5e1;
        margin-bottom: 3.5rem;
    }

    /* Card / list-entry style (originally "paper" bibliography cards) */
    .paper-entry {
        display: flex;
        gap: 1.2rem;
        padding: 1.6rem 1.8rem;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.02);
        transition: box-shadow 0.2s, transform 0.2s;
    }
    .paper-entry:hover {
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        transform: translateY(-2px);
        border-color: #cbd5e1;
    }
    .paper-rank-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.2rem;
        color: #cbd5e1;
        font-weight: 600;
        min-width: 30px;
        text-align: right;
    }
    .paper-body { flex: 1; }
    .paper-title-txt {
        font-family: 'Lora', serif;
        font-size: 1.15rem;
        font-weight: 600;
        color: #0f172a;
        line-height: 1.4;
        margin-bottom: 0.6rem;
    }
    .paper-meta-row {
        display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
        margin-bottom: 0.8rem;
    }
    .pmeta-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        background: #f1f5f9;
        color: #475569;
    }
    .pmeta-badge.year { background: #eff6ff; color: #2563eb; }
    .pmeta-badge.score { background: #ecfdf5; color: #059669; }
    .paper-quote {
        font-size: 0.85rem;
        color: #64748b;
        border-left: 2px solid #cbd5e1;
        padding-left: 0.8rem;
        font-style: italic;
        line-height: 1.5;
    }

    /* Small mono "chip" tag */
    .qchip {
        display: inline-block; background: #f8fafc;
        border: 1px solid #e2e8f0; border-radius: 6px;
        padding: 4px 10px; font-size: 0.75rem; color: #475569;
        margin: 3px 4px 3px 0; font-family: 'JetBrains Mono', monospace;
    }

    /* Stat grid box (number + label) */
    .stat-box {
        background: linear-gradient(180deg, #ffffff 0%, #fafaf9 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1.2rem; text-align: center;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.02);
    }
    .stat-num { font-size: 1.6rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; color: #0f172a; }
    .stat-txt { font-size: 0.65rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 6px; font-weight: 700; }

    /* Status pill row (colored dot + label, used in sidebar) */
    .status-pill { display: flex; align-items: center; gap: 8px; padding: 0.35rem 0; font-size: 0.8rem; }
    .sdot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; box-shadow: 0 0 0 2px rgba(255,255,255,0.8); }
    .sdot-g { background: #10b981; } /* green = OK   */
    .sdot-r { background: #ef4444; } /* red   = error */
    .sdot-a { background: #f59e0b; } /* amber = pending */

    /* Success banner */
    .success-bar {
        display: flex; align-items: center; gap: 12px;
        padding: 1rem 1.5rem;
        background: linear-gradient(90deg, #f0fdf4 0%, #ffffff 100%);
        border: 1px solid #a7f3d0;
        border-radius: 12px; margin-bottom: 2rem;
        font-size: 0.9rem; color: #065f46; font-weight: 500;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.05);
    }
    .success-icon {
        display: flex; align-items: center; justify-content: center;
        width: 24px; height: 24px; background: #10b981; color: white;
        border-radius: 50%; font-size: 12px; font-weight: bold;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3. COLOR / FONT CONSTANTS — reuse these in your own inline styles so new
#    components you add stay visually consistent with the theme.
# ══════════════════════════════════════════════════════════════════════════════
COLORS = {
    "bg":            "#f8fafc",  # page background
    "bg_dot":        "#cbd5e1",  # background grid dot color
    "card_bg":       "#ffffff",
    "text_primary":  "#0f172a",
    "text_body":     "#334155",
    "text_muted":    "#64748b",
    "text_faint":    "#94a3b8",
    "border":        "#e2e8f0",
    "border_strong": "#cbd5e1",
    "accent":        "#2563eb",  # primary blue
    "accent_dark":   "#1d4ed8",
    "accent_bg":     "#eff6ff",  # light blue chip/badge bg
    "accent_border": "#bfdbfe",
    "success":       "#10b981",
    "success_text":  "#065f46",
    "success_bg":    "#f0fdf4",
    "success_border": "#a7f3d0",
    "error":         "#dc2626",
    "error_text":    "#7f1d1d",
    "error_bg":      "#fef2f2",
    "error_border":  "#fecaca",
    "warning":       "#f59e0b",
    "warning_text":  "#b45309",
    "warning_bg":    "#fffbeb",
    "warning_border": "#fef3c7",
}

FONTS = {
    "sans": "'Inter', -apple-system, sans-serif",   # body text, inputs, buttons
    "serif": "'Lora', serif",                        # titles, headings, emphasis
    "mono": "'JetBrains Mono', monospace",            # numbers, badges, code-like bits
}


# ══════════════════════════════════════════════════════════════════════════════
# 4. COMPONENT HELPERS — small wrappers so you don't have to hand-write the
#    HTML for the most common pieces. Each one just calls st.markdown(...,
#    unsafe_allow_html=True) under the hood using the classes above.
# ══════════════════════════════════════════════════════════════════════════════

def badge(text: str):
    """Small uppercase pill, e.g. above a hero title."""
    st.markdown(f"<span class='chi-badge'>{text}</span>", unsafe_allow_html=True)


def hero(badge_text: str, title: str, subtitle: str):
    """Centered hero block: badge + big serif title + muted subtitle."""
    st.markdown(f"""
    <div style='text-align:center; margin-bottom: 3.5rem;'>
      <div class='chi-badge'>{badge_text}</div>
      <div class='chi-title'>{title}</div>
      <div class='chi-subtitle'>{subtitle}</div>
    </div>""", unsafe_allow_html=True)


def divider(label: str = ""):
    """Horizontal rule with a centered uppercase label, e.g. ──── SECTION ────"""
    st.markdown(f"<div class='fancy-divider'><span>{label}</span></div>", unsafe_allow_html=True)


def stat_box(value, label: str):
    """Single stat tile: big mono number on top, small uppercase label below.
    Use inside st.columns(...) to build a stat grid."""
    st.markdown(f"""
    <div class='stat-box'>
      <div class='stat-num'>{value}</div>
      <div class='stat-txt'>{label}</div>
    </div>""", unsafe_allow_html=True)


def status_pill(label: str, status: str = "ok"):
    """Colored-dot status row, e.g. for a sidebar system-status list.
    status: 'ok' (green) | 'error' (red) | 'pending' (amber)"""
    cls = {"ok": "sdot-g", "error": "sdot-r", "pending": "sdot-a"}.get(status, "sdot-g")
    st.markdown(
        f"<div class='status-pill'><span class='sdot {cls}'></span>"
        f"<span style='color:#334155;font-weight:500;'>{label}</span></div>",
        unsafe_allow_html=True,
    )


def success_bar(message: str, right_label: str = "", right_sublabel: str = ""):
    """Green success banner with a checkmark icon. Optional right-aligned
    mono label + sublabel (e.g. model name + token count in the original)."""
    right = ""
    if right_label or right_sublabel:
        right = f"<div style='margin-left:auto;text-align:right;'><div style='font-family:JetBrains Mono,monospace;font-size:0.75rem;color:#059669;font-weight:600;'>{right_label}</div><div style='font-size:0.7rem;color:#10b981;margin-top:2px;'>{right_sublabel}</div></div>"
    st.markdown(f"""
    <div class='success-bar'>
      <div class='success-icon'>✓</div>
      <div>{message}</div>
      {right}
    </div>""", unsafe_allow_html=True)


def error_box(title: str, message: str, footnote: str = ""):
    """Red error panel with mono-font message body and optional footnote."""
    foot = ""
    if footnote:
        foot = f"""
        <div style='font-size:0.85rem;color:#991b1b;margin-top:1rem;padding-top:1rem;border-top:1px solid #fca5a5;'>
          {footnote}
        </div>"""
    st.markdown(f"""
    <div style='background:#fef2f2;border:1px solid #fecaca;border-radius:12px;
                padding:1.8rem 2rem;margin-top:1.5rem;box-shadow:0 4px 12px rgba(239, 68, 68, 0.05);'>
      <div style='font-size:1.05rem;font-weight:700;color:#dc2626;margin-bottom:0.6rem;'>
        {title}
      </div>
      <div style='font-size:0.9rem;color:#7f1d1d;font-family:JetBrains Mono,monospace;
                  white-space:pre-wrap;word-break:break-all;line-height:1.5;'>{message}</div>
      {foot}
    </div>""", unsafe_allow_html=True)


def empty_state(eyebrow: str, message: str):
    """Dashed-border placeholder panel for an empty / idle state."""
    st.markdown(f"""
    <div style='text-align:center;padding:5rem 2rem;
                background:#f8fafc;border:2px dashed #e2e8f0;border-radius:16px;
                margin-top:2rem;'>
      <div style='font-size:0.85rem;font-weight:700;letter-spacing:0.18em;
                  text-transform:uppercase;color:#64748b;margin-bottom:1rem;'>
        {eyebrow}
      </div>
      <div style='font-size:1.05rem;color:#475569;'>
        {message}
      </div>
    </div>""", unsafe_allow_html=True)


def answer_box_label(icon: str, label: str):
    """Renders just the label header of an 'answer box' (icon + serif label).
    Follow this with answer_body() for the actual content, since Streamlit's
    own markdown renderer needs to run separately for things like bold/lists
    inside the body (see note in answer_body)."""
    st.markdown(f"""
    <div class='answer-box'>
        <div class='answer-box-label'>
            <span style='font-size:1.2rem;'>{icon}</span> {label}
        </div>
    </div>""", unsafe_allow_html=True)


def answer_body(markdown_text: str):
    """Renders body content (plain st.markdown, so **bold**, lists, etc. all
    work normally) inside a left-bordered container that visually continues
    from answer_box_label() above it."""
    st.markdown("<div class='generated-answer-container'>", unsafe_allow_html=True)
    st.markdown(markdown_text)
    st.markdown("</div>", unsafe_allow_html=True)


def card_entry(rank, title: str, badges: list[tuple[str, str]], quote: str = "", progress_pct: int | None = None):
    """Generic version of the original 'paper-entry' bibliography card.

    rank:          number or string shown in the left rail (e.g. 1, "01", "★")
    title:         card heading (serif)
    badges:        list of (text, variant) tuples. variant is "year", "score",
                    or "" for the neutral gray style.
    quote:         optional italic excerpt/snippet line
    progress_pct:  optional 0-100 int to render a thin progress bar under the card
    """
    rank_str = f"{rank:02d}" if isinstance(rank, int) else str(rank)
    badges_html = "".join(
        f"<span class='pmeta-badge {variant}'>{text}</span>" for text, variant in badges
    )
    quote_html = f"<div class='paper-quote'>“...{quote}...”</div>" if quote else ""
    progress_html = ""
    if progress_pct is not None:
        progress_html = f"""
        <div style='margin-top:1rem;height:4px;background:#f1f5f9;border-radius:99px;width:150px;overflow:hidden;'>
          <div style='height:100%;width:{progress_pct}%;background:#2563eb;border-radius:99px;'></div>
        </div>"""
    st.markdown(f"""
    <div class='paper-entry'>
      <div class='paper-rank-num'>{rank_str}</div>
      <div class='paper-body'>
        <div class='paper-meta-row'>{badges_html}</div>
        <div class='paper-title-txt'>{title}</div>
        {quote_html}
        {progress_html}
      </div>
    </div>""", unsafe_allow_html=True)


def chip(text: str, warning: bool = False):
    """Small mono tag. Pass warning=True for the amber 'alert' variant."""
    style = "color:#b45309;border-color:#fef3c7;background:#fffbeb;" if warning else ""
    st.markdown(f"<span class='qchip' style='{style}'>{text}</span>", unsafe_allow_html=True)


def sidebar_brand(eyebrow: str, title_html: str):
    """Sidebar header block: small uppercase eyebrow + serif title.
    title_html supports <br> for a 2-line title, matching the original."""
    st.markdown(f"""
    <div style='padding:1.5rem 0 0.8rem;'>
      <div style='font-size:0.65rem;font-weight:700;letter-spacing:0.18em;
                  text-transform:uppercase;color:#2563eb;margin-bottom:6px;'>
        {eyebrow}
      </div>
      <div style='font-family:"Inter",-apple-system,sans-serif;font-size:1.4rem;font-weight:700;color:#0f172a;line-height:1.2;letter-spacing:-0.02em;'>
        {title_html}
      </div>
    </div>
    <hr style='border:none;border-top:1px solid #e2e8f0;margin:0.5rem 0;'>
    """, unsafe_allow_html=True)


def sidebar_section_label(text: str):
    """Small uppercase section label used inside the sidebar (e.g. above a
    list of key/value rows or before an expander)."""
    st.markdown(
        f"<p style='font-size:0.65rem;font-weight:700;letter-spacing:0.12em;"
        f"text-transform:uppercase;color:#64748b;margin-bottom:0.8rem;margin-top:1rem;'>"
        f"{text}</p>", unsafe_allow_html=True,
    )


def sidebar_kv_row(label: str, value: str):
    """One label/value row with a bottom hairline, for a sidebar info list."""
    st.markdown(f"""
    <div style='display:flex;justify-content:space-between;padding:0.35rem 0;
                border-bottom:1px solid #f1f5f9;'>
      <span style='font-size:0.8rem;color:#475569;'>{label}</span>
      <span style='font-size:0.75rem;color:#0f172a;font-family:JetBrains Mono,monospace;
                   font-weight:500;max-width:110px;overflow:hidden;text-overflow:ellipsis;
                   white-space:nowrap;text-align:right;'>{value}</span>
    </div>""", unsafe_allow_html=True)


def sidebar_hr():
    st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:1.2rem 0;'>",
                unsafe_allow_html=True)
