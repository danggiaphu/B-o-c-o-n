from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.set_page_config(
        page_title="MedLink AI — Drug·Disease Intelligence",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
          --bg:         #0f1117;
          --surface:    #1a1d27;
          --surface-2:  #22263a;
          --border:     #2d3148;
          --text:       #e8eaf0;
          --muted:      #7c829e;
          --primary:    #0d9488;
          --primary-2:  #3b82f6;
          --danger:     #ef4444;
          --known-bg:   #0d2e1f;
          --known-fg:   #4ade80;
          --known-bd:   #166534;
          --pred-bg:    #172040;
          --pred-fg:    #93c5fd;
          --pred-bd:    #1e40af;
          --warn-bg:    #2d1a06;
          --warn-fg:    #fbbf24;
          --warn-bd:    #92400e;
          --radius:     12px;
          --shadow:     0 2px 12px rgba(0,0,0,0.35);
          --shadow-lg:  0 8px 32px rgba(0,0,0,0.5);
        }

        html, body, [class*="css"] {
          font-family: 'Inter', sans-serif !important;
          color: var(--text);
        }

        /* ── App background ── */
        .stApp { background: var(--bg) !important; }
        .main .block-container {
          padding-top: 1.2rem !important;
          padding-bottom: 2rem !important;
          max-width: 1200px;
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
          background: var(--surface) !important;
          border-right: 1px solid var(--border) !important;
        }
        section[data-testid="stSidebar"] * { color: var(--text) !important; }
        section[data-testid="stSidebar"] .stMarkdown h3 {
          font-size: 0.72rem !important;
          font-weight: 700 !important;
          letter-spacing: 0.1em !important;
          text-transform: uppercase !important;
          color: var(--muted) !important;
          margin: 1.2rem 0 0.4rem 0 !important;
        }
        section[data-testid="stSidebar"] input {
          background: var(--surface-2) !important;
          border: 1px solid var(--border) !important;
          color: var(--text) !important;
          border-radius: 8px !important;
        }

        /* ── Hero banner ── */
        .hero-banner {
          background: linear-gradient(135deg, #0d4f4a 0%, #1a2f6e 100%);
          border: 1px solid #1d3d6e;
          border-radius: var(--radius);
          padding: 1.4rem 1.8rem;
          margin-bottom: 1.2rem;
          box-shadow: var(--shadow-lg);
        }
        .hero-banner h1 {
          margin: 0 0 0.25rem 0;
          font-size: 1.6rem;
          font-weight: 800;
          letter-spacing: -0.03em;
          color: #fff;
        }
        .hero-banner p { margin: 0; opacity: 0.82; font-size: 0.9rem; color: #fff; }
        .hero-chip {
          display: inline-block;
          background: rgba(255,255,255,0.12);
          border: 1px solid rgba(255,255,255,0.25);
          color: #fff;
          border-radius: 999px;
          padding: 0.18rem 0.7rem;
          font-size: 0.7rem;
          font-weight: 700;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          margin-bottom: 0.5rem;
        }

        /* ── Cards ── */
        .card {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 1.2rem 1.4rem 1.3rem 1.4rem;
          margin-bottom: 1rem;
          box-shadow: var(--shadow);
        }
        .card-title {
          font-size: 0.98rem;
          font-weight: 700;
          color: var(--text);
          margin: 0 0 0.8rem 0;
          display: flex;
          align-items: center;
          gap: 0.4rem;
        }

        /* ── Buttons ── */
        .stButton > button {
          border-radius: 8px !important;
          border: 1px solid var(--border) !important;
          background: var(--surface-2) !important;
          color: var(--text) !important;
          font-weight: 600 !important;
          font-size: 0.88rem !important;
          padding: 0.45rem 1.1rem !important;
          transition: background 0.15s, border-color 0.15s;
        }
        .stButton > button:hover {
          background: var(--primary) !important;
          border-color: var(--primary) !important;
          color: #fff !important;
        }

        /* ── Form submit button ── */
        .stFormSubmitButton > button {
          border-radius: 8px !important;
          border: none !important;
          background: linear-gradient(90deg, var(--primary), var(--primary-2)) !important;
          color: #ffffff !important;
          font-weight: 700 !important;
          font-size: 0.95rem !important;
          width: 100% !important;
          padding: 0.6rem 1rem !important;
        }
        .stFormSubmitButton > button:hover { opacity: 0.88 !important; }

        /* ── Input fields ── */
        .stTextInput > div > div > input,
        .stSelectbox > div > div,
        .stMultiSelect > div > div {
          border-radius: 8px !important;
          border: 1px solid var(--border) !important;
          background: var(--surface-2) !important;
          color: var(--text) !important;
        }
        .stTextInput input, .stNumberInput input {
          color: var(--text) !important;
          -webkit-text-fill-color: var(--text) !important;
          background: var(--surface-2) !important;
        }

        /* ── Labels ── */
        .stTextInput label, .stSelectbox label, .stSlider label,
        .stNumberInput label, .stRadio label, .stMultiSelect label,
        .stCheckbox label {
          color: var(--muted) !important;
          font-size: 0.82rem !important;
          font-weight: 600 !important;
        }

        /* ── Slider & number input ── */
        .stSlider span, .stSlider p, .stSlider small,
        .stNumberInput span, .stNumberInput p, .stNumberInput button,
        .stRadio span, .stRadio p {
          color: var(--text) !important;
        }

        /* ── Radio buttons ── */
        div[role="radiogroup"] label p { color: var(--text) !important; }
        div[role="radiogroup"] label { color: var(--text) !important; }

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {
          gap: 0.2rem;
          background: transparent;
          border-bottom: 1px solid var(--border);
        }
        .stTabs [data-baseweb="tab"] {
          border-radius: 8px 8px 0 0 !important;
          background: transparent !important;
          border: 1px solid transparent !important;
          color: var(--muted) !important;
          font-weight: 600 !important;
          font-size: 0.9rem !important;
          padding: 0.4rem 0.9rem !important;
        }
        .stTabs [aria-selected="true"] {
          background: var(--surface) !important;
          border-color: var(--border) !important;
          border-bottom-color: var(--surface) !important;
          color: var(--primary) !important;
        }

        /* ── Headings ── */
        h1, h2, h3, h4, h5, h6,
        div[data-testid="stHeading"] h1,
        div[data-testid="stHeading"] h2,
        div[data-testid="stHeading"] h3,
        div[data-testid="stHeading"] h4 {
          color: var(--text) !important;
        }
        p, span, li, div { color: var(--text); }

        /* ── Metric cards ── */
        [data-testid="stMetric"] {
          background: var(--surface) !important;
          border: 1px solid var(--border) !important;
          border-radius: var(--radius) !important;
          padding: 0.9rem 1rem !important;
          box-shadow: var(--shadow) !important;
        }
        [data-testid="stMetricLabel"],
        [data-testid="stMetricValue"] { color: var(--text) !important; }

        /* ── Dataframe ── */
        .stDataFrame, iframe {
          border-radius: var(--radius) !important;
          border: 1px solid var(--border) !important;
        }

        /* ── Select dropdown ── */
        [data-baseweb="select"] { background: var(--surface-2) !important; }
        [data-baseweb="menu"] {
          background: var(--surface-2) !important;
          border: 1px solid var(--border) !important;
        }
        [data-baseweb="option"] {
          background: var(--surface-2) !important;
          color: var(--text) !important;
        }
        [data-baseweb="option"]:hover { background: var(--surface) !important; }

        /* ── Expander ── */
        .streamlit-expanderHeader {
          background: var(--surface) !important;
          border: 1px solid var(--border) !important;
          border-radius: 8px !important;
          color: var(--text) !important;
          font-weight: 600 !important;
        }
        .streamlit-expanderContent {
          background: var(--surface) !important;
          border: 1px solid var(--border) !important;
          border-top: none !important;
        }

        /* ── Badges ── */
        .badge-known {
          display: inline-block;
          background: var(--known-bg);
          color: var(--known-fg);
          border: 1px solid var(--known-bd);
          border-radius: 999px;
          padding: 0.15rem 0.65rem;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.03em;
        }
        .badge-pred {
          display: inline-block;
          background: var(--pred-bg);
          color: var(--pred-fg);
          border: 1px solid var(--pred-bd);
          border-radius: 999px;
          padding: 0.15rem 0.65rem;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.03em;
        }

        /* ── Result table ── */
        .result-table { width: 100%; border-collapse: collapse; font-size: 0.87rem; }
        .result-table thead tr { background: var(--surface-2); }
        .result-table th {
          text-align: left;
          padding: 0.5rem 0.75rem;
          font-weight: 700;
          color: var(--muted);
          font-size: 0.72rem;
          letter-spacing: 0.06em;
          text-transform: uppercase;
          border-bottom: 1px solid var(--border);
        }
        .result-table td {
          padding: 0.5rem 0.75rem;
          border-bottom: 1px solid var(--border);
          color: var(--text);
        }
        .result-table tr:last-child td { border-bottom: none; }
        .result-table tr:hover td { background: var(--surface-2); }

        .score-bar-wrap { display: flex; align-items: center; gap: 0.5rem; }
        .score-bar-bg {
          flex: 1;
          background: var(--surface-2);
          border-radius: 4px;
          height: 5px;
          min-width: 60px;
        }
        .score-bar-fill {
          height: 5px;
          border-radius: 4px;
          background: linear-gradient(90deg, var(--primary), var(--primary-2));
        }
        .score-val {
          font-weight: 700;
          font-size: 0.8rem;
          color: var(--primary-2);
          white-space: nowrap;
        }

        /* ── Info / warn boxes ── */
        .info-box {
          background: var(--pred-bg);
          border-left: 3px solid var(--primary-2);
          border-radius: 0 8px 8px 0;
          padding: 0.6rem 0.9rem;
          font-size: 0.84rem;
          color: var(--pred-fg);
          margin: 0.5rem 0;
        }
        .warn-box {
          background: var(--warn-bg);
          border-left: 3px solid var(--warn-fg);
          border-radius: 0 8px 8px 0;
          padding: 0.6rem 0.9rem;
          font-size: 0.84rem;
          color: var(--warn-fg);
          margin: 0.5rem 0;
        }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--muted); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(role: str | None) -> None:
    chip = "Admin Console" if role == "admin" else ("Workspace" if role else "Welcome")
    desc = (
        "Quản lý hệ thống, dữ liệu và theo dõi toàn bộ prediction logs."
        if role == "admin"
        else "Nhập tên Thuốc hoặc Bệnh để AI dự đoán liên kết và kiểm tra dữ liệu thực tế."
    )
    st.markdown(
        f"""
        <div class="hero-banner">
          <span class="hero-chip">{chip}</span>
          <h1>💊 MedLink AI</h1>
          <p>{desc}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
