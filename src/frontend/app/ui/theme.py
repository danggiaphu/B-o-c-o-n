from __future__ import annotations

import streamlit as st


def apply_theme() -> None:
    st.set_page_config(
        page_title="MedLink AI — Drug·Disease Intelligence",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Ghi chu ve cac nhom nut trong CSS ben duoi:
    # - `.stButton > button`: dung cho cac nut bam thuong tren giao dien
    #   nhu "Kiểm tra API", "Đăng xuất", "Tải lại", "Dự đoán tất cả",
    #   "Tải danh sách thuốc", "Tải danh sách bệnh", "Tải danh sách liên kết", "Tải logs".
    # - `.stFormSubmitButton > button`: dung cho cac nut gui form
    #   nhu "Đăng nhập", "Dự đoán", "Tra cứu", "Lưu thuốc", "Lưu bệnh", "Thêm liên kết".
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
          --bg:        #f0f4f9;
          --surface:   #ffffff;
          --border:    #e2e8f0;
          --text:      #1a202c;
          --muted:     #64748b;
          --primary:   #0d9488;
          --primary-2: #1e40af;
          --danger:    #dc2626;
          --known-bg:  #dcfce7;
          --known-fg:  #15803d;
          --pred-bg:   #eff6ff;
          --pred-fg:   #1d4ed8;
          --radius:    12px;
          --shadow:    0 2px 12px rgba(0,0,0,0.07);
          --shadow-lg: 0 8px 32px rgba(0,0,0,0.10);
        }

        html, body, [class*="css"] {
          font-family: 'Inter', sans-serif;
          color: var(--text);
        }

        .stApp {
          background: var(--bg);
        }

        /* ── Sidebar ── */
        section[data-testid="stSidebar"] {
          background: var(--surface);
          border-right: 1px solid var(--border);
        }
        section[data-testid="stSidebar"] .stMarkdown h3 {
          font-size: 0.78rem;
          font-weight: 700;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          color: var(--muted);
          margin: 1.2rem 0 0.4rem 0;
        }

        /* ── Hero banner ── */
        .hero-banner {
          background: linear-gradient(135deg, #0d9488 0%, #1e40af 100%);
          border-radius: var(--radius);
          padding: 1.4rem 1.8rem;
          margin-bottom: 1.2rem;
          color: #fff;
          box-shadow: var(--shadow-lg);
        }
        .hero-banner h1 {
          margin: 0 0 0.25rem 0;
          font-size: 1.65rem;
          font-weight: 800;
          letter-spacing: -0.03em;
          color: #fff;
        }
        .hero-banner p {
          margin: 0;
          opacity: 0.88;
          font-size: 0.93rem;
        }
        .hero-chip {
          display: inline-block;
          background: rgba(255,255,255,0.22);
          border: 1px solid rgba(255,255,255,0.35);
          color: #fff;
          border-radius: 999px;
          padding: 0.18rem 0.7rem;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.04em;
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
          font-size: 1rem;
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
          border: none !important;
          background: linear-gradient(90deg, var(--primary), var(--primary-2)) !important;
          color: #ffffff !important;
          font-weight: 600 !important;
          font-size: 0.9rem !important;
          padding: 0.5rem 1.2rem !important;
          transition: opacity 0.15s ease;
        }
        .stButton > button:hover {
          opacity: 0.88 !important;
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

        /* ── Input fields ── */
        .stTextInput > div > div > input, .stSelectbox > div > div {
          border-radius: 8px !important;
          border: 1px solid var(--border) !important;
        }
        .stTextInput label,
        .stSelectbox label,
        .stSlider label,
        .stNumberInput label,
        .stRadio label,
        .stMultiSelect label,
        .stCheckbox label {
          color: #000000 !important;
        }

        /* ── Tabs ── */
        .stNumberInput input {
          color: #ffffff !important;
          -webkit-text-fill-color: #ffffff !important;
        }
        .stNumberInput button {
          color: #ffffff !important;
        }
        .stSlider span,
        .stSlider p,
        .stSlider small,
        .stNumberInput span,
        .stNumberInput p,
        .stRadio span,
        .stRadio p {
          color: #000000 !important;
        }

        .stTabs [data-baseweb="tab-list"] {
          gap: 0.3rem;
          background: transparent;
          border-bottom: 2px solid var(--border);
          padding-bottom: 0;
        }
        .stTabs [data-baseweb="tab"] {
          border-radius: 8px 8px 0 0 !important;
          background: transparent !important;
          border: 1px solid transparent !important;
          color: var(--muted) !important;
          font-weight: 600 !important;
          font-size: 1rem !important;
          padding: 0.45rem 1rem !important;
        }
        .stTabs [aria-selected="true"] {
          background: var(--surface) !important;
          border-color: var(--border) !important;
          border-bottom-color: var(--surface) !important;
          color: var(--primary) !important;
        }

        /* ── Metric cards ── */
        div[data-testid="stHeading"] h1,
        div[data-testid="stHeading"] h2,
        div[data-testid="stHeading"] h3,
        div[data-testid="stHeading"] h4 {
          color: #000000 !important;
        }

        [data-testid="stMetric"] {
          background: var(--surface);
          border: 1px solid var(--border);
          border-radius: var(--radius);
          padding: 0.9rem 1rem;
          box-shadow: var(--shadow);
        }

        /* ── Known / Predicted badges ── */
        .badge-known {
          display: inline-block;
          background: #dcfce7;
          color: #14532d;
          border: 1px solid #4ade80;
          border-radius: 999px;
          padding: 0.15rem 0.65rem;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.03em;
        }
        .badge-pred {
          display: inline-block;
          background: #fef9c3;
          color: #713f12;
          border: 1px solid #facc15;
          border-radius: 999px;
          padding: 0.15rem 0.65rem;
          font-size: 0.72rem;
          font-weight: 700;
          letter-spacing: 0.03em;
        }

        /* ── Result table ── */
        .result-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
        .result-table thead tr { background: #f8fafc; }
        .result-table th {
          text-align: left;
          padding: 0.5rem 0.7rem;
          font-weight: 700;
          color: var(--muted);
          font-size: 0.74rem;
          letter-spacing: 0.05em;
          text-transform: uppercase;
          border-bottom: 1px solid var(--border);
        }
        .result-table td {
          padding: 0.5rem 0.7rem;
          border-bottom: 1px solid #f1f5f9;
          color: var(--text);
        }
        .result-table tr:last-child td { border-bottom: none; }
        .result-table tr:hover td { background: #f8fafc; }
        .score-bar-wrap { display: flex; align-items: center; gap: 0.5rem; }
        .score-bar-bg { flex: 1; background: #e2e8f0; border-radius: 4px; height: 6px; min-width: 60px; }
        .score-bar-fill { height: 6px; border-radius: 4px; background: linear-gradient(90deg, #0d9488, #1e40af); }
        .score-val { font-weight: 700; font-size: 0.82rem; color: var(--primary-2); white-space: nowrap; }

        /* ── Alert / info boxes ── */
        .info-box {
          background: #eff6ff;
          border-left: 3px solid #3b82f6;
          border-radius: 0 8px 8px 0;
          padding: 0.6rem 0.9rem;
          font-size: 0.85rem;
          color: #1e40af;
          margin: 0.5rem 0;
        }
        .warn-box {
          background: #fffbeb;
          border-left: 3px solid #f59e0b;
          border-radius: 0 8px 8px 0;
          padding: 0.6rem 0.9rem;
          font-size: 0.85rem;
          color: #92400e;
          margin: 0.5rem 0;
        }
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
