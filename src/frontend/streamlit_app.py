from __future__ import annotations

from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Also ensure project root is on path for cross-package imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from frontend.app.config import API_DEFAULT  # noqa: E402
from frontend.app.pages.admin import render_admin_console  # noqa: E402
from frontend.app.pages.auth import render_login  # noqa: E402
from frontend.app.pages.user import render_user_workspace  # noqa: E402
from frontend.app.services.api_client import ApiClient  # noqa: E402
from frontend.app.state import clear_auth_state, current_role, is_authenticated  # noqa: E402
from frontend.app.ui.theme import apply_theme, render_hero  # noqa: E402


def main() -> None:
    apply_theme()

    # ── Sidebar ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            """
            <div style="padding:0.5rem 0 0.3rem 0">
              <div style="font-size:1.3rem;font-weight:800;color:#0d9488">💊 MedLink AI</div>
              <div style="font-size:0.75rem;color:#64748b;margin-top:0.1rem">Drug · Disease Intelligence</div>
            </div>
            <hr style="border:none;border-top:1px solid #e2e8f0;margin:0.7rem 0">
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### ⚙️ Cấu hình")
        api_base_url = st.text_input(
            "Backend URL",
            value=API_DEFAULT,
            help="URL gốc của FastAPI backend, ví dụ: http://127.0.0.1:8000/api",
        )

        col_check, _ = st.columns([1, 1])
        with col_check:
            if st.button("🔌 Kiểm tra API", use_container_width=True):
                try:
                    status = ApiClient(api_base_url).health()
                    st.success(f"✅ Online — {status.get('status', 'ok')}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"❌ Offline: {exc}")

        st.markdown("<hr style='border:none;border-top:1px solid #e2e8f0;margin:0.7rem 0'>", unsafe_allow_html=True)

        if is_authenticated():
            username = str(st.session_state.get("username", ""))
            role = current_role()
            role_icon = "👑" if role == "admin" else "👤"
            st.markdown(
                f"""
                <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;
                            padding:0.6rem 0.8rem;margin-bottom:0.8rem">
                  <div style="font-weight:700;color:#15803d">{role_icon} {username}</div>
                  <div style="font-size:0.76rem;color:#64748b;text-transform:capitalize">{role}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### 📌 Điều hướng")
            nav_options = ["🔬 Dự đoán", "🛠️ Admin Console"] if role == "admin" else ["🔬 Dự đoán"]
            menu = st.radio("", nav_options, index=0, label_visibility="collapsed")

            if st.button("🚪 Đăng xuất", use_container_width=True):
                clear_auth_state()
                st.rerun()
        else:
            menu = "__login__"
            role = None
            st.markdown(
                '<div class="info-box">Vui lòng đăng nhập để sử dụng hệ thống.</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <hr style="border:none;border-top:1px solid #e2e8f0;margin:0.7rem 0">
            <div style="font-size:0.73rem;color:#94a3b8;text-align:center">
              MedLink AI v1.0 · FuzzyGCN<br>Drug–Disease Link Prediction
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Main content ───────────────────────────────────────────────────
    render_hero(role)

    if menu == "__login__":
        render_login(api_base_url)
        return

    token = str(st.session_state.get("token", ""))
    if not token:
        render_login(api_base_url)
        return

    if "Dự đoán" in menu:
        render_user_workspace(api_base_url, token)
        return

    if "Admin Console" in menu:
        if current_role() != "admin":
            st.error("❌ Bạn không có quyền Admin.")
            return
        render_admin_console(api_base_url, token)


if __name__ == "__main__":
    main()

