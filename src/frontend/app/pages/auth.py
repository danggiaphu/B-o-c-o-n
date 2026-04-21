from __future__ import annotations

import streamlit as st

from ..services.api_client import ApiClient


def render_login(api_base_url: str) -> None:
    # Center-column layout
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown(
            """
            <div class="card" style="margin-top:1rem">
              <div class="card-title" style="font-size:1.2rem;justify-content:center;margin-bottom:1.2rem">
                🔐 Đăng nhập hệ thống
              </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form"):
            username = st.text_input("Tên đăng nhập", placeholder="admin hoặc user")
            password = st.text_input("Mật khẩu", type="password", placeholder="••••••••")
            submitted = st.form_submit_button("Đăng nhập", use_container_width=True)

        st.markdown(
            '<div style="text-align:center;margin-top:0.6rem;font-size:0.8rem;color:#64748b">'
            'Tài khoản mặc định: <code>admin / admin123</code> · <code>user / user123</code>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if not submitted:
        return

    try:
        client = ApiClient(api_base_url)
        data = client.login(username=username, password=password)
        st.session_state["token"] = data["token"]
        st.session_state["username"] = data["username"]
        st.session_state["role"] = data["role"]
        st.success("✅ Đăng nhập thành công!")
        st.rerun()
    except Exception as exc:  # noqa: BLE001
        st.error(f"❌ Đăng nhập thất bại: {exc}")
