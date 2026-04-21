from __future__ import annotations

import streamlit as st


SESSION_KEYS = [
    "token",
    "username",
    "role",
    "history_rows",
    "admin_predictions",
]


def clear_auth_state() -> None:
    for key in SESSION_KEYS:
        st.session_state.pop(key, None)


def is_authenticated() -> bool:
    return bool(st.session_state.get("token"))


def current_role() -> str:
    return str(st.session_state.get("role", ""))
