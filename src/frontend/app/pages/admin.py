from __future__ import annotations

import pandas as pd
import streamlit as st

from ..services.api_client import ApiClient
from ..ui.components import card_close, card_open, show_metric_row


def render_admin_console(api_base_url: str, token: str) -> None:
    client = ApiClient(api_base_url, token=token)

    # ── System Overview ──────────────────────────────────────────────
    card_open("Tổng quan hệ thống", "📊")
    try:
        stats = client.admin_stats()
        by_dir = client.admin_prediction_direction_stats()
        show_metric_row(stats)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Phân bổ Predictions theo hướng**")
            if by_dir:
                df_dir = pd.DataFrame(by_dir)
                df_dir["direction"] = df_dir["direction"].map(
                    {"drug_to_disease": "💊→🦠 Thuốc→Bệnh", "disease_to_drug": "🦠→💊 Bệnh→Thuốc"}
                ).fillna(df_dir["direction"])
                st.dataframe(df_dir, use_container_width=True, hide_index=True)
            else:
                st.info("Chưa có dữ liệu.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Không tải được thống kê: {exc}")
    card_close()

    # ── Management Tabs ──────────────────────────────────────────────
    tab_drugs, tab_diseases, tab_links, tab_logs = st.tabs(
        ["💊 Quản lý Thuốc", "🦠 Quản lý Bệnh", "🔗 Liên kết", "📋 Prediction Logs"]
    )

    # -- Drugs --
    with tab_drugs:
        card_open("Thêm / Cập nhật thuốc", "💊")
        with st.form("admin_save_drug"):
            c1, c2 = st.columns(2)
            drug_id = c1.number_input("Drug ID", min_value=0, step=1)
            drug_name = c2.text_input("Tên thuốc")
            c3, c4 = st.columns(2)
            external_id = c3.text_input("External ID (tùy chọn)")
            smiles = c4.text_input("SMILES (tùy chọn)")
            if st.form_submit_button("💾 Lưu thuốc", use_container_width=True):
                try:
                    client.admin_save_drug(
                        drug_id=int(drug_id),
                        name=str(drug_name),
                        external_id=str(external_id) if external_id else None,
                        smiles=str(smiles) if smiles else None,
                    )
                    st.success("✅ Đã lưu thuốc.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"❌ {exc}")
        card_close()

        card_open("Danh sách thuốc", "📋")
        if st.button("🔄 Tải danh sách thuốc", use_container_width=True, key="load_drugs"):
            try:
                rows = client.list_drugs(limit=500)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ {exc}")
        card_close()

    # -- Diseases --
    with tab_diseases:
        card_open("Thêm / Cập nhật bệnh", "🦠")
        with st.form("admin_save_disease"):
            c1, c2 = st.columns(2)
            disease_id = c1.number_input("Disease ID", min_value=0, step=1)
            disease_name = c2.text_input("Tên bệnh")
            if st.form_submit_button("💾 Lưu bệnh", use_container_width=True):
                try:
                    client.admin_save_disease(disease_id=int(disease_id), name=str(disease_name))
                    st.success("✅ Đã lưu bệnh.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"❌ {exc}")
        card_close()

        card_open("Danh sách bệnh", "📋")
        if st.button("🔄 Tải danh sách bệnh", use_container_width=True, key="load_diseases"):
            try:
                rows = client.list_diseases(limit=500)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ {exc}")
        card_close()

    # -- Links --
    with tab_links:
        card_open("Thêm liên kết Thuốc–Bệnh", "🔗")
        with st.form("admin_save_link"):
            c1, c2 = st.columns(2)
            drug_id_l = c1.number_input("Drug ID", min_value=0, step=1, key="lid_drug")
            disease_id_l = c2.number_input("Disease ID", min_value=0, step=1, key="lid_disease")
            if st.form_submit_button("➕ Thêm liên kết", use_container_width=True):
                try:
                    client.admin_save_link(drug_id=int(drug_id_l), disease_id=int(disease_id_l))
                    st.success("✅ Đã thêm liên kết.")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"❌ {exc}")
        card_close()

        card_open("Danh sách liên kết", "📋")
        if st.button("🔄 Tải danh sách liên kết", use_container_width=True, key="load_links"):
            try:
                rows = client.list_links(limit=1000)
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ {exc}")
        card_close()

    # -- Prediction Logs --
    with tab_logs:
        card_open("Lịch sử dự đoán toàn hệ thống", "📋")
        c1, c2 = st.columns([1, 4])
        with c1:
            if st.button("🔄 Tải logs", use_container_width=True, key="load_logs"):
                st.session_state.pop("admin_predictions", None)
        with c2:
            st.markdown(
                '<div style="font-size:0.83rem;color:#64748b;padding-top:0.4rem">'
                "Hiển thị tối đa 400 bản ghi gần nhất.</div>",
                unsafe_allow_html=True,
            )

        if st.session_state.get("admin_predictions") is None:
            try:
                st.session_state["admin_predictions"] = client.admin_predictions(limit=400)
            except Exception as exc:  # noqa: BLE001
                st.error(f"❌ {exc}")
                st.session_state["admin_predictions"] = []

        rows = st.session_state.get("admin_predictions", [])
        if rows:
            df = pd.DataFrame(rows)
            if "score" in df.columns:
                df["score (%)"] = (pd.to_numeric(df["score"], errors="coerce") * 100).map(
                    lambda x: f"{x:.2f}%"
                )
            if "direction" in df.columns:
                df["direction"] = df["direction"].map(
                    {"drug_to_disease": "💊→🦠 Thuốc→Bệnh", "disease_to_drug": "🦠→💊 Bệnh→Thuốc"}
                ).fillna(df["direction"])
            display_cols = [c for c in ["user_id", "direction", "input_name", "target_name", "score (%)", "timestamp"] if c in df.columns]
            st.dataframe(df[display_cols] if display_cols else df, use_container_width=True, hide_index=True)
        else:
            st.markdown('<div class="info-box">Chưa có dữ liệu.</div>', unsafe_allow_html=True)
        card_close()
