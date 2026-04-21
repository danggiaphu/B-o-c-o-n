from __future__ import annotations

import io
import re

import streamlit as st

from ..services.api_client import ApiClient
from ..ui.components import card_open, card_close, show_history_table, show_split_result_table


def _load_dropdown_values(key: str, fetcher):
    if st.session_state.get(key) is None:
        try:
            st.session_state[key] = fetcher()
        except Exception as exc:
            st.error(f"Không tải được dữ liệu dropdown: {exc}")
            st.session_state[key] = []
    return st.session_state[key]


def _clean_filename(value: str) -> str:
    filename = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return filename.strip("_.-") or "molecule"


def _render_molecule_preview(smiles: str | None, name: str) -> bytes | None:
    if not smiles:
        st.info("Không có SMILES để hiển thị cấu trúc phân tử.")
        return None

    try:
        from draw import draw_molecule_image  # type: ignore
    except Exception as exc:  # noqa: BLE001
        st.error(f"Không thể vẽ cấu trúc phân tử. RDKit chưa được cài đặt hoặc không khả dụng: {exc}")
        return None

    try:
        image = draw_molecule_image(smiles, size=(320, 320))
        st.image(image, caption=f"Cấu trúc phân tử của {name}", use_column_width=False)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Không thể tạo ảnh phân tử: {exc}")
        return None


def _render_molecule_download(image_bytes: bytes | None, name: str) -> None:
    if not image_bytes:
        return

    st.download_button(
        "Tải ảnh PNG",
        data=image_bytes,
        file_name=f"{_clean_filename(name)}.png",
        mime="image/png",
    )


def render_user_workspace(api_base_url: str, token: str) -> None:
    client = ApiClient(api_base_url, token=token)

    tab_a, tab_b, tab_c, tab_d = st.tabs([
        "💊 Thuốc → Bệnh",
        "🦠 Bệnh → Thuốc",
        "🧬 Protein",
        "📋 Lịch sử",
    ])

    # ── Tab A: Drug → Disease ──────────────────────────────────────────
    with tab_a:
        card_open("Tìm bệnh tiềm năng từ tên thuốc", "💊")
        st.markdown(
            '<div class="info-box">Chọn số lượng kết quả trước, sau đó chọn tên thuốc để AI dự đoán các bệnh có thể liên quan. '
            'Kết quả <span class="badge-known" style="font-size:0.78rem">✅ Đã biết</span> '
            'là liên kết có trong dữ liệu huấn luyện; '
            '<span class="badge-pred" style="font-size:0.78rem">🔬 Dự đoán</span> '
            'là phát hiện mới từ AI.</div>',
            unsafe_allow_html=True,
        )

        drugs = _load_dropdown_values("drug_options", lambda: client.list_drugs(limit=1000))
        drug_image_bytes: bytes | None = None

        # ── Bước 1: Chọn số lượng kết quả ──
        st.markdown('<h4 style="color:#0d9488;margin-bottom:0.5rem">📊 Bước 1: Cấu hình kết quả</h4>', unsafe_allow_html=True)
        config_cols = st.columns([2, 2])
        with config_cols[0]:
            top_k = st.slider("Số kết quả (Top-K)", min_value=1, max_value=50, value=10, step=1, key="topk_d2d")
        with config_cols[1]:
            threshold = st.slider(
                "Ngưỡng tối thiểu", min_value=0.0, max_value=1.0, value=0.0, step=0.01,
                key="thr_d2d", help="Chỉ hiển thị kết quả có điểm ≥ ngưỡng này"
            )

        # ── Bước 2: Chọn thuốc ──
        st.markdown('<h4 style="color:#0d9488;margin-bottom:0.5rem">💊 Bước 2: Chọn thuốc</h4>', unsafe_allow_html=True)
        
        with st.form("form_drug_to_disease"):
            drug_name = st.selectbox(
                "Tên thuốc",
                options=[item["name"] for item in drugs],
                index=0 if drugs else -1,
                key="sel_d2d",
            )
            drug_smiles = next((item.get("smiles") for item in drugs if item["name"] == drug_name), None)
            st.markdown('<h5 style="color:#000000;margin-bottom:0.7rem">Cấu trúc phân tử</h5>', unsafe_allow_html=True)
            drug_image_bytes = _render_molecule_preview(drug_smiles, drug_name)
            submitted = st.form_submit_button("🔍 Dự đoán", use_container_width=True)

        _render_molecule_download(drug_image_bytes, drug_name)

        # ── Bước 3: Hiển thị kết quả ──
        if submitted:
            if not drug_name or not str(drug_name).strip():
                st.warning("Vui lòng chọn tên thuốc.")
            else:
                with st.spinner("Đang xử lý..."):
                    try:
                        data = client.predict_drug_to_disease(
                            name=drug_name, top_k=top_k, threshold=threshold
                        )
                        results = data.get("results", [])
                        st.session_state.pop("history_rows", None)  # invalidate cache
                        st.markdown(
                            f'<div style="font-size:0.82rem;color:#64748b;margin:0.3rem 0 0.7rem 0">'
                            f'<strong>📋 Kết quả cho:</strong> {data.get("input_name", drug_name)} '
                            f'(Top-K: {top_k}, Ngưỡng: {threshold})'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        show_split_result_table(results, entity_label="Tên bệnh")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"❌ Lỗi dự đoán: {exc}")
        card_close()

    # ── Tab B: Disease → Drug ──────────────────────────────────────────
    with tab_b:
        card_open("Tìm thuốc tiềm năng từ tên bệnh", "🦠")
        st.markdown(
            '<div class="info-box">Chọn số lượng kết quả trước, sau đó chọn tên bệnh để AI dự đoán các thuốc có thể điều trị. '
            'Kết quả ưu tiên các thuốc đã có liên kết xác nhận trong dataset.</div>',
            unsafe_allow_html=True,
        )

        diseases = _load_dropdown_values("disease_options", lambda: [item["name"] for item in client.list_diseases(limit=1000)])

        # ── Bước 1: Chọn số lượng kết quả ──
        st.markdown('<h4 style="color:#0d9488;margin-bottom:0.5rem">📊 Bước 1: Cấu hình kết quả</h4>', unsafe_allow_html=True)
        config_cols = st.columns([2, 2])
        with config_cols[0]:
            top_k = st.slider("Số kết quả (Top-K)", min_value=1, max_value=50, value=10, step=1, key="topk_dis2d")
        with config_cols[1]:
            threshold = st.slider(
                "Ngưỡng tối thiểu", min_value=0.0, max_value=1.0, value=0.0, step=0.01,
                key="thr_dis2d", help="Chỉ hiển thị kết quả có điểm ≥ ngưỡng này"
            )

        # ── Bước 2: Chọn bệnh ──
        st.markdown('<h4 style="color:#0d9488;margin-bottom:0.5rem">🦠 Bước 2: Chọn bệnh</h4>', unsafe_allow_html=True)

        with st.form("form_disease_to_drug"):
            disease_name = st.selectbox("Tên bệnh", options=diseases, index=0 if diseases else -1, key="sel_dis2d")
            submitted = st.form_submit_button("🔍 Dự đoán", use_container_width=True)

        # ── Bước 3: Hiển thị kết quả ──
        if submitted:
            if not disease_name or not str(disease_name).strip():
                st.warning("Vui lòng chọn tên bệnh.")
            else:
                with st.spinner("Đang xử lý..."):
                    try:
                        data = client.predict_disease_to_drug(
                            name=disease_name, top_k=top_k, threshold=threshold
                        )
                        results = data.get("results", [])
                        st.session_state.pop("history_rows", None)  # invalidate cache
                        st.markdown(
                            f'<div style="font-size:0.82rem;color:#64748b;margin:0.3rem 0 0.7rem 0">'
                            f'<strong>📋 Kết quả cho:</strong> {data.get("input_name", disease_name)} '
                            f'(Top-K: {top_k}, Ngưỡng: {threshold})'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        show_split_result_table(results, entity_label="Tên thuốc")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"❌ Lỗi dự đoán: {exc}")
        card_close()

    # ── Tab C: Protein ─────────────────────────────────────────────────
    with tab_c:
        card_open("Tra cứu Protein", "🧬")
        st.markdown(
            '<div class="info-box">Chọn số lượng kết quả trước, sau đó chọn Protein để tra cứu các thuốc và bệnh liên quan theo dataset.</div>',
            unsafe_allow_html=True,
        )

        proteins = _load_dropdown_values("protein_options", lambda: client.list_proteins(limit=1000))
        protein_names = [item["name"] for item in proteins]

        # ── Bước 1: Chọn số lượng kết quả ──
        st.markdown('<h4 style="color:#0d9488;margin-bottom:0.5rem">📊 Bước 1: Cấu hình kết quả</h4>', unsafe_allow_html=True)
        config_cols = st.columns([2, 2])
        with config_cols[0]:
            top_k = st.slider("Giới hạn kết quả", min_value=1, max_value=200, value=50, step=1, key="topk_protein")
        with config_cols[1]:
            show_sequence = st.checkbox("Hiển thị chuỗi protein", value=False, key="show_protein_sequence")

        # ── Bước 2: Chọn protein ──
        st.markdown('<h4 style="color:#0d9488;margin-bottom:0.5rem">🧬 Bước 2: Chọn Protein</h4>', unsafe_allow_html=True)

        with st.form("form_protein"):
            protein_name = st.selectbox(
                "Protein",
                options=protein_names,
                index=0 if protein_names else -1,
                key="sel_protein",
            )
            submitted = st.form_submit_button("🔍 Tra cứu", use_container_width=True)

        # ── Bước 3: Hiển thị kết quả ──
        if submitted:
            if not protein_name or not str(protein_name).strip():
                st.warning("Vui lòng chọn Protein.")
            else:
                protein_row = next((item for item in proteins if item["name"] == protein_name), None)
                if protein_row is None:
                    st.warning("Protein không tồn tại trong dữ liệu.")
                else:
                    with st.spinner("Đang tìm protein..."):
                        data = client.get_protein_links(protein_id=int(protein_row["id"]))
                        drugs = data.get("drugs", [])[:top_k]
                        diseases = data.get("diseases", [])[:top_k]

                        st.markdown(
                            f'<div style="font-size:0.82rem;color:#64748b;margin:0.3rem 0 0.7rem 0">'
                            f'<strong>📋 Kết quả cho:</strong> {data.get("accession")} '
                            f'(Giới hạn: {top_k})'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        if show_sequence:
                            st.text_area("Protein sequence", data.get("sequence", ""), height=240)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader(f"Thuốc liên quan ({len(drugs)})")
                            if not drugs:
                                st.info("Không tìm thấy thuốc liên quan.")
                            else:
                                st.dataframe(
                                    [{"ID": item["id"], "Tên thuốc": item["name"]} for item in drugs],
                                    use_container_width=True,
                                )

                        with col2:
                            st.subheader(f"Bệnh liên quan ({len(diseases)})")
                            if not diseases:
                                st.info("Không tìm thấy bệnh liên quan.")
                            else:
                                st.dataframe(
                                    [{"ID": item["id"], "Tên bệnh": item["name"]} for item in diseases],
                                    use_container_width=True,
                                )
        card_close()

    # ── Tab D: History ─────────────────────────────────────────────────
    with tab_d:
        card_open("Lịch sử tra cứu của bạn", "📋")
        col_refresh, col_info = st.columns([1, 4])
        with col_refresh:
            if st.button("🔄 Tải lại", use_container_width=True):
                st.session_state.pop("history_rows", None)
        with col_info:
            st.markdown(
                '<div style="font-size:0.83rem;color:#64748b;padding-top:0.4rem">'
                'Hiển thị tối đa 200 bản ghi gần nhất.</div>',
                unsafe_allow_html=True,
            )
        if st.session_state.get("history_rows") is None:
            try:
                st.session_state["history_rows"] = client.history()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Không tải được lịch sử: {exc}")
                st.session_state["history_rows"] = []
        show_history_table(st.session_state.get("history_rows", []))
        card_close()

