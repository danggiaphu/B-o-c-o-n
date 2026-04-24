from __future__ import annotations

import html
import io
import re
from urllib.parse import quote

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


def _render_molecule_preview(smiles: str | None, name: str | None) -> bytes | None:
    if not smiles or not name:
        st.info("Không có SMILES để hiển thị cấu trúc phân tử.")
        return None

    try:
        from draw import draw_molecule_image  # type: ignore
    except Exception as exc:  # noqa: BLE001
        st.error(f"Không thể vẽ cấu trúc phân tử. RDKit chưa được cài đặt hoặc không khả dụng: {exc}")
        return None

    try:
        image = draw_molecule_image(smiles, size=(320, 320))
        st.image(image, caption=f"Cấu trúc phân tử của {name}", width="content")

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Không thể tạo ảnh phân tử: {exc}")
        return None


def _render_molecule_download(image_bytes: bytes | None, name: str | None) -> None:
    if not image_bytes or not name:
        return

    st.download_button(
        "Tải ảnh PNG",
        data=image_bytes,
        file_name=f"{_clean_filename(name)}.png",
        mime="image/png",
    )


def _html_to_data_url(html_content: str) -> str:
    return f"data:text/html;charset=utf-8,{quote(html_content)}"


def _build_node_network_html(
    selected_diseases: list[dict],
    selected_proteins: list[dict],
    selected_drugs: list[dict],
    direct_links: set[tuple[int, int]],
    protein_drug_links: set[tuple[int, int]],
    protein_disease_links: set[tuple[int, int]],
) -> str:
    columns = [
        ("Bệnh", selected_diseases, "#111111"),
        ("Protein", selected_proteins, "#fbbf24"),
        ("Thuốc", selected_drugs, "#a3e635"),
    ]
    column_x = [140, 520, 900]
    start_y = 120
    gap_y = 155
    node_radius = 40
    svg_height = max(340, 180 + gap_y * max(len(selected_diseases), len(selected_proteins), len(selected_drugs), 1))
    svg_width = 1040

    positions: dict[tuple[str, int], tuple[int, int]] = {}
    circles_html: list[str] = []

    for idx, (label, items, fill) in enumerate(columns):
        x = column_x[idx]
        circles_html.append(
            f'<text x="{x}" y="52" text-anchor="middle" font-size="22" font-weight="800" fill="#0f172a">{label}</text>'
        )
        if not items:
            circles_html.append(f'<text x="{x}" y="96" text-anchor="middle" font-size="14" fill="#64748b">Chưa chọn</text>')
            continue

        for item_index, item in enumerate(items):
            y = start_y + item_index * gap_y
            positions[(label, int(item["id"]))] = (x, y)
            name = html.escape(str(item["name"]))
            circles_html.append(
                f"""
                <circle cx="{x}" cy="{y}" r="{node_radius}" fill="{fill}" stroke="#0f172a" stroke-width="4"></circle>
                <text x="{x}" y="{y + node_radius + 28}" text-anchor="middle" font-size="15" font-weight="700" fill="#111827">{name}</text>
                """
            )

    edges_html: list[str] = []
    legend_items = [("#16a34a", "Thuốc chữa bệnh"), ("#f59e0b", "Protein liên quan")]
    for i, (color, label) in enumerate(legend_items):
        legend_x = 60 + i * 240
        edges_html.append(
            f"""
            <line x1="{legend_x}" y1="20" x2="{legend_x + 48}" y2="20" stroke="{color}" stroke-width="6" stroke-linecap="round"></line>
            <text x="{legend_x + 58}" y="25" font-size="14" font-weight="700" fill="#334155">{label}</text>
            """
        )

    for disease in selected_diseases:
        disease_key = ("Bệnh", int(disease["id"]))
        if disease_key not in positions:
            continue
        x1, y1 = positions[disease_key]
        for drug in selected_drugs:
            drug_key = ("Thuốc", int(drug["id"]))
            if drug_key not in positions:
                continue
            if (int(drug["id"]), int(disease["id"])) in direct_links:
                x2, y2 = positions[drug_key]
                edges_html.append(
                    f'<line x1="{x1 + node_radius}" y1="{y1}" x2="{x2 - node_radius}" y2="{y2}" stroke="#16a34a" stroke-width="5" stroke-linecap="round" opacity="0.95"></line>'
                )

    for protein in selected_proteins:
        protein_key = ("Protein", int(protein["id"]))
        if protein_key not in positions:
            continue
        px, py = positions[protein_key]

        for disease in selected_diseases:
            disease_key = ("Bệnh", int(disease["id"]))
            if disease_key not in positions:
                continue
            if (int(protein["id"]), int(disease["id"])) in protein_disease_links:
                dx, dy = positions[disease_key]
                edges_html.append(
                    f'<line x1="{dx + node_radius}" y1="{dy}" x2="{px - node_radius}" y2="{py}" stroke="#f59e0b" stroke-width="5" stroke-linecap="round" opacity="0.95"></line>'
                )

        for drug in selected_drugs:
            drug_key = ("Thuốc", int(drug["id"]))
            if drug_key not in positions:
                continue
            if (int(protein["id"]), int(drug["id"])) in protein_drug_links:
                dx, dy = positions[drug_key]
                edges_html.append(
                    f'<line x1="{px + node_radius}" y1="{py}" x2="{dx - node_radius}" y2="{dy}" stroke="#f59e0b" stroke-width="5" stroke-linecap="round" opacity="0.95"></line>'
                )

    return (
        f'<div style="background:#ffffff;border:1px solid #dbe4f0;border-radius:18px;'
        f'padding:1rem 1rem 0.5rem 1rem;box-shadow:0 10px 30px rgba(15,23,42,0.08);overflow:auto">'
        f'<svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f"{''.join(edges_html)}"
        f"{''.join(circles_html)}"
        f"</svg>"
        f"</div>"
    )


def _render_batch_relation_view(client: ApiClient) -> None:
    st.markdown(
        '<div class="info-box">Chọn số lượng thuốc, protein và bệnh. Mỗi mục sẽ hiển thị thành một node. '
        'Đường xanh là liên kết Thuốc - Bệnh, đường vàng là liên kết Protein - Thuốc hoặc Protein - Bệnh.</div>',
        unsafe_allow_html=True,
    )

    drugs = _load_dropdown_values("drug_options", lambda: client.list_drugs(limit=1000))
    diseases = _load_dropdown_values("batch_disease_options", lambda: client.list_diseases(limit=1000))
    proteins = _load_dropdown_values("protein_options", lambda: client.list_proteins(limit=1000))

    drug_options = {item["name"]: item for item in drugs}
    disease_options = {item["name"]: item for item in diseases}
    protein_options = {item["name"]: item for item in proteins}

    count_cols = st.columns(3)
    with count_cols[0]:
        drug_count = int(st.number_input("Số lượng thuốc", min_value=0, max_value=8, value=3, step=1, key="batch_drug_count"))
    with count_cols[1]:
        protein_count = int(st.number_input("Số lượng protein", min_value=0, max_value=8, value=2, step=1, key="batch_protein_count"))
    with count_cols[2]:
        disease_count = int(st.number_input("Số lượng bệnh", min_value=0, max_value=8, value=3, step=1, key="batch_disease_count"))

    select_cols = st.columns(3)
    with select_cols[0]:
        selected_drug_names = st.multiselect(
            "Thuốc",
            options=list(drug_options.keys()),
            default=list(drug_options.keys())[:drug_count],
            max_selections=drug_count or None,
            key="batch_drug_names",
        )
    with select_cols[1]:
        selected_protein_names = st.multiselect(
            "Protein",
            options=list(protein_options.keys()),
            default=list(protein_options.keys())[:protein_count],
            max_selections=protein_count or None,
            key="batch_protein_names",
        )
    with select_cols[2]:
        selected_disease_names = st.multiselect(
            "Bệnh",
            options=list(disease_options.keys()),
            default=list(disease_options.keys())[:disease_count],
            max_selections=disease_count or None,
            key="batch_disease_names",
        )

    selected_drugs = [drug_options[name] for name in selected_drug_names if name in drug_options]
    selected_proteins = [protein_options[name] for name in selected_protein_names if name in protein_options]
    selected_diseases = [disease_options[name] for name in selected_disease_names if name in disease_options]

    if not selected_drugs and not selected_proteins and not selected_diseases:
        st.info("Hãy chọn ít nhất một thuốc, protein hoặc bệnh để hiển thị sơ đồ.")
        return

    if st.button("🕸️ Hiển thị sơ đồ liên kết", width="stretch", key="batch_render_graph"):
        st.session_state["batch_graph_ready"] = True

    if not st.session_state.get("batch_graph_ready"):
        return

    with st.spinner("Đang dựng sơ đồ liên kết..."):
        direct_links = {(int(row["drug_id"]), int(row["disease_id"])) for row in client.list_links(limit=2000)}
        protein_drug_links: set[tuple[int, int]] = set()
        protein_disease_links: set[tuple[int, int]] = set()

        for protein in selected_proteins:
            protein_id = int(protein["id"])
            data = client.get_protein_links(protein_id=protein_id)
            for drug in data.get("drugs", []):
                protein_drug_links.add((protein_id, int(drug["id"])))
            for disease in data.get("diseases", []):
                protein_disease_links.add((protein_id, int(disease["id"])))

    graph_html = _build_node_network_html(
        selected_diseases=selected_diseases,
        selected_proteins=selected_proteins,
        selected_drugs=selected_drugs,
        direct_links=direct_links,
        protein_drug_links=protein_drug_links,
        protein_disease_links=protein_disease_links,
    )
    graph_height = max(
        420,
        260 + 155 * max(len(selected_diseases), len(selected_proteins), len(selected_drugs), 1),
    )
    st.iframe(_html_to_data_url(graph_html), height=graph_height)


def _build_graph_panel_html(
    selected_diseases: list[dict],
    selected_proteins: list[dict],
    selected_drugs: list[dict],
    direct_links: set[tuple[int, int]],
    protein_drug_links: set[tuple[int, int]],
    protein_disease_links: set[tuple[int, int]],
) -> tuple[str, int]:
    # Dynamic layout: support up to 8 nodes per column
    node_columns = [
        ("Benh",    "Bệnh",    selected_diseases, "#2b2b2b", "#5d5d5d", "#f3f4f6"),
        ("Protein", "Protein", selected_proteins, "#f5c242", "#d8a21d", "#4a3410"),
        ("Thuoc",   "Thuốc",   selected_drugs,    "#9fdb4d", "#6fad29", "#274313"),
    ]
    column_x = [160, 520, 880]
    radius = 38
    gap_y = 130
    top_offset = 100  # space for legend + column title

    max_items = max(len(selected_diseases), len(selected_proteins), len(selected_drugs), 1)
    svg_width = 1040
    svg_height = top_offset + max_items * gap_y + 60

    positions: dict[tuple[str, int], tuple[int, int]] = {}
    edge_parts: list[str] = []
    node_parts: list[str] = []

    # Legend
    legend = [
        ("node", "#9fdb4d", "Thuốc"),
        ("node", "#2b2b2b", "Bệnh"),
        ("node", "#f5c242", "Protein"),
        ("line", "#43b44b", "Thuốc chữa Bệnh"),
        ("line", "#d4a63c", "Protein liên quan"),
    ]
    for idx, (kind, color, label) in enumerate(legend):
        x = 26 + idx * 190
        if kind == "node":
            edge_parts.append(f'<circle cx="{x}" cy="28" r="7" fill="{color}" stroke="#888" stroke-width="1.5"></circle>')
        else:
            edge_parts.append(
                f'<line x1="{x - 6}" y1="28" x2="{x + 20}" y2="28" stroke="{color}" stroke-width="4" stroke-linecap="round"></line>'
            )
        edge_parts.append(
            f'<text x="{x + 14}" y="33" font-size="13" font-weight="700" fill="#d7d7d2">{label}</text>'
        )

    # Column titles
    for col_idx, (key, label, items, fill, stroke, text_fill) in enumerate(node_columns):
        x = column_x[col_idx]
        node_parts.append(
            f'<text x="{x}" y="70" text-anchor="middle" font-size="17" font-weight="800" fill="#f3f4f6">{label}</text>'
        )

    # Nodes
    for col_idx, (key, label, items, fill, stroke, text_fill) in enumerate(node_columns):
        x = column_x[col_idx]
        for item_idx, item in enumerate(items):
            y = top_offset + item_idx * gap_y
            positions[(key, int(item["id"]))] = (x, y)
            name = html.escape(str(item["name"]))
            words = name.split()
            if len(words) > 1 and len(name) > 12:
                line1 = html.escape(" ".join(words[:2]))[:15]
                line2 = html.escape(" ".join(words[2:]))[:15] if len(words) > 2 else ""
                if line2:
                    display = (
                        f"<tspan x='{x}' dy='-9'>{line1}</tspan>"
                        f"<tspan x='{x}' dy='15'>{line2}</tspan>"
                    )
                else:
                    display = f"<tspan x='{x}' dy='-2'>{line1}</tspan>"
            else:
                short_name = name if len(name) <= 14 else f"{name[:11]}..."
                display = f"<tspan x='{x}' dy='5'>{short_name}</tspan>"

            node_parts.append(
                f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="4"></circle>'
            )
            node_parts.append(
                f'<text x="{x}" y="{y}" text-anchor="middle" font-size="12" font-weight="700" fill="{text_fill}">{display}</text>'
            )

    # Edges: Drug -> Disease (green)
    for disease in selected_diseases:
        disease_key = ("Benh", int(disease["id"]))
        if disease_key not in positions:
            continue
        dx, dy = positions[disease_key]
        for drug in selected_drugs:
            drug_key = ("Thuoc", int(drug["id"]))
            if drug_key in positions and (int(drug["id"]), int(disease["id"])) in direct_links:
                tx, ty = positions[drug_key]
                mid_x = (dx + tx) // 2
                edge_parts.append(
                    f'<path d="M{dx + radius},{dy} C{mid_x},{dy} {mid_x},{ty} {tx - radius},{ty}" '
                    f'fill="none" stroke="#43b44b" stroke-width="3" opacity="0.85"></path>'
                )

    # Edges: Protein -> Disease / Drug (amber)
    for protein in selected_proteins:
        protein_key = ("Protein", int(protein["id"]))
        if protein_key not in positions:
            continue
        px, py = positions[protein_key]
        for disease in selected_diseases:
            disease_key = ("Benh", int(disease["id"]))
            if disease_key in positions and (int(protein["id"]), int(disease["id"])) in protein_disease_links:
                dx2, dy2 = positions[disease_key]
                edge_parts.append(
                    f'<line x1="{px - radius}" y1="{py}" x2="{dx2 + radius}" y2="{dy2}" '
                    f'stroke="#d4a63c" stroke-width="2.5" opacity="0.82"></line>'
                )
        for drug in selected_drugs:
            drug_key = ("Thuoc", int(drug["id"]))
            if drug_key in positions and (int(protein["id"]), int(drug["id"])) in protein_drug_links:
                tx2, ty2 = positions[drug_key]
                edge_parts.append(
                    f'<line x1="{px + radius}" y1="{py}" x2="{tx2 - radius}" y2="{ty2}" '
                    f'stroke="#d4a63c" stroke-width="2.5" opacity="0.82"></line>'
                )

    html_block = (
        '<div style="background:#2f2f2c;border:1px solid #4a4a46;border-radius:16px;padding:0.85rem;'
        'box-shadow:0 8px 24px rgba(0,0,0,0.18);overflow-x:auto">'
        '<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="8" y="52" width="1024" rx="12" height="{inner_h}" fill="#31312f" stroke="#5b5b56" stroke-width="1.5"></rect>'
        '{edges}{nodes}'
        '</svg>'
        '</div>'
    ).format(
        height=svg_height,
        width=svg_width,
        inner_h=svg_height - 60,
        edges="".join(edge_parts),
        nodes="".join(node_parts),
    )
    return html_block, svg_height + 24


def _render_graph_relation_view(client: ApiClient) -> None:
    st.markdown(
        '<div style="background:#2f2f2c;border:1px solid #4a4a46;border-radius:16px;padding:1rem 1.1rem;'
        'margin-bottom:1rem;box-shadow:0 8px 24px rgba(0,0,0,0.14)">'
        '<div style="color:#f3f4f6;font-size:1.05rem;font-weight:800;margin-bottom:0.4rem">Cấu hình đồ thị</div>'
        '<div style="color:#bebeb8;font-size:0.9rem">Chọn số lượng node và tạo sơ đồ liên kết giữa Thuốc, Protein và Bệnh.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    drugs = _load_dropdown_values("drug_options", lambda: client.list_drugs(limit=1000))
    diseases = _load_dropdown_values("graph_disease_options", lambda: client.list_diseases(limit=1000))
    proteins = _load_dropdown_values("protein_options", lambda: client.list_proteins(limit=1000))

    drug_options = {item["name"]: item for item in drugs}
    disease_options = {item["name"]: item for item in diseases}
    protein_options = {item["name"]: item for item in proteins}

    c1, c2, c3 = st.columns(3)
    with c1:
        drug_count = int(st.slider("Số thuốc", 0, 8, 4, 1, key="graph_drug_count"))
    with c2:
        protein_count = int(st.slider("Số protein", 0, 8, 4, 1, key="graph_protein_count"))
    with c3:
        disease_count = int(st.slider("Số bệnh", 0, 8, 4, 1, key="graph_disease_count"))

    s1, s2, s3 = st.columns(3)
    with s1:
        selected_drug_names = st.multiselect(
            "Thuốc",
            options=list(drug_options.keys()),
            default=list(drug_options.keys())[:drug_count],
            max_selections=drug_count or None,
            key="graph_drug_names",
        )
    with s2:
        selected_protein_names = st.multiselect(
            "Protein",
            options=list(protein_options.keys()),
            default=list(protein_options.keys())[:protein_count],
            max_selections=protein_count or None,
            key="graph_protein_names",
        )
    with s3:
        selected_disease_names = st.multiselect(
            "Bệnh",
            options=list(disease_options.keys()),
            default=list(disease_options.keys())[:disease_count],
            max_selections=disease_count or None,
            key="graph_disease_names",
        )

    selected_drugs = [drug_options[name] for name in selected_drug_names if name in drug_options]
    selected_proteins = [protein_options[name] for name in selected_protein_names if name in protein_options]
    selected_diseases = [disease_options[name] for name in selected_disease_names if name in disease_options]

    if st.button("Tạo đồ thị ↗", width="stretch", key="graph_render_button"):
        st.session_state["graph_ready"] = True

    if not st.session_state.get("graph_ready"):
        return

    direct_links = {(int(row["drug_id"]), int(row["disease_id"])) for row in client.list_links(limit=2000)}
    protein_drug_links: set[tuple[int, int]] = set()
    protein_disease_links: set[tuple[int, int]] = set()
    for protein in selected_proteins:
        protein_id = int(protein["id"])
        data = client.get_protein_links(protein_id=protein_id)
        for drug in data.get("drugs", []):
            protein_drug_links.add((protein_id, int(drug["id"])))
        for disease in data.get("diseases", []):
            protein_disease_links.add((protein_id, int(disease["id"])))

    graph_html, graph_height = _build_graph_panel_html(
        selected_diseases=selected_diseases,
        selected_proteins=selected_proteins,
        selected_drugs=selected_drugs,
        direct_links=direct_links,
        protein_drug_links=protein_drug_links,
        protein_disease_links=protein_disease_links,
    )
    st.iframe(_html_to_data_url(graph_html), height=graph_height)


def render_user_workspace(api_base_url: str, token: str) -> None:
    client = ApiClient(api_base_url, token=token)

    tab_a, tab_b, tab_c, tab_d, tab_graph = st.tabs([
        "💊 Thuốc → Bệnh",
        "🦠 Bệnh → Thuốc",
        "🧬 Protein",
        "📋 Lịch sử",
        "🕸️ Sơ đồ",
    ])

    # ── Tab A: Drug → Disease ──────────────────────────────────────────
    with tab_a:
        card_open("Tìm bệnh tiềm năng từ tên thuốc", "💊")
        st.markdown(
            '<div class="info-box">Chọn số lượng kết quả trước, sau đó chọn tên thuốc để AI dự đoán các bệnh có thể liên quan. '
            'Kết quả <span class="badge-known" style="font-size:0.88rem">✅ Đã biết</span> '
            'là liên kết có trong dữ liệu huấn luyện; '
            '<span class="badge-pred" style="font-size:0.88rem">🔬 Dự đoán</span> '
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
            submitted = st.form_submit_button("🔍 Dự đoán", width='stretch')

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
                            f'<div style="font-size:0.92rem;color:#000000;margin:0.3rem 0 0.7rem 0">'
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
            submitted = st.form_submit_button("🔍 Dự đoán", width='stretch')

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
                            f'<div style="font-size:0.92rem;color:#000000;margin:0.3rem 0 0.7rem 0">'
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
            submitted = st.form_submit_button("🔍 Tra cứu", width='stretch')

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
                            f'<div style="font-size:0.92rem;color:#000000;margin:0.3rem 0 0.7rem 0">'
                            f'<strong>📋 Kết quả cho:</strong> {data.get("accession")} '
                            f'(Giới hạn: {top_k})'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        if show_sequence:
                            st.markdown('<h5 style="color:#000000;margin-bottom:0.7rem;font-size:1.0rem">Protein sequence</h5>', unsafe_allow_html=True)
                            st.code(data.get("sequence", ""), language="plain")

                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader(f"Thuốc liên quan ({len(drugs)})")
                            if not drugs:
                                st.info("Không tìm thấy thuốc liên quan.")
                            else:
                                st.dataframe(
                                    [{"ID": item["id"], "Tên thuốc": item["name"]} for item in drugs],
                                    width='stretch',
                                )

                        with col2:
                            st.subheader(f"Bệnh liên quan ({len(diseases)})")
                            if not diseases:
                                st.info("Không tìm thấy bệnh liên quan.")
                            else:
                                st.dataframe(
                                    [{"ID": item["id"], "Tên bệnh": item["name"]} for item in diseases],
                                    width='stretch',
                                )
        card_close()

    # ── Tab D: History ─────────────────────────────────────────────────
    with tab_d:
        card_open("Lịch sử tra cứu của bạn", "📋")
        col_refresh, col_info = st.columns([1, 4])
        with col_refresh:
            if st.button("🔄 Tải lại", width='stretch'):
                st.session_state.pop("history_rows", None)
        with col_info:
            st.markdown(
                '<div style="font-size:0.93rem;color:#000000;padding-top:0.4rem">'
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

    # ── Tab E: Graph View ──────────────
    with tab_graph:
        card_open("Sơ đồ liên kết nhiều node: Thuốc - Protein - Bệnh", "🕸️")
        _render_graph_relation_view(client)
        card_close()
