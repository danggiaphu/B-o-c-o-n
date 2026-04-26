from __future__ import annotations

import pandas as pd
import streamlit as st

# Màu dark mode dùng chung cho các metric card inline
_METRIC_CARDS = [
    # (key, label, bg, border, fg)
    ("total",    "Tổng kết quả",          "#1a1d27", "#2d3148", "#e8eaf0"),
    ("known",    "✅ Liên kết đã biết",   "#0d2e1f", "#166534", "#4ade80"),
    ("pred",     "🔬 AI dự đoán mới",     "#172040", "#1e40af", "#93c5fd"),
]


def card_open(title: str, icon: str = "") -> None:
    icon_html = f"<span style='margin-right:0.35rem'>{icon}</span>" if icon else ""
    st.markdown(
        f'<div class="card"><div class="card-title">{icon_html}{title}</div>',
        unsafe_allow_html=True,
    )


def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _metric_row(total: int, known: int, pred: int) -> str:
    items = [
        (total, "Tổng kết quả",        "#1a1d27", "#2d3148", "#e8eaf0"),
        (known, "✅ Liên kết đã biết", "#0d2e1f", "#166534", "#4ade80"),
        (pred,  "🔬 AI dự đoán mới",   "#172040", "#1e40af", "#93c5fd"),
    ]
    cols = "".join(
        f"""<div style="flex:1;background:{bg};border:1px solid {bd};border-radius:10px;
                  padding:0.75rem 1rem;min-width:0">
              <div style="font-size:0.7rem;font-weight:700;color:{fg};opacity:0.7;
                          text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.2rem;
                          white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{label}</div>
              <div style="font-size:1.7rem;font-weight:800;color:{fg};line-height:1">{val}</div>
            </div>"""
        for val, label, bg, bd, fg in items
    )
    return f'<div style="display:flex;gap:0.75rem;margin-bottom:1rem">{cols}</div>'


def _render_group_table(group: list[dict], entity_label: str) -> None:
    if not group:
        st.markdown('<div class="info-box">Không tìm thấy kết quả thuộc nhóm này.</div>', unsafe_allow_html=True)
        return

    rows_html = ""
    for i, r in enumerate(group, 1):
        score_pct = float(r.get("score", 0.0)) * 100
        bar_w = max(2, int(score_pct))
        badge = (
            '<span class="badge-known">✅ Đã biết</span>'
            if r.get("known")
            else '<span class="badge-pred">🔬 Dự đoán</span>'
        )
        rows_html += f"""
        <tr>
          <td style="color:#7c829e;font-size:0.8rem">{i}</td>
          <td><strong>{r.get("name", "")}</strong></td>
          <td>{badge}</td>
          <td>
            <div class="score-bar-wrap">
              <div class="score-bar-bg"><div class="score-bar-fill" style="width:{bar_w}%"></div></div>
              <span class="score-val">{score_pct:.2f}%</span>
            </div>
          </td>
        </tr>"""

    st.markdown(
        f"""<table class="result-table">
          <thead><tr><th>#</th><th>{entity_label}</th><th>Trạng thái</th><th>Điểm dự đoán</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>""",
        unsafe_allow_html=True,
    )


def show_result_table(results: list[dict], entity_label: str = "Tên") -> None:
    if not results:
        st.markdown('<div class="info-box">Không có kết quả phù hợp với điều kiện hiện tại.</div>', unsafe_allow_html=True)
        return

    known_count = sum(1 for r in results if r.get("known", False))
    pred_count = len(results) - known_count
    st.markdown(_metric_row(len(results), known_count, pred_count), unsafe_allow_html=True)
    _render_group_table(results, entity_label)

    if known_count > 0:
        st.markdown(
            f'<div class="info-box" style="margin-top:0.8rem">'
            f'<strong>✅ {known_count} kết quả</strong> có trong dữ liệu huấn luyện. '
            f'<strong>🔬 {pred_count} kết quả</strong> là dự đoán mới từ AI.</div>',
            unsafe_allow_html=True,
        )


def show_split_result_table(results: list[dict], entity_label: str = "Tên") -> None:
    if not results:
        st.markdown('<div class="info-box">Không có kết quả phù hợp với điều kiện hiện tại.</div>', unsafe_allow_html=True)
        return

    df_tri_benh = [r for r in results if r.get("known", False)]
    df_canh_bao = [r for r in results if not r.get("known", False)]
    known_count, pred_count = len(df_tri_benh), len(df_canh_bao)

    # Metric row — dùng màu split riêng
    items = [
        (len(results), "Tổng kết quả",              "#1a1d27", "#2d3148", "#e8eaf0"),
        (known_count,  "✅ Nhóm Điều Trị",          "#0d2e1f", "#166534", "#4ade80"),
        (pred_count,   "⚠️ Nguy Cơ / Cần Kiểm Chứng","#2d1a06","#92400e", "#fbbf24"),
    ]
    cols = "".join(
        f"""<div style="flex:1;background:{bg};border:1px solid {bd};border-radius:10px;
                  padding:0.75rem 1rem;min-width:0">
              <div style="font-size:0.7rem;font-weight:700;color:{fg};opacity:0.7;
                          text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.2rem">{label}</div>
              <div style="font-size:1.7rem;font-weight:800;color:{fg};line-height:1">{val}</div>
            </div>"""
        for val, label, bg, bd, fg in items
    )
    st.markdown(f'<div style="display:flex;gap:0.75rem;margin-bottom:1rem">{cols}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h3 style="color:#4ade80;margin:0 0 0.2rem 0;font-size:1rem">✅ Nhóm Thuốc Điều Trị</h3>', unsafe_allow_html=True)
        st.caption(f"{known_count} liên kết đã được xác nhận trong dữ liệu huấn luyện")
        _render_group_table(df_tri_benh, entity_label)

    with col2:
        st.markdown('<h3 style="color:#fbbf24;margin:0 0 0.2rem 0;font-size:1rem">⚠️ Nhóm Nguy Cơ / Tác Dụng Phụ</h3>', unsafe_allow_html=True)
        st.caption(f"{pred_count} kết quả do AI dự đoán — cần kiểm chứng lâm sàng")
        _render_group_table(df_canh_bao, entity_label)

    st.markdown(
        '<div class="info-box" style="margin-top:0.8rem">'
        '<strong>Hướng dẫn đọc kết quả:</strong> '
        '✅ <strong>Nhóm Điều Trị</strong> — liên kết có trong dataset, đã xác nhận qua nghiên cứu. '
        '⚠️ <strong>Nhóm Nguy Cơ</strong> — AI suy luận từ cấu trúc đồ thị, '
        '<em>không phải khuyến cáo y tế</em>, chỉ dùng để nghiên cứu.</div>',
        unsafe_allow_html=True,
    )


def show_history_table(rows: list[dict]) -> None:
    if not rows:
        st.markdown('<div class="info-box">Chưa có lịch sử tra cứu.</div>', unsafe_allow_html=True)
        return
    df = pd.DataFrame(rows)
    if "score" in df.columns:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        df["score (%)"] = (df["score"] * 100).map(lambda x: f"{x:.2f}%")
    if "direction" in df.columns:
        df["direction"] = df["direction"].map(
            {"drug_to_disease": "💊→🦠 Thuốc→Bệnh", "disease_to_drug": "🦠→💊 Bệnh→Thuốc"}
        ).fillna(df["direction"])
    if "known" in df.columns:
        df["phân loại"] = df["known"].map(lambda v: "✅ Điều trị" if v else "⚠️ Dự đoán/Tác dụng phụ")
    display_cols = [c for c in ["direction", "input_name", "target_name", "phân loại", "score (%)", "timestamp"] if c in df.columns]
    st.dataframe(df[display_cols] if display_cols else df, use_container_width=True, hide_index=True)


def show_metric_row(stats: dict) -> None:
    items = [
        ("total_users",       "👤 Users",        "#0d2e1f", "#166534", "#4ade80"),
        ("total_drugs",       "💊 Thuốc",         "#172040", "#1e40af", "#93c5fd"),
        ("total_diseases",    "🦠 Bệnh",          "#2d1e06", "#92500e", "#fbbf24"),
        ("total_links",       "🔗 Liên kết",      "#1e1040", "#5b21b6", "#c084fc"),
        ("total_predictions", "🔬 Predictions",   "#2d1206", "#991b1b", "#fca5a5"),
    ]
    cols_html = "".join(
        f"""<div style="flex:1;background:{bg};border:1px solid {bd};border-radius:10px;
                  padding:0.75rem 1rem;min-width:0">
              <div style="font-size:0.7rem;font-weight:700;color:{fg};opacity:0.7;
                          text-transform:uppercase;letter-spacing:0.05em;margin-bottom:0.2rem;
                          white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{label}</div>
              <div style="font-size:1.7rem;font-weight:800;color:{fg};line-height:1">{int(stats.get(key, 0))}</div>
            </div>"""
        for key, label, bg, bd, fg in items
    )
    st.markdown(f'<div style="display:flex;gap:0.75rem;margin-bottom:0.5rem">{cols_html}</div>', unsafe_allow_html=True)
