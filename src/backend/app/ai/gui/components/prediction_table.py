from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


def render_prediction_results_from_csv(csv_path: Path, threshold: float) -> None:
    # Doc ket qua du doan tu CSV backend.
    if not csv_path.exists():
        st.error(f"Khong tim thay file ket qua: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        st.info("File ket qua rong.")
        return

    # Chuan hoa cot Probability de loc theo nguong.
    if "Probability" not in df.columns:
        if "probability" in df.columns:
            df = df.rename(columns={"probability": "Probability"})
        elif "score" in df.columns:
            df = df.rename(columns={"score": "Probability"})
        else:
            st.error("CSV ket qua khong co cot Probability.")
            return

    df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
    df = df.dropna(subset=["Probability"])
    df = df[df["Probability"] >= float(threshold)]
    if df.empty:
        st.info("Khong co ket qua nao dat nguong hien tai.")
        return

    # Uu tien hien thi danh sach benh du doan kem Probability.
    if {"disease_id", "disease_name"}.issubset(df.columns):
        bang_hien_thi = df[["disease_id", "disease_name", "Probability"]].copy()
        bang_hien_thi = bang_hien_thi.sort_values(by="Probability", ascending=False).reset_index(drop=True)
        st.dataframe(bang_hien_thi, use_container_width=True)
        return

    # Neu CSV la ket qua theo huong thuoc thi hien thi cap cot tuong ung kem Probability.
    if {"drug_id", "drug_name"}.issubset(df.columns):
        bang_hien_thi = df[["drug_id", "drug_name", "Probability"]].copy()
        bang_hien_thi = bang_hien_thi.sort_values(by="Probability", ascending=False).reset_index(drop=True)
        st.dataframe(bang_hien_thi, use_container_width=True)
        return

    st.error("CSV ket qua khong co cap cot hop le de hien thi.")
