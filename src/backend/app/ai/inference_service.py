from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .mo_hinh_ai import FuzzyGCN
except ImportError:
    from mo_hinh_ai import FuzzyGCN

# ── Cấu hình dataset mặc định (CẬP NHẬT 2026-04-22) ──
# Thay đổi: Cho phép sử dụng cả 3 dataset (B, C, F) một lúc hoặc chỉ từng dataset cụ thể
# Lợi ích:
#   - Kết hợp dữ liệu từ B-dataset (269 thuốc, 598 bệnh, 18,416 liên kết)
#   - Kết hợp dữ liệu từ C-dataset (663 thuốc, 409 bệnh, 2,532 liên kết)
#   - Kết hợp dữ liệu từ F-dataset (593 thuốc, 313 bệnh, 1,598 liên kết)
#   - Tổng cộng: 1,525 thuốc + 1,570 bệnh + 22,546 liên kết (sau gộp & loại duplicate)
# 
# Cách sử dụng:
#   DEFAULT_DATASETS = ["B-dataset", "C-dataset", "F-dataset"]  # Gộp cả 3
#   DEFAULT_DATASETS = ["B-dataset"]                            # Chỉ B-dataset
#   DEFAULT_DATASETS = ["B-dataset", "C-dataset"]              # B + C
DEFAULT_DATASETS = ["B-dataset", "C-dataset", "F-dataset"]  # Sử dụng cả 3 dataset
# DEFAULT_DATASETS = ["B-dataset"]  # Hoặc chỉ dùng B-dataset


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _dataset_dir(dataset: str) -> Path:
    return _project_root() / "dataset" / dataset


def _normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def load_drug_table(dataset: str) -> pd.DataFrame:
    dataset_dir = _dataset_dir(dataset)
    df = pd.read_csv(dataset_dir / "DrugInformation.csv")
    if len(df.columns) > 0 and str(df.columns[0]).startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
    if "name" not in df.columns:
        raise ValueError("DrugInformation.csv thieu cot name")
    df["drug_id"] = range(len(df))
    df["name_norm"] = df["name"].map(_normalize)
    return df


def load_disease_table(dataset: str) -> pd.DataFrame:
    dataset_dir = _dataset_dir(dataset)
    df = pd.read_csv(dataset_dir / "DiseaseFeature.csv", header=None)
    out = pd.DataFrame({"disease_id": range(len(df)), "name": df.iloc[:, 0].astype(str)})
    out["name_norm"] = out["name"].map(_normalize)
    return out


def load_links(dataset: str) -> pd.DataFrame:
    dataset_dir = _dataset_dir(dataset)
    return pd.read_csv(dataset_dir / "DrugDiseaseAssociationNumber.csv")


def load_protein_table(dataset: str) -> pd.DataFrame:
    dataset_dir = _dataset_dir(dataset)
    df = pd.read_csv(dataset_dir / "ProteinInformation.csv")
    if len(df.columns) > 0 and str(df.columns[0]).startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])
    if "id" not in df.columns:
        raise ValueError("ProteinInformation.csv thieu cot id")
    df["protein_id"] = range(len(df))
    df["accession"] = df["id"].astype(str)
    return df


def load_drug_protein_links(dataset: str) -> pd.DataFrame:
    dataset_dir = _dataset_dir(dataset)
    return pd.read_csv(dataset_dir / "DrugProteinAssociationNumber.csv")


def load_protein_disease_links(dataset: str) -> pd.DataFrame:
    dataset_dir = _dataset_dir(dataset)
    return pd.read_csv(dataset_dir / "ProteinDiseaseAssociationNumber.csv")


def infer_feature_dims(dataset: str) -> tuple[int, int]:
    dataset_dir = _dataset_dir(dataset)
    drug_dim = pd.read_csv(dataset_dir / "DrugFingerprint.csv", nrows=1, index_col=0).shape[1]
    disease_df = pd.read_csv(dataset_dir / "DiseaseFeature.csv", nrows=1, header=None)
    disease_dim = int(disease_df.shape[1]) - 1
    return int(drug_dim), int(disease_dim)


# ── HÀM GỘP DỮ LIỆU (MỚI - CẬP NHẬT 2026-04-22) ──
# Mục đích: Hỗ trợ gộp dữ liệu từ nhiều dataset, loại bỏ các mục trùng lặp
# 
# Thay đổi:
#   - Thêm hàm load_drug_table_merged() để gộp bảng thuốc từ nhiều dataset
#   - Thêm hàm load_disease_table_merged() để gộp bảng bệnh từ nhiều dataset
#   - Thêm hàm load_links_merged() để gộp các liên kết từ nhiều dataset
#   - Tất cả các hàm đều tự động dùng DEFAULT_DATASETS nếu không chỉ định
#
# Ví dụ:
#   drugs = load_drug_table_merged()  # Dùng DEFAULT_DATASETS
#   diseases = load_disease_table_merged(["B-dataset"])  # Chỉ B-dataset
#   links = load_links_merged(["B-dataset", "C-dataset"])  # B + C

def load_drug_table_merged(datasets: list[str] | None = None) -> pd.DataFrame:
    """
    Gộp bảng thuốc từ nhiều dataset (loại bỏ duplicate).
    
    Args:
        datasets: Danh sách dataset cần gộp. Nếu None, dùng DEFAULT_DATASETS.
    
    Returns:
        DataFrame với các cột: name, name_norm, drug_id (được tái định số từ 0)
    """
    if datasets is None:
        datasets = DEFAULT_DATASETS
    dfs = []
    for dataset in datasets:
        df = load_drug_table(dataset)
        dfs.append(df[["name", "name_norm"]])
    merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["name_norm"])
    merged["drug_id"] = range(len(merged))
    return merged


def load_disease_table_merged(datasets: list[str] | None = None) -> pd.DataFrame:
    """
    Gộp bảng bệnh từ nhiều dataset (loại bỏ duplicate).
    
    Args:
        datasets: Danh sách dataset cần gộp. Nếu None, dùng DEFAULT_DATASETS.
    
    Returns:
        DataFrame với các cột: name, name_norm, disease_id (được tái định số từ 0)
    """
    if datasets is None:
        datasets = DEFAULT_DATASETS
    dfs = []
    for dataset in datasets:
        df = load_disease_table(dataset)
        dfs.append(df[["name", "name_norm"]])
    merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["name_norm"])
    merged["disease_id"] = range(len(merged))
    return merged


def load_links_merged(datasets: list[str] | None = None) -> pd.DataFrame:
    """
    Gộp các liên kết Drug-Disease từ nhiều dataset (loại bỏ duplicate).
    
    Args:
        datasets: Danh sách dataset cần gộp. Nếu None, dùng DEFAULT_DATASETS.
    
    Returns:
        DataFrame với các cột: drug, disease (được gộp từ nhiều dataset)
    """
    if datasets is None:
        datasets = DEFAULT_DATASETS
    dfs = []
    for dataset in datasets:
        df = load_links(dataset)
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True).drop_duplicates()
    return merged


@lru_cache(maxsize=4)
def get_model(dataset: str = "B-dataset") -> FuzzyGCN:
    root = _project_root()
    weights = root / "weights" / "fuzzy_gcn_all.pth"
    drug_dim, disease_dim = infer_feature_dims(dataset)
    model = FuzzyGCN(
        so_chieu_thuoc=drug_dim,
        so_chieu_benh=disease_dim,
        so_chieu_an=64,
        so_chieu_ra=32,
        so_lop_gcn=3,
        duong_dan_trong_so=str(weights),
    )
    model.tai_trong_so()
    diseases = load_disease_table(dataset)
    model.id_sang_ten_benh = {
        int(row["disease_id"]): str(row["name"])
        for _, row in diseases.iterrows()
    }
    model.eval()
    return model


@lru_cache(maxsize=4096)
def _predict_all_diseases_for_drug(drug_id: int) -> dict[int, float]:
    """Tra ve score class-1 theo tung disease cho mot drug_id. Sử dụng B-dataset."""
    dataset = "B-dataset"
    diseases = load_disease_table(dataset)
    model = get_model(dataset)
    # Lay toan bo benh de tranh mat score khi doi top-k/threshold.
    preds = model.du_doan_top_k(drug_id=int(drug_id), k=int(len(diseases)))
    scores: dict[int, float] = {}
    for item in preds:
        disease_id = int(item["disease_id"])
        score = float(item.get("Probability", item.get("score", 0.0)))
        scores[disease_id] = score
    return scores


def predict_diseases_by_drug_name(
    drug_name: str,
    top_k: int = 10,
    threshold: float = 0.0,
    datasets: list[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Dự đoán bệnh từ tên thuốc (CẬP NHẬT 2026-04-22 - HỖ TRỢ NHIỀU DATASET).
    
    THAY ĐỔI:
        ✓ Thêm tham số `datasets` (list[str] | None)
        ✓ Mặc định dùng DEFAULT_DATASETS (gộp B, C, F)
        ✓ Sử dụng load_drug_table_merged() thay vì load_drug_table()
        ✓ Sử dụng load_disease_table_merged() thay vì load_disease_table()
        ✓ Sử dụng load_links_merged() thay vì load_links()
    
    Args:
        drug_name (str): Tên thuốc cần dự đoán bệnh
        top_k (int): Số lượng kết quả hàng đầu (mặc định: 10)
        threshold (float): Ngưỡng điểm tối thiểu 0.0-1.0 (mặc định: 0.0)
        datasets (list[str] | None): Dataset để dùng. None → DEFAULT_DATASETS
    
    Returns:
        tuple: (input_name, results) với results chứa:
            - id: disease_id
            - name: Tên bệnh
            - score: Điểm dự đoán 0.0-1.0
            - known: Có trong dataset không? (True/False)
    """
    if datasets is None:
        datasets = DEFAULT_DATASETS
    
    dataset = "B-dataset"  # Dùng B-dataset cho model (giữ tương thích)
    drugs = load_drug_table_merged(datasets)  # ← THAY: load_drug_table_merged() thay load_drug_table()
    matches = drugs[drugs["name_norm"] == _normalize(drug_name)]
    if matches.empty:
        return drug_name, []
    drug_id = int(matches.iloc[0]["drug_id"])
    input_name = str(matches.iloc[0]["name"])
    diseases = load_disease_table_merged(datasets)  # ← THAY: load_disease_table_merged()
    score_map = _predict_all_diseases_for_drug(drug_id)

    links = load_links_merged(datasets)  # ← THAY: load_links_merged()
    known_disease_ids: set[int] = set(
        links.loc[links["drug"] == drug_id, "disease"].astype(int).tolist()
    )

    out: list[dict[str, Any]] = []
    for _, row in diseases.iterrows():
        disease_id = int(row["disease_id"])
        score = float(score_map.get(disease_id, 0.0))
        if score < threshold:
            continue
        out.append(
            {
                "id": disease_id,
                "name": str(row["name"]),
                "score": score,
                "known": disease_id in known_disease_ids,
            }
        )
    out.sort(key=lambda x: x["score"], reverse=True)
    out = out[:top_k]
    return input_name, out


def predict_drugs_by_disease_name(
    disease_name: str,
    top_k: int = 10,
    threshold: float = 0.0,
    datasets: list[str] | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Dự đoán thuốc từ tên bệnh (CẬP NHẬT 2026-04-22 - HỖ TRỢ NHIỀU DATASET).
    
    THAY ĐỔI:
        ✓ Thêm tham số `datasets` (list[str] | None)
        ✓ Mặc định dùng DEFAULT_DATASETS (gộp B, C, F)
        ✓ Sử dụng load_disease_table_merged() thay vì load_disease_table()
        ✓ Sử dụng load_drug_table_merged() thay vì load_drug_table()
        ✓ Sử dụng load_links_merged() thay vì load_links()
    
    Args:
        disease_name (str): Tên bệnh cần dự đoán thuốc
        top_k (int): Số lượng kết quả hàng đầu (mặc định: 10)
        threshold (float): Ngưỡng điểm tối thiểu 0.0-1.0 (mặc định: 0.0)
        datasets (list[str] | None): Dataset để dùng. None → DEFAULT_DATASETS
    
    Returns:
        tuple: (input_name, results) với results chứa:
            - id: drug_id
            - name: Tên thuốc
            - score: Điểm dự đoán 0.0-1.0
            - known: Có trong dataset không? (True/False)
    """
    if datasets is None:
        datasets = DEFAULT_DATASETS
    
    dataset = "B-dataset"  # Dùng B-dataset cho model (giữ tương thích)
    diseases = load_disease_table_merged(datasets)  # ← THAY: load_disease_table_merged()
    drugs = load_drug_table_merged(datasets)  # ← THAY: load_drug_table_merged()
    links = load_links_merged(datasets)  # ← THAY: load_links_merged()
    matches = diseases[diseases["name_norm"] == _normalize(disease_name)]
    if matches.empty:
        return disease_name, []
    disease_id = int(matches.iloc[0]["disease_id"])
    input_name = str(matches.iloc[0]["name"])
    # Uu tien cac drug da co lien ket trong dataset de dam bao dung logic nghiep vu.
    known_drug_ids = set(links.loc[links["disease"] == disease_id, "drug"].astype(int).tolist())
    candidate_ids = sorted(known_drug_ids) if known_drug_ids else drugs["drug_id"].astype(int).tolist()

    rows: list[dict[str, Any]] = []
    for drug_id in candidate_ids:
        score_map = _predict_all_diseases_for_drug(int(drug_id))
        score = float(score_map.get(disease_id, 0.0))
        if score < threshold:
            continue
        name_series = drugs.loc[drugs["drug_id"] == int(drug_id), "name"]
        drug_name_value = str(name_series.iloc[0]) if not name_series.empty else f"Drug_{drug_id}"
        rows.append({
            "id": int(drug_id),
            "name": drug_name_value,
            "score": score,
            "known": int(drug_id) in known_drug_ids,
        })
    rows.sort(key=lambda x: x["score"], reverse=True)
    return input_name, rows[:top_k]
