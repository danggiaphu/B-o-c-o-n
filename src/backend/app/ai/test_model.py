from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

try:
    from .mo_hinh_ai import FuzzyGCN
except ImportError:
    from mo_hinh_ai import FuzzyGCN


def infer_feature_dims(dataset_dir: Path) -> tuple[int, int]:
    drug_dim = pd.read_csv(dataset_dir / "DrugFingerprint.csv", nrows=1, index_col=0).shape[1]
    # DiseaseFeature.csv khong co header; cot dau tien la ten benh.
    disease_df = pd.read_csv(dataset_dir / "DiseaseFeature.csv", nrows=1, header=None)
    disease_dim = int(disease_df.shape[1]) - 1
    if disease_dim <= 0:
        raise ValueError("So chieu dac trung benh khong hop le. Kiem tra file DiseaseFeature.csv")
    return int(drug_dim), int(disease_dim)


def infer_feature_dims_from_weights(weights_path: Path) -> tuple[int, int]:
    state = torch.load(weights_path, map_location="cpu")
    drug_in = int(state["ma_hoa_thuoc.weight"].shape[1])
    disease_in = int(state["ma_hoa_benh.weight"].shape[1])
    return drug_in, disease_in


def infer_meta(weights_path: Path) -> dict:
    meta_path = weights_path.with_suffix(".json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiem thu nhanh checkpoint FuzzyGCN.")
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(Path(__file__).resolve().parents[3]),
        help="Duong dan thu muc goc.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="B-dataset",
        help="Thu muc dataset de suy ra kich thuoc dau vao.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "weights" / "fuzzy_gcn_all.pth"),
        help="Duong dan toi file checkpoint .pth.",
    )
    parser.add_argument(
        "--from-weights",
        action="store_true",
        help="Infer input dimensions from checkpoint instead of dataset files.",
    )
    parser.add_argument("--drug-id", type=int, default=0, help="ID thuoc de du doan top-k.")
    parser.add_argument("--top-k", type=int, default=5, help="So luong ket qua top-k.")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Kich thuoc lop an.")
    parser.add_argument("--out-dim", type=int, default=32, help="Kich thuoc dau ra.")
    parser.add_argument("--gcn-layers", type=int, default=2, help="So lop GCN.")
    parser.add_argument(
        "--out-csv",
        type=str,
        default="",
        help="Neu cung cap, se luu bang ket qua ra CSV voi cot Probability.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    dataset_dir = project_root / args.dataset
    weights_path = Path(args.weights)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Khong tim thay dataset: {dataset_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Khong tim thay checkpoint: {weights_path}")

    meta = infer_meta(weights_path)
    if args.from_weights:
        drug_dim, disease_dim = infer_feature_dims_from_weights(weights_path)
        args.hidden_dim = int(meta.get("hidden_channels", args.hidden_dim))
        args.out_dim = int(meta.get("out_channels", args.out_dim))
        args.gcn_layers = int(meta.get("gcn_layers", args.gcn_layers))
    else:
        drug_dim, disease_dim = infer_feature_dims(dataset_dir)

    model = FuzzyGCN(
        so_chieu_thuoc=drug_dim,
        so_chieu_benh=disease_dim,
        so_chieu_an=int(args.hidden_dim),
        so_chieu_ra=int(args.out_dim),
        so_lop_gcn=int(args.gcn_layers),
        duong_dan_trong_so=str(weights_path),
    )

    loaded = model.tai_trong_so()
    if not loaded:
        raise RuntimeError(f"Khong the load trong so: {weights_path}")

    model.eval()
    ket_qua = model.du_doan_top_k(drug_id=args.drug_id, k=args.top_k)

    # Tao DataFrame backend inference co cot Probability de frontend loc threshold.
    df_kq = pd.DataFrame(ket_qua)
    if "Probability" not in df_kq.columns:
        if "probability" in df_kq.columns:
            df_kq = df_kq.rename(columns={"probability": "Probability"})
        elif "score" in df_kq.columns:
            df_kq = df_kq.rename(columns={"score": "Probability"})

    print(f"Checkpoint: {weights_path}")
    print(f"Kich thuoc: drug={drug_dim}, disease={disease_dim}")
    print(f"Top-{args.top_k} du doan cho drug_id={args.drug_id}:")
    if not df_kq.empty:
        for idx, row in df_kq.iterrows():
            print(
                f"{idx + 1}. disease_id={row.get('disease_id')}, "
                f"disease_name={row.get('disease_name')}, Probability={float(row.get('Probability', 0.0)):.6f}"
            )

    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        # Luu CSV voi cot Probability (float) theo yeu cau.
        df_kq.to_csv(out_csv, index=False, encoding="utf-8")
        print(f"Da luu CSV ket qua tai: {out_csv}")


if __name__ == "__main__":
    main()
