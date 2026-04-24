from __future__ import annotations

# Import cac thu vien can thiet de xu ly tham so, du lieu va huan luyen.
import argparse  # Thu vien doc tham so dong lenh.
import json  # Thu vien luu thong tin checkpoint va ket qua.
import random  # Thu vien tao so ngau nhien.
import time  # Thu vien do thoi gian moi epoch.
from dataclasses import dataclass  # Thu vien tao lop cau hinh don gian.
from pathlib import Path  # Thu vien xu ly duong dan file.
from typing import Dict, List, Set, Tuple  # Thu vien kieu du lieu.

import numpy as np  # Thu vien tinh toan ma tran.
import pandas as pd  # Thu vien doc file CSV.
import torch  # Thu vien PyTorch.
import torch.nn.functional as F  # Thu vien ham mat mat va kich hoat.
import torch.optim  # 🔧 FIX: Import torch.optim (cho ReduceLROnPlateau)
import torch.optim.lr_scheduler  # 🔧 FIX: Import scheduler chi tiet
from torch.amp.grad_scaler import GradScaler  # 🔧 FIX: Import GradScaler (dung trong dung: torch.amp.GradScaler(...))
from sklearn.model_selection import StratifiedKFold  # Thu vien chia K-Fold co can bang nhan.
from sklearn.preprocessing import StandardScaler  # Thu vien chuan hoa dac trung.
from torch_geometric.data import HeteroData  # Thu vien du lieu do thi.

try:
    from .model_factory import khoi_tao_mo_hinh_gnn
except ImportError:
    from model_factory import khoi_tao_mo_hinh_gnn


# =========================
# KHU VUC CAU HINH CHINH
# =========================
@dataclass
class CauHinh:
    # Thu muc goc cua du an.
    thu_muc_goc: str = str(Path(__file__).resolve().parents[4])
    # Ten dataset se dung (vi du: B-dataset).
    ten_dataset: str = "F-dataset"
    # Duong dan cac file chinh.
    tep_thuoc: str = "DrugFingerprint.csv"
    tep_benh: str = "DiseaseFeature.csv"
    tep_lien_ket: str = "DrugDiseaseAssociationNumber.csv"

    # Tham so huan luyen.
    so_epoch: int = 600
    toc_do_hoc: float = 3e-4
    weight_decay: float = 2e-4
    so_lop_gcn: int = 3
    kich_thuoc_an: int = 192
    kich_thuoc_ra: int = 96
    kiem_nhan_dung_som: int = 35

    # K-Fold.
    so_fold: int = 10

    # Ti le negative sampling (so am / so duong).
    ti_le_am: float = 1.2

    # Tham so dung som (early stopping).
    patience: int = 35  # So epoch cho phep khong cai thien.
    min_delta: float = 5e-4  # Muc cai thien toi thieu de tinh la tot hon.

    # Bat/tat chuan hoa dac trung.
    bat_chuan_hoa: bool = True

    # Cau hinh scheduler de giam toc do hoc khi AUC dung lai.
    factor_lr: float = 0.7
    patience_lr: int = 4
    min_lr: float = 5e-6
    clip_grad: float = 0.8
    pos_weight_duong: float = 1.1

    # Thiet bi.
    thiet_bi: str = "auto"  # auto|cuda|cpu
    bat_amp: bool = False

    # Thu muc luu trong so.
    thu_muc_trong_so: str = str(Path(__file__).resolve().parents[3] / "model_weights")

    # Seed.
    seed: int = 42


# =========================
# CAC HAM TAI VA CHUAN HOA DU LIEU
# =========================

def dat_seed(seed: int) -> None:
    # Dat seed cho random.
    random.seed(seed)
    # Dat seed cho numpy.
    np.random.seed(seed)
    # Dat seed cho torch.
    torch.manual_seed(seed)
    # Neu co CUDA thi dat them seed.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def doc_ma_tran(csv_path: Path) -> np.ndarray:
    # Doc file CSV va bo cot index neu co.
    df = pd.read_csv(csv_path, index_col=0)
    # Chuyen thanh ma tran float32.
    matrix = df.to_numpy(dtype=np.float32)
    # Kiem tra dung 2 chieu.
    if matrix.ndim != 2:
        raise ValueError(f"File dac trung khong phai 2D: {csv_path}")
    return matrix


def doc_lien_ket(csv_path: Path) -> np.ndarray:
    # Doc file lien ket duong.
    df = pd.read_csv(csv_path)
    # Kiem tra cot can thiet.
    if not {"drug", "disease"}.issubset(df.columns):
        raise ValueError(f"File lien ket thieu cot drug/disease: {csv_path}")
    # Tra ve ma tran 2 cot drug, disease.
    return df[["drug", "disease"]].to_numpy(dtype=np.int64)


def can_chinh_hang(matrix: np.ndarray, muc_tieu: int | None, nhan: str) -> np.ndarray:
    # Neu co muc tieu thi thu khop truc tiep.
    if muc_tieu is not None:
        if matrix.shape[0] == muc_tieu:
            return matrix
        if matrix.shape[1] == muc_tieu:
            return matrix.T
        print(
            f"Canh bao: {nhan} shape {matrix.shape} khong khop muc tieu {muc_tieu}. "
            "Se dung heuristic."
        )
    # Heuristic: neu so hang >= so cot thi xem hang la thuc the.
    if matrix.shape[0] >= matrix.shape[1]:
        return matrix
    return matrix.T


def chuan_hoa_dac_trung(thuoc: np.ndarray, benh: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Chuan hoa dac trung de giam lech thang do va giam overfitting.
    scaler_thuoc = StandardScaler()
    scaler_benh = StandardScaler()
    thuoc_chuan = scaler_thuoc.fit_transform(thuoc)
    benh_chuan = scaler_benh.fit_transform(benh)
    return thuoc_chuan.astype(np.float32), benh_chuan.astype(np.float32)


# =========================
# CAC HAM TAO DO THI
# =========================

def tao_do_thi(thuoc: np.ndarray, benh: np.ndarray, canh_duong: np.ndarray) -> HeteroData:
    # Tao doi tuong HeteroData.
    data = HeteroData()
    # Gan dac trung thuoc.
    data["drug"].x = torch.from_numpy(thuoc)
    # Gan dac trung benh.
    data["disease"].x = torch.from_numpy(benh)
    # Tao edge_index tu canh duong.
    edge_index = torch.from_numpy(canh_duong.T).long()
    data["drug", "interacts", "disease"].edge_index = edge_index
    data["disease", "rev_interacts", "drug"].edge_index = edge_index.flip(0)
    return data


# =========================
# CAC HAM NEGATIVE SAMPLING
# =========================

def tao_canh_am(
    so_thuoc: int,
    so_benh: int,
    tap_duong: Set[Tuple[int, int]],
    so_luong: int,
) -> np.ndarray:
    # Tao danh sach canh am de luu.
    canh_am: List[Tuple[int, int]] = []
    # Lap den khi du so luong canh am.
    while len(canh_am) < so_luong:
        # Lay nhieu mau mot luc de nhanh hon.
        so_mau = max(2048, (so_luong - len(canh_am)) * 3)
        mau_thuoc = np.random.randint(0, so_thuoc, size=so_mau)
        mau_benh = np.random.randint(0, so_benh, size=so_mau)
        for t, b in zip(mau_thuoc.tolist(), mau_benh.tolist()):
            if (t, b) not in tap_duong:
                canh_am.append((t, b))
                if len(canh_am) >= so_luong:
                    break
    return np.array(canh_am, dtype=np.int64)


# =========================
# CAC HAM TINH DIEM VA CHI SO
# =========================

def giai_ma_diem(emb: torch.Tensor, cap: torch.Tensor, offset_benh: int) -> torch.Tensor:
    # Lay chi so thuoc va benh.
    idx_thuoc = cap[0]
    idx_benh = cap[1] + offset_benh
    # Lay embedding.
    e_thuoc = emb[idx_thuoc]
    e_benh = emb[idx_benh]
    # Tinh diem bang tich vo huong.
    return (e_thuoc * e_benh).sum(dim=-1)


def tim_nguong_toi_uu_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # Tim nguong phan loai toi uu theo F1 tren tap train de can bang Precision/Recall.
    from sklearn.metrics import f1_score

    nguong_tot_nhat = 0.5
    f1_tot_nhat = -1.0
    for nguong in np.linspace(0.2, 0.8, 61):
        y_du_doan = (y_score >= nguong).astype(int)
        f1 = f1_score(y_true, y_du_doan, zero_division=0)
        if f1 > f1_tot_nhat:
            f1_tot_nhat = float(f1)
            nguong_tot_nhat = float(nguong)
    return nguong_tot_nhat


def tinh_chi_so(y_true: np.ndarray, y_score: np.ndarray, nguong: float = 0.5) -> Dict[str, float]:
    # Import sklearn trong ham.
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    # Tinh AUC.
    auc = roc_auc_score(y_true, y_score)
    # Tinh AUPR.
    aupr = average_precision_score(y_true, y_score)
    # Chuyen score thanh nhan.
    y_pred = (y_score >= nguong).astype(int)
    # Tinh cac chi so.
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    return {
        "AUC": float(auc),
        "AUPR": float(aupr),
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "MCC": float(mcc),
    }


# =========================
# HAM HUAN LUYEN 1 FOLD
# =========================

def huan_luyen_1_fold(
    cau_hinh: CauHinh,
    data_train: HeteroData,
    canh_train: np.ndarray,
    nhan_train: np.ndarray,
    canh_test: np.ndarray,
    nhan_test: np.ndarray,
    so_thuoc: int,
    so_benh: int,
    thu_muc_trong_so: Path,
    fold_id: int,
) -> Dict[str, float]:
    # Chon thiet bi.
    dung_cuda = torch.cuda.is_available() and cau_hinh.thiet_bi in {"auto", "cuda"}
    device = torch.device("cuda" if dung_cuda else "cpu")
    bat_amp = bool(cau_hinh.bat_amp and dung_cuda)

    # Neu bat buoc CUDA ma khong co thi bao loi.
    if cau_hinh.thiet_bi == "cuda" and not dung_cuda:
        raise RuntimeError("Ban chon CUDA nhung PyTorch khong nhan GPU.")

    # Dua data len device.
    data_train = data_train.to(device)

    # Tao mo hinh thong qua factory.
    mo_hinh = khoi_tao_mo_hinh_gnn(
        data_train=data_train,
        cau_hinh=cau_hinh,
        duong_dan_trong_so=str(thu_muc_trong_so / f"best_fold_{fold_id}.pth"),
    ).to(device)

    # Tao optimizer.
    optimizer = torch.optim.Adam(
        mo_hinh.parameters(), lr=cau_hinh.toc_do_hoc, weight_decay=cau_hinh.weight_decay
    )

    # Tao scheduler giam toc do hoc khi AUC dung lai.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cau_hinh.factor_lr,
        patience=cau_hinh.patience_lr,
        min_lr=cau_hinh.min_lr
        # verbose=False,  # 🔧 FIX: Đổi verbose=False (loại bỏ tham số không hợp lệ)
    )

    # Tao scaler cho AMP.
    # scaler = torch.amp.GradScaler("cuda", enabled=bat_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=bat_amp)

    

    # Luu AUC tot nhat.
    best_auc = -1.0
    dem_dung_som = 0

    # Tao loss co trong so nhe de tang Recall nhung van giu Precision on dinh.
    pos_weight = torch.tensor([cau_hinh.pos_weight_duong], device=device)
    ham_mat_mat = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # In header cho bang chi so.
    print(
        "Epoch           Time            AUC             AUPR            Accuracy                Precision               Recall          F1-score                Mcc"
    )

    # Bat dau huan luyen.
    for epoch in range(1, cau_hinh.so_epoch + 1):
        thoi_gian_bat_dau = time.perf_counter()

        # Chuyen du lieu train sang tensor.
        cap_train = torch.from_numpy(canh_train.T).long().to(device)
        nhan_train_t = torch.from_numpy(nhan_train).float().to(device)

        # Forward va tinh loss.
        mo_hinh.train()
        optimizer.zero_grad(set_to_none=True)

        if bat_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                emb = mo_hinh(data_train)
                logits = giai_ma_diem(emb, cap_train, so_thuoc)
                loss = ham_mat_mat(logits, nhan_train_t)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(mo_hinh.parameters(), cau_hinh.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            emb = mo_hinh(data_train)
            logits = giai_ma_diem(emb, cap_train, so_thuoc)
            loss = ham_mat_mat(logits, nhan_train_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mo_hinh.parameters(), cau_hinh.clip_grad)
            optimizer.step()

        # Danh gia tren tap test.
        mo_hinh.eval()
        with torch.no_grad():
            # Lay score tren train de tim nguong phan loai toi uu theo F1.
            diem_train = torch.sigmoid(giai_ma_diem(emb, cap_train, so_thuoc))
            cap_test = torch.from_numpy(canh_test.T).long().to(device)
            nhan_test_t = torch.from_numpy(nhan_test).float().to(device)
            emb_test = mo_hinh(data_train)
            diem_test = torch.sigmoid(giai_ma_diem(emb_test, cap_test, so_thuoc))

        # Tim nguong toi uu tren train va danh gia tren test.
        nguong_toi_uu = tim_nguong_toi_uu_f1(
            nhan_train_t.detach().cpu().numpy(),
            diem_train.detach().cpu().numpy(),
        )
        chi_so = tinh_chi_so(
            nhan_test_t.cpu().numpy(),
            diem_test.cpu().numpy(),
            nguong=nguong_toi_uu,
        )

        thoi_gian_ket_thuc = time.perf_counter()
        thoi_gian_epoch = thoi_gian_ket_thuc - thoi_gian_bat_dau

        # In bang chi so.
        print(
            f"{epoch:03d}             {thoi_gian_epoch:6.2f}s         "
            f"{chi_so['AUC']:.4f}          {chi_so['AUPR']:.4f}          "
            f"{chi_so['Accuracy']:.4f}                  {chi_so['Precision']:.4f}                 "
            f"{chi_so['Recall']:.4f}        {chi_so['F1']:.4f}                  {chi_so['MCC']:.4f}"
        )

        # Cap nhat scheduler theo AUC de hoc sau hon va on dinh hon.
        scheduler.step(chi_so["AUC"])

        # Luu checkpoint neu AUC tot hon.
        if chi_so["AUC"] > best_auc + cau_hinh.min_delta:
            best_auc = chi_so["AUC"]
            dem_dung_som = 0
            print(f"AUC improved at epoch {epoch}")
            torch.save(mo_hinh.state_dict(), thu_muc_trong_so / f"best_fold_{fold_id}.pth")
        else:
            dem_dung_som += 1
            print(f"Khong cai thien AUC: {dem_dung_som}/{cau_hinh.patience} epoch")
            if dem_dung_som >= cau_hinh.patience:
                print("Kich hoat Dung Som de chong Overfitting!")
                print(f"Dung som tai epoch {epoch} do khong cai thien AUC.")
                break

    return chi_so


# =========================
# HAM CHINH
# =========================

def parse_args() -> argparse.Namespace:
    # Tao parser va nap tham so.
    parser = argparse.ArgumentParser(description="Huan luyen FuzzyGCN voi K-Fold.")
    parser.add_argument("--dataset", type=str, default=CauHinh.ten_dataset)
    parser.add_argument("--epochs", type=int, default=CauHinh.so_epoch)
    parser.add_argument("--learning-rate", type=float, default=CauHinh.toc_do_hoc)
    parser.add_argument("--k-fold", type=int, default=CauHinh.so_fold)
    parser.add_argument("--negative-rate", type=float, default=CauHinh.ti_le_am)
    parser.add_argument("--patience", type=int, default=CauHinh.patience)
    parser.add_argument("--min-delta", type=float, default=CauHinh.min_delta)
    parser.add_argument("--lr-factor", type=float, default=CauHinh.factor_lr)
    parser.add_argument("--lr-patience", type=int, default=CauHinh.patience_lr)
    parser.add_argument("--min-lr", type=float, default=CauHinh.min_lr)
    parser.add_argument("--clip-grad", type=float, default=CauHinh.clip_grad)
    parser.add_argument("--pos-weight", type=float, default=CauHinh.pos_weight_duong)
    parser.add_argument("--device", type=str, default=CauHinh.thiet_bi)
    parser.add_argument("--amp", action="store_true", default=CauHinh.bat_amp)
    return parser.parse_args()


def main() -> None:
    # Doc tham so.
    #args = parse_args()
    args = parse_args()
    print(f"So fold: {args.k_fold}")

    # Tao cau hinh tu tham so.
    cau_hinh = CauHinh(
        ten_dataset=args.dataset,
        so_epoch=int(args.epochs),
        toc_do_hoc=float(args.learning_rate),
        so_fold=int(args.k_fold),
        ti_le_am=float(args.negative_rate),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        factor_lr=float(args.lr_factor),
        patience_lr=int(args.lr_patience),
        min_lr=float(args.min_lr),
        clip_grad=float(args.clip_grad),
        pos_weight_duong=float(args.pos_weight),
        thiet_bi=str(args.device),
        bat_amp=bool(args.amp),
    )

    # Dat seed.
    dat_seed(cau_hinh.seed)

    # Chuan bi duong dan.
    thu_muc_goc = Path(cau_hinh.thu_muc_goc)
    duong_dataset = thu_muc_goc / "dataset" / cau_hinh.ten_dataset
    thu_muc_trong_so = Path(cau_hinh.thu_muc_trong_so)
    thu_muc_trong_so.mkdir(parents=True, exist_ok=True)

    # Doc du lieu.
    canh_duong = doc_lien_ket(duong_dataset / cau_hinh.tep_lien_ket)
    max_thuoc = int(canh_duong[:, 0].max()) + 1
    max_benh = int(canh_duong[:, 1].max()) + 1

    thuoc_raw = doc_ma_tran(duong_dataset / cau_hinh.tep_thuoc)
    benh_raw = doc_ma_tran(duong_dataset / cau_hinh.tep_benh)
    thuoc = can_chinh_hang(thuoc_raw, max_thuoc, "Thuoc")
    benh = can_chinh_hang(benh_raw, max_benh, "Benh")

    so_thuoc = thuoc.shape[0]
    so_benh = benh.shape[0]

    # Loc lai canh duong theo kich thuoc thuc te de tranh loi index tren GPU.
    hop_le = (
        (canh_duong[:, 0] >= 0)
        & (canh_duong[:, 0] < so_thuoc)
        & (canh_duong[:, 1] >= 0)
        & (canh_duong[:, 1] < so_benh)
    )
    canh_duong = canh_duong[hop_le]

    # Tao tap canh duong.
    tap_duong = set(zip(canh_duong[:, 0].tolist(), canh_duong[:, 1].tolist()))

    # Tao canh am theo ti le.
    so_am = max(1, int(len(canh_duong) * cau_hinh.ti_le_am))
    canh_am = tao_canh_am(so_thuoc, so_benh, tap_duong, so_am)

    # Gop canh duong + am thanh dataset tong.
    canh_tat_ca = np.vstack([canh_duong, canh_am])
    nhan_tat_ca = np.hstack([
        np.ones(len(canh_duong), dtype=np.int64),
        np.zeros(len(canh_am), dtype=np.int64),
    ])

    # Chuan hoa dac trung neu bat.
    if cau_hinh.bat_chuan_hoa:
        thuoc, benh = chuan_hoa_dac_trung(thuoc, benh)

    # Tao StratifiedKFold.
    skf = StratifiedKFold(n_splits=cau_hinh.so_fold, shuffle=True, random_state=cau_hinh.seed)

    # Luu chi so trung binh.
    danh_sach_chi_so: List[Dict[str, float]] = []
    
    import shutil

    # Xoa toan bo folder fold cu de tranh nham lan
    thu_muc_fold_goc = duong_dataset / "fold"
    if thu_muc_fold_goc.exists():
        shutil.rmtree(thu_muc_fold_goc)
        print(f"Da xoa folder fold cu: {thu_muc_fold_goc}")
        

    # Bat dau K-Fold.
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(canh_tat_ca, nhan_tat_ca), start=0):
        print(f"\n=== Fold {fold_id + 1}/{cau_hinh.so_fold} ===")

        canh_train = canh_tat_ca[train_idx]
        nhan_train = nhan_tat_ca[train_idx]
        canh_test = canh_tat_ca[test_idx]
        nhan_test = nhan_tat_ca[test_idx]

        # Luu data_train.csv va data_test.csv vao fold tuong ung (0-based)
        thu_muc_fold = duong_dataset / "fold" / str(fold_id)
        thu_muc_fold.mkdir(parents=True, exist_ok=True)

        pd.DataFrame({
            "drug": canh_train[:, 0],
            "disease": canh_train[:, 1],
            "label": nhan_train,
        }).to_csv(thu_muc_fold / "data_train.csv", index=False)

        pd.DataFrame({
            "drug": canh_test[:, 0],
            "disease": canh_test[:, 1],
            "label": nhan_test,
        }).to_csv(thu_muc_fold / "data_test.csv", index=False)

        print(f"Da luu fold {fold_id} vao {thu_muc_fold}")

        # Tao do thi chi tu tap train duong
        data_train = tao_do_thi(thuoc, benh, canh_train[nhan_train == 1])

        # Huan luyen 1 fold
        chi_so = huan_luyen_1_fold(
            cau_hinh=cau_hinh,
            data_train=data_train,
            canh_train=canh_train,
            nhan_train=nhan_train,
            canh_test=canh_test,
            nhan_test=nhan_test,
            so_thuoc=so_thuoc,
            so_benh=so_benh,
            thu_muc_trong_so=thu_muc_trong_so,
            fold_id=fold_id + 1,
        )

        danh_sach_chi_so.append(chi_so)

    # Tinh trung binh toan bo fold.
    trung_binh: Dict[str, float] = {}
    for key in danh_sach_chi_so[0].keys():
        trung_binh[key] = float(np.mean([cs[key] for cs in danh_sach_chi_so]))

    print("\n=== Ket qua trung binh K-Fold ===")
    print(
        f"AUC={trung_binh['AUC']:.4f}, AUPR={trung_binh['AUPR']:.4f}, "
        f"Accuracy={trung_binh['Accuracy']:.4f}, Precision={trung_binh['Precision']:.4f}, "
        f"Recall={trung_binh['Recall']:.4f}, F1={trung_binh['F1']:.4f}, MCC={trung_binh['MCC']:.4f}"
    )
# Luu thong so lan chay vao file .txt
    import datetime
    thoi_gian = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    tep_thong_so = duong_dataset / f"ket_qua_{thoi_gian}.txt"
    with open(tep_thong_so, "w", encoding="utf-8") as f:
        f.write("=" * 50 + "\n")
        f.write(f"THONG SO LAN CHAY: {thoi_gian}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("[CAU HINH]\n")
        f.write(f"Dataset        : {cau_hinh.ten_dataset}\n")
        f.write(f"So fold        : {cau_hinh.so_fold}\n")
        f.write(f"So epoch       : {cau_hinh.so_epoch}\n")
        f.write(f"Toc do hoc     : {cau_hinh.toc_do_hoc}\n")
        f.write(f"Weight decay   : {cau_hinh.weight_decay}\n")
        f.write(f"Kich thuoc an  : {cau_hinh.kich_thuoc_an}\n")
        f.write(f"Kich thuoc ra  : {cau_hinh.kich_thuoc_ra}\n")
        f.write(f"Ti le am       : {cau_hinh.ti_le_am}\n")
        f.write(f"Patience       : {cau_hinh.patience}\n")
        f.write(f"Min delta      : {cau_hinh.min_delta}\n")
        f.write(f"Factor LR      : {cau_hinh.factor_lr}\n")
        f.write(f"Patience LR    : {cau_hinh.patience_lr}\n")
        f.write(f"Min LR         : {cau_hinh.min_lr}\n")
        f.write(f"Clip grad      : {cau_hinh.clip_grad}\n")
        f.write(f"Pos weight     : {cau_hinh.pos_weight_duong}\n")
        f.write(f"Thiet bi       : {cau_hinh.thiet_bi}\n")
        f.write(f"Bat AMP        : {cau_hinh.bat_amp}\n")
        f.write(f"Seed           : {cau_hinh.seed}\n\n")
        
        f.write("[KET QUA TUNG FOLD]\n")
        for i, cs in enumerate(danh_sach_chi_so):
            f.write(f"Fold {i:2d} | AUC={cs['AUC']:.4f} | AUPR={cs['AUPR']:.4f} | "
                    f"Acc={cs['Accuracy']:.4f} | Prec={cs['Precision']:.4f} | "
                    f"Rec={cs['Recall']:.4f} | F1={cs['F1']:.4f} | MCC={cs['MCC']:.4f}\n")
        
        f.write("\n[KET QUA TRUNG BINH]\n")
        f.write(f"AUC       : {trung_binh['AUC']:.4f}\n")
        f.write(f"AUPR      : {trung_binh['AUPR']:.4f}\n")
        f.write(f"Accuracy  : {trung_binh['Accuracy']:.4f}\n")
        f.write(f"Precision : {trung_binh['Precision']:.4f}\n")
        f.write(f"Recall    : {trung_binh['Recall']:.4f}\n")
        f.write(f"F1        : {trung_binh['F1']:.4f}\n")
        f.write(f"MCC       : {trung_binh['MCC']:.4f}\n")

    print(f"Da luu thong so vao: {tep_thong_so}")
    
    # Luu ket qua trung binh ra file json.
    tep_kq = thu_muc_trong_so / "kfold_metrics.json"
    tep_kq.write_text(json.dumps(trung_binh, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
