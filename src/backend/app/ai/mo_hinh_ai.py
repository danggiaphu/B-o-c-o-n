from __future__ import annotations

# Import thu vien co ban de dinh nghia duong dan va kieu du lieu.
from pathlib import Path
from typing import Dict, List

# Import thu vien PyTorch de xay dung mo hinh hoc sau.
import torch
import torch.nn as nn

# Import HeteroData tu PyTorch Geometric de lam viec voi do thi.
from torch_geometric.data import HeteroData

try:
    from .fuzzy_layer import LopFuzzy
    from .gcn_flow import chay_forward_gcn, tao_cac_lop_gcn
    from .gnn_algorithm import chuyen_sang_dong_nhat
except ImportError:
    from fuzzy_layer import LopFuzzy
    from gcn_flow import chay_forward_gcn, tao_cac_lop_gcn
    from gnn_algorithm import chuyen_sang_dong_nhat


class FuzzyGCN(nn.Module):
    # Ham khoi tao mo hinh FuzzyGCN.
    def __init__(
        self,
        so_chieu_thuoc: int,
        so_chieu_benh: int,
        so_chieu_an: int = 64,
        so_chieu_ra: int = 32,
        so_lop_gcn: int = 3,
        duong_dan_trong_so: str = "weights/fuzzy_gcn.pth",
    ) -> None:
        super().__init__()  # Goi lop cha cua PyTorch.
        if so_lop_gcn < 1:  # Kiem tra so lop hop le.
            raise ValueError("so_lop_gcn phai >= 1")

        # Lop ma hoa dac trung thuoc ve khong gian an.
        self.ma_hoa_thuoc = nn.Linear(so_chieu_thuoc, so_chieu_an)
        # Lop ma hoa dac trung benh ve khong gian an.
        self.ma_hoa_benh = nn.Linear(so_chieu_benh, so_chieu_an)

        # Lop fuzzy de giam nhieu truoc khi dua vao GCN.
        self.lop_fuzzy = LopFuzzy(sigma_khoi_tao=1.0)

        # Luu so lop GCN de dung trong forward.
        self.so_lop_gcn = so_lop_gcn
        # Tao danh sach cac lop GCN tu luong rieng.
        self.cac_lop_gcn = tao_cac_lop_gcn(
            so_lop_gcn=so_lop_gcn,
            so_chieu_an=so_chieu_an,
            so_chieu_ra=so_chieu_ra,
        )

        # Luu duong dan trong so de doc/ghi.
        self.duong_dan_trong_so = Path(duong_dan_trong_so)

        # Tu dien gan id benh sang ten benh (co the nap tu pipeline huan luyen).
        self.id_sang_ten_benh: Dict[int, str] = {}

    # Ham chuyen do thi HeteroData sang do thi dong nhat (homogeneous).
    def _chuyen_sang_dong_nhat(self, data: HeteroData) -> HeteroData:
        return chuyen_sang_dong_nhat(
            data=data,
            ma_hoa_thuoc=self.ma_hoa_thuoc,
            ma_hoa_benh=self.ma_hoa_benh,
            lop_fuzzy=self.lop_fuzzy,
        )

    # Ham forward cua mo hinh.
    def forward(self, data: HeteroData) -> torch.Tensor:
        homo = self._chuyen_sang_dong_nhat(data)  # Chuyen do thi sang dang dong nhat.
        return chay_forward_gcn(
            homo=homo,
            cac_lop_gcn=self.cac_lop_gcn,
            dang_huan_luyen=self.training,
        )

    # Ham nap trong so neu co san.
    def tai_trong_so(self) -> bool:
        if not self.duong_dan_trong_so.exists():  # Kiem tra file co ton tai khong.
            return False
        state = torch.load(self.duong_dan_trong_so, map_location="cpu")  # Doc trong so.
        self.load_state_dict(state)  # Nap trong so vao mo hinh.
        self.eval()  # Chuyen ve che do danh gia.
        return True

    # Ham du doan mo phong Top-K.
    @torch.no_grad()
    def du_doan_top_k(self, drug_id: int, k: int = 5) -> List[Dict[str, float]]:
        self.tai_trong_so()  # Co gang nap trong so truoc khi du doan.

        k = max(1, int(k))  # Dam bao k >= 1.
        torch.manual_seed(int(drug_id) + 2026)  # Dat seed de ket qua on dinh.

        # Gia lap 20 benh neu chua co bang id->ten.
        tong_benh = max(20, len(self.id_sang_ten_benh))
        logits_class_1 = torch.randn(tong_benh)  # Logits cho class 1 (co lien ket).
        logits_2_lop = torch.stack([-logits_class_1, logits_class_1], dim=1)  # [N, 2]
        probs_2_lop = torch.softmax(logits_2_lop, dim=1)  # Xac suat tung class.
        xac_suat_class_1 = probs_2_lop[:, 1]  # Lay xac suat class co lien ket.

        top_probs, top_indices = torch.topk(xac_suat_class_1, k=min(k, tong_benh))  # Lay Top-K.

        ket_qua: List[Dict[str, float]] = []  # Tao danh sach ket qua.
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            ten_benh = self.id_sang_ten_benh.get(idx, f"Benh_{idx}")  # Lay ten benh.
            ket_qua.append(
                {
                    "disease_id": idx,
                    "disease_name": ten_benh,
                    # Cot xac suat chuan (Class 1 - co lien ket) de frontend loc theo threshold.
                    "Probability": float(prob),
                    # Giu score de tuong thich nguoc voi ma cu.
                    "score": float(prob),
                }
            )

        return ket_qua
