from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData


def chuyen_sang_dong_nhat(
    data: HeteroData,
    ma_hoa_thuoc: torch.nn.Module,
    ma_hoa_benh: torch.nn.Module,
    lop_fuzzy: torch.nn.Module,
) -> HeteroData:
    # Chuyen du lieu ve do thi dong nhat sau khi ma hoa va giam nhieu.
    data = data.clone()
    dac_trung_thuoc = ma_hoa_thuoc(data["drug"].x)
    dac_trung_benh = ma_hoa_benh(data["disease"].x)
    data["drug"].x = lop_fuzzy(dac_trung_thuoc)
    data["disease"].x = lop_fuzzy(dac_trung_benh)
    return data.to_homogeneous(node_attrs=["x"])


def tinh_trong_so_canh(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    # Tinh trong so canh dua tren do tuong dong cosine.
    src, dst = edge_index
    cos_sim = F.cosine_similarity(x[src], x[dst])
    return torch.exp(-((1.0 - cos_sim) ** 2) / 2.0)


def truyen_qua_cac_lop_gcn(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    cac_lop_gcn: torch.nn.ModuleList,
    dang_huan_luyen: bool,
) -> torch.Tensor:
    # Chay qua toan bo cac lop GCN va tra ve embedding cuoi.
    for i, lop in enumerate(cac_lop_gcn):
        x = lop(x, edge_index, edge_weight=edge_weight)
        if i != len(cac_lop_gcn) - 1:
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=dang_huan_luyen)
    return x
