from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


def tao_cac_lop_gcn(
    so_lop_gcn: int,
    so_chieu_an: int,
    so_chieu_ra: int,
) -> nn.ModuleList:
    # Tao danh sach cac lop GCN theo cau hinh.
    cac_lop_gcn = nn.ModuleList()
    if so_lop_gcn == 1:
        cac_lop_gcn.append(GCNConv(so_chieu_an, so_chieu_ra))
        return cac_lop_gcn

    cac_lop_gcn.append(GCNConv(so_chieu_an, so_chieu_an))
    for _ in range(so_lop_gcn - 2):
        cac_lop_gcn.append(GCNConv(so_chieu_an, so_chieu_an))
    cac_lop_gcn.append(GCNConv(so_chieu_an, so_chieu_ra))
    return cac_lop_gcn


def tinh_trong_so_canh(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    # Tinh trong so canh dua tren do tuong dong cosine.
    src, dst = edge_index
    cos_sim = F.cosine_similarity(x[src], x[dst])
    return torch.exp(-((1.0 - cos_sim) ** 2) / 2.0)


def truyen_qua_cac_lop_gcn(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    cac_lop_gcn: nn.ModuleList,
    dang_huan_luyen: bool,
) -> torch.Tensor:
    # Chay qua toan bo cac lop GCN va tra ve embedding cuoi.
    for i, lop in enumerate(cac_lop_gcn):
        x = lop(x, edge_index, edge_weight=edge_weight)
        if i != len(cac_lop_gcn) - 1:
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=dang_huan_luyen)
    return x


def chay_forward_gcn(
    homo: Data,
    cac_lop_gcn: nn.ModuleList,
    dang_huan_luyen: bool,
) -> torch.Tensor:
    # Luong forward GCN: tinh trong so canh -> truyen qua cac lop GCN.
    x, edge_index = homo.x, homo.edge_index
    edge_weight = tinh_trong_so_canh(x, edge_index)
    return truyen_qua_cac_lop_gcn(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        cac_lop_gcn=cac_lop_gcn,
        dang_huan_luyen=dang_huan_luyen,
    )
