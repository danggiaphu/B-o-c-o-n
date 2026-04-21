from __future__ import annotations

import torch
import torch.nn as nn


class LopFuzzy(nn.Module):
    # Ham khoi tao lop fuzzy, dung de giam nhieu tren dac trung dau vao.
    def __init__(self, sigma_khoi_tao: float = 1.0) -> None:
        super().__init__()  # Goi lop cha cua PyTorch de khoi tao dung cach.
        self.mu = nn.Parameter(torch.tensor(0.0))  # Tham so trung binh mu co the hoc duoc.
        self.sigma = nn.Parameter(torch.tensor(sigma_khoi_tao))  # Tham so do lech chuan sigma hoc duoc.

    # Ham lan truyen thuan cua lop fuzzy.
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sigma = torch.clamp(self.sigma, min=1e-6)  # Chan sigma nho qua de tranh chia 0.
        membership = torch.exp(-((x - self.mu) ** 2) / (2 * sigma**2))  # Tinh ham thanh vien Gaussian.
        return x * membership  # Nhan voi x de giam anh huong cua nhieu.
