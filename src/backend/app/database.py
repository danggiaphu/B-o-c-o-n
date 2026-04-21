from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from .models import Base

# ── SQL Server (Windows Auth) ────────────────────────────────────────────────
# Có thể override qua biến môi trường MODEL_GNN_DB_URL
_DEFAULT_URL = (
    "mssql+pyodbc://LAPTOP-OGHK3ST3\\SQLEXPRESS/He_Thong_Du_Doan_Thuoc"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&trusted_connection=yes"
    "&TrustServerCertificate=yes"
)

SQLALCHEMY_DATABASE_URL: str = os.getenv("MODEL_GNN_DB_URL", _DEFAULT_URL)

_is_mssql = SQLALCHEMY_DATABASE_URL.startswith("mssql")

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    fast_executemany=True,   # tăng tốc bulk insert cho pyodbc
    pool_pre_ping=True,      # kiểm tra connection trước khi dùng
    pool_size=5,
    max_overflow=10,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db() -> None:
    """Tạo toàn bộ bảng nếu chưa tồn tại."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
