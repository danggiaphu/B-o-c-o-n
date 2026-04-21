"""
Script tạo database He_Thong_Du_Doan_Thuoc trong SQL Server.
Chạy: python create_db.py
"""
from __future__ import annotations

import os
from sqlalchemy import create_engine, text

# Kết nối đến master database để tạo database mới
MASTER_URL = (
    "mssql+pyodbc://LAPTOP-OGHK3ST3\\SQLEXPRESS/master"
    "?driver=ODBC+Driver+17+for+SQL+Server"
    "&trusted_connection=yes"
    "&TrustServerCertificate=yes"
)

def create_database() -> None:
    engine = create_engine(MASTER_URL, isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        # Kiểm tra xem database đã tồn tại chưa
        result = conn.execute(text("SELECT name FROM sys.databases WHERE name = 'He_Thong_Du_Doan_Thuoc'"))
        if result.fetchone():
            print("Database 'He_Thong_Du_Doan_Thuoc' đã tồn tại.")
            return

        # Tạo database
        conn.execute(text("CREATE DATABASE [He_Thong_Du_Doan_Thuoc]"))
        print("Database 'He_Thong_Du_Doan_Thuoc' đã được tạo thành công.")

if __name__ == "__main__":
    create_database()