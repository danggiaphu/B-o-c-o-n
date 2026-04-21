"""
Khởi động toàn bộ ứng dụng MedLink AI từ 1 file duy nhất.

  python main.py              # chạy cả backend (8000) + frontend (8501)
  python main.py --api-only   # chỉ chạy FastAPI
  python main.py --ui-only    # chỉ chạy Streamlit
  python main.py --api-port 8080 --ui-port 8502
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

import uvicorn

# ── Cấu hình đường dẫn ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "src" / "backend"
FRONTEND_ENTRY = ROOT / "src" / "frontend" / "streamlit_app.py"

# Đảm bảo backend có thể import được
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


# ── Khởi động FastAPI (chạy trong thread nền) ────────────────────────────────
def _run_api(host: str, port: int, reload: bool) -> None:
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
        app_dir=str(BACKEND_DIR),
        log_level="info",
    )


def start_api(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> threading.Thread:
    t = threading.Thread(target=_run_api, args=(host, port, reload), daemon=True, name="fastapi")
    t.start()
    return t


# ── Khởi động Streamlit (chạy subprocess, chiếm terminal chính) ──────────────
def start_streamlit(host: str = "localhost", port: int = 8501) -> None:
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(FRONTEND_ENTRY),
        "--server.address", host,
        "--server.port", str(port),
        "--server.headless", "true",
    ]
    subprocess.run(cmd, check=False)


# ── Parse args ───────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MedLink AI — khởi động toàn bộ ứng dụng")
    p.add_argument("--api-host",  default="127.0.0.1", help="Host FastAPI (mặc định: 127.0.0.1)")
    p.add_argument("--api-port",  type=int, default=8000, help="Port FastAPI (mặc định: 8000)")
    p.add_argument("--ui-host",   default="localhost",  help="Host Streamlit (mặc định: localhost)")
    p.add_argument("--ui-port",   type=int, default=8501, help="Port Streamlit (mặc định: 8501)")
    p.add_argument("--reload",    action="store_true",  help="Bật auto-reload cho FastAPI (dev)")
    p.add_argument("--api-only",  action="store_true",  help="Chỉ chạy FastAPI backend")
    p.add_argument("--ui-only",   action="store_true",  help="Chỉ chạy Streamlit frontend")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    if args.api_only:
        # ── Chỉ chạy API (blocking) ──────────────────────────────────
        print(f"[API]  http://{args.api_host}:{args.api_port}")
        _run_api(args.api_host, args.api_port, args.reload)

    elif args.ui_only:
        # ── Chỉ chạy Streamlit (blocking) ────────────────────────────
        print(f"[UI]   http://{args.ui_host}:{args.ui_port}")
        start_streamlit(args.ui_host, args.ui_port)

    else:
        # ── Chạy cả hai ──────────────────────────────────────────────
        print("=" * 55)
        print("  MedLink AI — Khởi động ứng dụng")
        print("=" * 55)
        print(f"  [API]  http://{args.api_host}:{args.api_port}")
        print(f"  [UI]   http://{args.ui_host}:{args.ui_port}")
        print("=" * 55)

        # Khởi động FastAPI trong thread nền
        api_thread = start_api(args.api_host, args.api_port, args.reload)

        # Chờ API sẵn sàng trước khi mở UI
        print("  Đang chờ API khởi động", end="", flush=True)
        for _ in range(20):
            time.sleep(0.5)
            if api_thread.is_alive():
                print(".", end="", flush=True)
                break
        time.sleep(1.5)
        print(" OK")

        # Khởi động Streamlit (chiếm terminal chính — blocking)
        start_streamlit(args.ui_host, args.ui_port)
