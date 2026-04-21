from __future__ import annotations

try:
    from ..frontend.streamlit_app import main
except ImportError:
    try:
        from backend.app.frontend.streamlit_app import main
    except ImportError:
        from frontend.streamlit_app import main


if __name__ == "__main__":
    main()
