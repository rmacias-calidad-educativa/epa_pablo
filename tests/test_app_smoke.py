from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_starts_without_exception():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(
        app_path,
        default_timeout=60,
    ).run()

    assert not at.exception
