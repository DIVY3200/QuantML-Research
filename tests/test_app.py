"""Headless smoke test of the Streamlit dashboard on synthetic data (no network)."""

from __future__ import annotations

from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_dashboard_runs_on_synthetic_data():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Synthetic demo data").run()
    run_button = next(b for b in at.button if b.label == "Run research")
    run_button.click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error or "SYNTHETIC" in at.error[0].value
    assert "result" in at.session_state
    assert len(at.tabs) == 7
    assert any("Sharpe" in m.value for m in at.metric)
