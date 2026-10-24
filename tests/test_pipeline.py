"""Configuration handling and the end-to-end pipeline on synthetic data."""

from __future__ import annotations

import pandas as pd
import pytest

from quantml.config import ConfigError, ResearchConfig, load_config
from quantml.pipeline import load_and_prepare, run_research
from quantml.reporting import all_metrics_frame, predictions_frame, save_run


def test_yaml_defaults_match_dataclass_defaults():
    assert load_config().to_dict() == ResearchConfig().to_dict()


@pytest.mark.parametrize("overrides, message", [
    ({"split": {"train": 0.7}}, "sum to 1"),
    ({"strategy": {"ma_short": 50, "ma_long": 20}}, "ma_short"),
    ({"backtest": {"execution": "same_close"}}, "execution"),
    ({"strategy": {"ml_threshold": 1.2}}, "ml_threshold"),
    ({"unknown_section": {}}, "Unknown"),
])
def test_invalid_config_is_rejected(overrides, message):
    with pytest.raises(ConfigError, match=message):
        load_config(overrides=overrides)


@pytest.fixture
def research(small_config):
    prepared = load_and_prepare(small_config)
    return run_research(prepared.prices, small_config, prepared.provenance())


def test_pipeline_end_to_end(research):
    assert research.lookahead_audit.passed
    seg = research.segments
    assert seg["train"].end < seg["validation"].start <= seg["validation"].end < seg["test"].start
    for s in seg.values():
        indices = [bt.daily.index for bt in s.backtests.values()]
        assert all(idx.equals(indices[0]) for idx in indices), "identical dates for all strategies"
    assert research.run_info["data"]["is_synthetic"] is True
    assert set(research.signals) == {"Buy & Hold", "MA Crossover (20/50)", research.ml_name}


def test_walk_forward_is_out_of_sample(research):
    wf = research.walk_forward
    assert wf is not None and len(wf.folds) >= 2
    assert (wf.folds["train_end"] < wf.folds["test_start"]).all()
    assert wf.folds["test_start"].is_monotonic_increasing
    assert (wf.folds["test_start"].iloc[1:].to_numpy() > wf.folds["test_end"].iloc[:-1].to_numpy()).all()
    assert wf.probabilities.index.is_unique


def test_threshold_selected_on_validation_comes_from_grid(small_config):
    cfg = small_config.with_overrides({"strategy": {"threshold_selection": "validation"},
                                       "walk_forward": {"enabled": False}})
    prepared = load_and_prepare(cfg)
    result = run_research(prepared.prices, cfg)
    assert result.threshold in cfg.strategy.threshold_grid
    assert not result.threshold_table.empty
    assert result.walk_forward is None


def test_random_forest_runs(small_config):
    cfg = small_config.with_overrides({"model": {"name": "random_forest",
                                                 "random_forest": {"n_estimators": 30}},
                                       "walk_forward": {"enabled": False}})
    prepared = load_and_prepare(cfg)
    result = run_research(prepared.prices, cfg)
    assert result.probabilities.between(0, 1).all()


def test_reports_are_written(research, tmp_path):
    folder = save_run(research, tmp_path / "run")
    for name in ("config.yaml", "run_info.json", "metrics.csv", "predictions.csv",
                 "daily_results.csv", "summary.md", "walk_forward_folds.csv"):
        assert (folder / name).exists(), name
    metrics = pd.read_csv(folder / "metrics.csv")
    assert {"segment", "strategy", "sharpe_net"} <= set(metrics.columns)
    assert "SYNTHETIC" in (folder / "summary.md").read_text(encoding="utf-8")
    assert len(all_metrics_frame(research)) == 12
    assert not predictions_frame(research)["prob_up"].isna().any()


def test_compact_reports_skip_daily_files(research, tmp_path):
    folder = save_run(research, tmp_path / "compact", compact=True)
    assert (folder / "metrics.csv").exists() and (folder / "walk_forward_folds.csv").exists()
    for name in ("daily_results.csv", "predictions.csv", "walk_forward_daily.csv"):
        assert not (folder / name).exists(), name
