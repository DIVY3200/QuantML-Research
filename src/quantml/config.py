"""Typed experiment configuration loaded from YAML.

The YAML file in ``config/default_config.yaml`` and the dataclass defaults below describe the
same defaults; a unit test keeps them in sync. Every run serialises the fully resolved
configuration with :meth:`ResearchConfig.to_dict` so that experiments can be reproduced.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.metadata
import platform
import sys
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_config.yaml"

DATA_SOURCES = ("yfinance", "csv", "synthetic")
EXECUTION_MODES = ("next_open", "next_close")
TARGET_BASES = ("close", "execution")
MODEL_NAMES = ("logistic_regression", "random_forest")
THRESHOLD_SELECTION_MODES = ("fixed", "validation")
WINDOW_TYPES = ("expanding", "rolling")

_TRACKED_PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "statsmodels",
    "yfinance", "plotly", "streamlit", "PyYAML",
)


class ConfigError(ValueError):
    """Raised when a configuration value is missing, unknown or out of range."""


@dataclass
class DataConfig:
    source: str = "yfinance"
    ticker: str = "SPY"
    start: str = "2010-01-01"
    end: str = "2024-12-31"
    csv_path: str | None = None
    cache_dir: str = "data/raw"
    use_cache: bool = True


@dataclass
class ValidationConfig:
    min_observations: int = 500
    max_gap_calendar_days: int = 5
    ohlc_tolerance: float = 0.001
    extreme_return_threshold: float = 0.25


@dataclass
class FeatureConfig:
    return_lags: int = 5
    momentum_windows: list[int] = field(default_factory=lambda: [5, 10, 20])
    volatility_windows: list[int] = field(default_factory=lambda: [10, 20])
    ma_distance_windows: list[int] = field(default_factory=lambda: [10, 20, 50])
    ma_ratio_pairs: list[list[int]] = field(default_factory=lambda: [[5, 20], [20, 50]])
    range_window: int = 20
    volume_window: int = 20
    use_volume: str = "auto"


@dataclass
class TargetConfig:
    horizon: int = 1
    basis: str = "close"
    positive_threshold: float = 0.0


@dataclass
class SplitConfig:
    train: float = 0.6
    validation: float = 0.2
    test: float = 0.2
    embargo: int = 0


@dataclass
class ModelConfig:
    name: str = "logistic_regression"
    tune_on_validation: bool = True
    refit_on_train_and_validation: bool = False
    logistic_regression: dict[str, Any] = field(default_factory=lambda: {
        "C": 1.0,
        "C_grid": [0.001, 0.01, 0.1, 1.0, 10.0],
        "max_iter": 2000,
        "class_weight": None,
    })
    random_forest: dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 4,
        "max_depth_grid": [2, 4, 6],
        "min_samples_leaf": 50,
        "max_features": "sqrt",
        "class_weight": None,
    })


@dataclass
class StrategyConfig:
    ma_short: int = 20
    ma_long: int = 50
    ml_threshold: float = 0.5
    threshold_selection: str = "fixed"
    threshold_grid: list[float] = field(default_factory=lambda: [0.48, 0.50, 0.52, 0.54, 0.56])
    below_threshold_position: float = 0.0


@dataclass
class BacktestConfig:
    execution: str = "next_open"
    fee_bps: float = 1.0
    slippage_bps: float = 2.0
    initial_capital: float = 10000.0


@dataclass
class WalkForwardConfig:
    enabled: bool = True
    initial_train_size: int = 756
    test_size: int = 63
    window: str = "expanding"
    max_train_size: int | None = None


@dataclass
class BootstrapConfig:
    enabled: bool = True
    n_resamples: int = 1000
    block_size: int = 20
    confidence: float = 0.95


@dataclass
class EvaluationConfig:
    periods_per_year: int = 252
    risk_free_rate: float = 0.0
    min_obs_annualize: int = 60
    cost_grid_bps: list[float] = field(default_factory=lambda: [0, 1, 2, 5, 10, 20, 50])
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)


@dataclass
class OutputConfig:
    reports_dir: str = "reports"


@dataclass
class ResearchConfig:
    """Complete, serialisable description of one research experiment."""

    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def with_overrides(self, overrides: dict[str, Any]) -> "ResearchConfig":
        """Return a new validated config with ``overrides`` deep-merged on top of this one."""
        return config_from_dict(deep_merge(self.to_dict(), overrides))

    def validate(self) -> None:
        """Raise :class:`ConfigError` describing the first invalid setting found."""
        _check(self.data.source in DATA_SOURCES, f"data.source must be one of {DATA_SOURCES}")
        _check(self.backtest.execution in EXECUTION_MODES,
               f"backtest.execution must be one of {EXECUTION_MODES}")
        _check(self.target.basis in TARGET_BASES, f"target.basis must be one of {TARGET_BASES}")
        _check(self.target.horizon >= 1, "target.horizon must be >= 1")
        _check(self.model.name in MODEL_NAMES, f"model.name must be one of {MODEL_NAMES}")
        _check(self.strategy.threshold_selection in THRESHOLD_SELECTION_MODES,
               f"strategy.threshold_selection must be one of {THRESHOLD_SELECTION_MODES}")
        _check(self.walk_forward.window in WINDOW_TYPES,
               f"walk_forward.window must be one of {WINDOW_TYPES}")

        s = self.split
        _check(s.train > 0 and s.test > 0 and s.validation >= 0,
               "split.train and split.test must be > 0 and split.validation >= 0")
        _check(abs(s.train + s.validation + s.test - 1.0) < 1e-6,
               f"split proportions must sum to 1 (got {s.train + s.validation + s.test:.4f})")
        _check(s.embargo >= 0, "split.embargo must be >= 0")
        _check(s.validation > 0 or (self.strategy.threshold_selection == "fixed"
                                    and not self.model.tune_on_validation),
               "validation proportion must be > 0 when tuning or selecting thresholds on validation")

        st = self.strategy
        _check(1 <= st.ma_short < st.ma_long, "strategy.ma_short must be >= 1 and < strategy.ma_long")
        _check(0.0 < st.ml_threshold < 1.0, "strategy.ml_threshold must be strictly between 0 and 1")
        _check(all(0.0 < t < 1.0 for t in st.threshold_grid), "threshold_grid values must be in (0, 1)")
        _check(0.0 <= st.below_threshold_position < 1.0,
               "strategy.below_threshold_position must be in [0, 1)")

        bt = self.backtest
        _check(bt.fee_bps >= 0 and bt.slippage_bps >= 0, "fee_bps and slippage_bps must be >= 0")
        _check(bt.initial_capital > 0, "backtest.initial_capital must be > 0")

        f = self.features
        windows = [*f.momentum_windows, *f.volatility_windows, *f.ma_distance_windows,
                   f.range_window, f.volume_window]
        _check(f.return_lags >= 1, "features.return_lags must be >= 1")
        _check(all(int(w) >= 2 for w in windows), "feature windows must be >= 2")
        _check(all(len(p) == 2 and 1 <= p[0] < p[1] for p in f.ma_ratio_pairs),
               "features.ma_ratio_pairs must be [short, long] with short < long")
        _check(str(f.use_volume).lower() in ("auto", "true", "false"),
               "features.use_volume must be auto, true or false")

        wf = self.walk_forward
        _check(wf.initial_train_size >= 50, "walk_forward.initial_train_size must be >= 50")
        _check(wf.test_size >= 1, "walk_forward.test_size must be >= 1")

        ev = self.evaluation
        _check(ev.periods_per_year > 0, "evaluation.periods_per_year must be > 0")
        _check(ev.risk_free_rate > -1, "evaluation.risk_free_rate must be > -100%")
        _check(0 < ev.bootstrap.confidence < 1, "bootstrap.confidence must be in (0, 1)")
        _check(ev.bootstrap.block_size >= 1, "bootstrap.block_size must be >= 1")
        _check(self.validation.min_observations >= 50, "validation.min_observations must be >= 50")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` into a copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in (overrides or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _build(cls: type, values: dict[str, Any], path: str) -> Any:
    if not isinstance(values, dict):
        raise ConfigError(f"Section '{path}' must be a mapping, got {type(values).__name__}")
    hints = typing.get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ConfigError(f"Unknown configuration key(s) in '{path or 'root'}': {unknown}")
    kwargs = {}
    for name, value in values.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, value, f"{path}.{name}".strip("."))
        else:
            kwargs[name] = value
    return cls(**kwargs)


def config_from_dict(values: dict[str, Any]) -> ResearchConfig:
    """Build and validate a :class:`ResearchConfig` from a (possibly partial) mapping."""
    cfg = _build(ResearchConfig, values or {}, "")
    cfg.validate()
    return cfg


def load_config(path: str | Path | None = None,
                overrides: dict[str, Any] | None = None) -> ResearchConfig:
    """Load a YAML config (default: ``config/default_config.yaml``) and apply overrides."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return config_from_dict(deep_merge(raw, overrides or {}))


def resolve_path(path: str | Path) -> Path:
    """Resolve a config path relative to the project root (absolute paths are kept)."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def environment_info() -> dict[str, Any]:
    """Python, platform and package versions, recorded with every run."""
    versions: dict[str, str | None] = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
    }
