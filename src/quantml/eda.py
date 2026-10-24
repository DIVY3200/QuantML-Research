"""Exploratory statistics for a single instrument.

Price *levels* are non-stationary and are not comparable across time; *returns* (relative
changes) are the quantity strategies earn and the basis for every statistic below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox


def daily_returns(prices: pd.DataFrame) -> pd.Series:
    return prices["close"].pct_change(fill_method=None).rename("daily_return")


def return_summary(prices: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """Observation count, return moments and missing-value counts (returns as decimals)."""
    r = daily_returns(prices).dropna()
    rows = {
        "Observations (price rows)": len(prices),
        "Return observations": len(r),
        "Mean daily return": r.mean(),
        "Daily return volatility (std)": r.std(ddof=1),
        "Annualised mean return (x periods)": r.mean() * periods_per_year,
        "Annualised volatility (x sqrt(periods))": r.std(ddof=1) * np.sqrt(periods_per_year),
        "Minimum daily return": r.min(),
        "Maximum daily return": r.max(),
        "Skewness": stats.skew(r) if len(r) > 2 else np.nan,
        "Excess kurtosis": stats.kurtosis(r) if len(r) > 3 else np.nan,
        "Missing values (OHLC)": int(prices[["open", "high", "low", "close"]].isna().sum().sum()),
        "Missing values (volume)": int(prices["volume"].isna().sum()) if "volume" in prices else 0,
    }
    return pd.DataFrame({"value": rows})


def rolling_statistics(prices: pd.DataFrame, window: int = 21,
                       periods_per_year: int = 252) -> pd.DataFrame:
    r = daily_returns(prices)
    return pd.DataFrame({
        "rolling_mean_return": r.rolling(window, min_periods=window).mean(),
        "rolling_volatility_ann": r.rolling(window, min_periods=window).std() * np.sqrt(periods_per_year),
    }, index=prices.index)


def price_drawdown(prices: pd.DataFrame) -> pd.Series:
    close = prices["close"]
    return (close / close.cummax() - 1).rename("drawdown")


def statistical_tests(prices: pd.DataFrame, lags: int = 10) -> pd.DataFrame:
    """Ljung-Box (serial correlation of returns) and Jarque-Bera (normality) tests.

    These describe the full sample; a rejected null does not imply a profitable strategy.
    """
    r = daily_returns(prices).dropna()
    if len(r) <= lags + 1:
        return pd.DataFrame()
    lb = acorr_ljungbox(r, lags=[lags], return_df=True)
    lb_sq = acorr_ljungbox(r ** 2, lags=[lags], return_df=True)
    jb_stat, jb_p = stats.jarque_bera(r)
    return pd.DataFrame([
        {"test": f"Ljung-Box (returns, {lags} lags)",
         "null_hypothesis": "No autocorrelation up to the lag",
         "statistic": float(lb["lb_stat"].iloc[0]), "p_value": float(lb["lb_pvalue"].iloc[0])},
        {"test": f"Ljung-Box (squared returns, {lags} lags)",
         "null_hypothesis": "No volatility clustering",
         "statistic": float(lb_sq["lb_stat"].iloc[0]), "p_value": float(lb_sq["lb_pvalue"].iloc[0])},
        {"test": "Jarque-Bera", "null_hypothesis": "Returns are normally distributed",
         "statistic": float(jb_stat), "p_value": float(jb_p)},
    ])
