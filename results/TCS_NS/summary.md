# Research run: TCS.NS

- Data: Yahoo Finance via yfinance (2010-01-04 to 2024-12-31, 3700 rows); retrieved 2026-10-24T14:33:46.221021+00:00
- Price adjustment: Adjusted for splits and dividends: raw Open/High/Low/Close multiplied by Adj Close / Close (applied once).
- Execution: next_open; costs: 1.0 bps fees + 2.0 bps slippage per unit turnover
- Model: logistic_regression {'C': 0.001, 'max_iter': 2000, 'class_weight': None}; threshold 0.50

## In-sample (training period) (2010-03-17 to 2019-01-28)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2010-03-17           2010-03-17               2010-03-17
End date                            2019-01-28           2019-01-28               2019-01-28
Trading days                             2,190                2,190                    2,190
Cumulative return (net)                449.83%               31.16%                  722.14%
Cumulative return (gross)              450.00%               33.55%                  786.34%
CAGR (net)                              21.67%                3.17%                   27.43%
Annualised volatility                   25.13%               19.32%                   21.32%
Sharpe ratio (net)                        0.91                 0.26                     1.24
Sharpe ratio (gross)                      0.91                 0.27                     1.28
Maximum drawdown                       -25.25%              -33.29%                  -23.03%
Calmar ratio                              0.86                 0.10                     1.19
Win rate (days in market)               51.80%               49.89%                   53.12%
Win rate (round trips)                 100.00%               50.00%                   72.22%
Position changes                             1                   60                      251
Round trips                                  1                   30                      126
Total turnover (x equity)                 1.00                60.00                   251.00
Annualised turnover (x equity)            0.12                 6.90                    28.88
Average exposure                        99.95%               61.69%                   75.39%
Total costs (sum, % of equity)           0.03%                1.80%                    7.53%
Cost drag (gross - net cumulative)       0.17%                2.38%                   64.20%
```

## Validation (used for model choices) (2019-01-29 to 2022-01-12)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2019-01-29           2019-01-29               2019-01-29
End date                            2022-01-12           2022-01-12               2022-01-12
Trading days                               730                  730                      730
Cumulative return (net)                107.49%               37.06%                   77.89%
Cumulative return (gross)              107.56%               37.76%                   82.71%
CAGR (net)                              28.66%               11.50%                   22.00%
Annualised volatility                   27.14%               20.15%                   23.33%
Sharpe ratio (net)                        1.06                 0.64                     0.97
Sharpe ratio (gross)                      1.07                 0.65                     1.01
Maximum drawdown                       -27.21%              -23.28%                  -26.21%
Calmar ratio                              1.05                 0.49                     0.84
Win rate (days in market)               53.64%               50.58%                   52.79%
Win rate (round trips)                 100.00%               55.56%                   80.00%
Position changes                             1                   17                       89
Round trips                                  1                    9                       45
Total turnover (x equity)                 1.00                17.00                    89.00
Annualised turnover (x equity)            0.35                 5.87                    30.72
Average exposure                        99.86%               70.41%                   73.70%
Total costs (sum, % of equity)           0.03%                0.51%                    2.67%
Cost drag (gross - net cumulative)       0.06%                0.70%                    4.82%
```

## Out-of-sample (untouched test period) (2022-01-13 to 2024-12-31)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2022-01-13           2022-01-13               2022-01-13
End date                            2024-12-31           2024-12-31               2024-12-31
Trading days                               731                  731                      731
Cumulative return (net)                 12.96%              -14.86%                   32.61%
Cumulative return (gross)               12.99%              -14.37%                   34.97%
CAGR (net)                               4.29%               -5.40%                   10.22%
Annualised volatility                   20.97%               15.70%                   19.83%
Sharpe ratio (net)                        0.30                -0.28                     0.59
Sharpe ratio (gross)                      0.31                -0.26                     0.62
Maximum drawdown                       -24.98%              -29.78%                  -22.27%
Calmar ratio                              0.17                -0.18                     0.46
Win rate (days in market)               49.45%               46.99%                   50.08%
Win rate (round trips)                 100.00%               50.00%                   66.67%
Position changes                             1                   19                       59
Round trips                                  1                   10                       30
Total turnover (x equity)                 1.00                19.00                    59.00
Annualised turnover (x equity)            0.34                 6.55                    20.34
Average exposure                        99.86%               56.77%                   87.41%
Total costs (sum, % of equity)           0.03%                0.57%                    1.77%
Cost drag (gross - net cumulative)       0.03%                0.49%                    2.36%
```

## Walk-forward out-of-sample (46 folds)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2013-04-03           2013-04-03               2013-04-03
End date                            2024-12-30           2024-12-30               2024-12-30
Trading days                             2,894                2,894                    2,894
Cumulative return (net)                589.16%               19.18%                  239.95%
Cumulative return (gross)              589.38%               21.82%                  294.63%
CAGR (net)                              18.30%                1.54%                   11.24%
Annualised volatility                   24.01%               18.29%                   19.75%
Sharpe ratio (net)                        0.82                 0.17                     0.64
Sharpe ratio (gross)                      0.82                 0.19                     0.70
Maximum drawdown                       -27.21%              -40.15%                  -26.09%
Calmar ratio                              0.67                 0.04                     0.43
Win rate (days in market)               51.75%               49.52%                   50.76%
Win rate (round trips)                 100.00%               48.65%                   55.02%
Position changes                             1                   73                      497
Round trips                                  1                   37                      249
Total turnover (x equity)                 1.00                73.00                   497.00
Annualised turnover (x equity)            0.09                 6.36                    43.28
Average exposure                        99.97%               61.40%                   72.56%
Total costs (sum, % of equity)           0.03%                2.19%                   14.91%
Cost drag (gross - net cumulative)       0.21%                2.64%                   54.68%
```

## Checks
- Label purge: 1 row(s) removed before each split boundary; no training label uses prices from a later period.
- Look-ahead audit PASSED: features recomputed on truncated history at 25 dates (max abs diff 0.00e+00).
- Preprocessing (standardisation) is inside the model pipeline and fitted on training rows only.

## Warnings
- none
