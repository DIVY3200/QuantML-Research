# Research run: RELIANCE.NS

- Data: Yahoo Finance via yfinance (2010-01-04 to 2024-12-31, 3700 rows); retrieved 2026-10-24T14:33:44.437541+00:00
- Price adjustment: Adjusted for splits and dividends: raw Open/High/Low/Close multiplied by Adj Close / Close (applied once).
- Execution: next_open; costs: 1.0 bps fees + 2.0 bps slippage per unit turnover
- Model: logistic_regression {'C': 0.001, 'max_iter': 2000, 'class_weight': None}; threshold 0.50

## In-sample (training period) (2010-03-17 to 2019-01-28)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2010-03-17           2010-03-17               2010-03-17
End date                            2019-01-28           2019-01-28               2019-01-28
Trading days                             2,190                2,190                    2,190
Cumulative return (net)                150.40%              -21.22%                  265.32%
Cumulative return (gross)              150.48%              -19.81%                  329.89%
CAGR (net)                              11.14%               -2.71%                   16.08%
Annualised volatility                   26.08%               19.47%                   20.21%
Sharpe ratio (net)                        0.54                -0.04                     0.84
Sharpe ratio (gross)                      0.54                -0.03                     0.93
Maximum drawdown                       -39.15%              -51.11%                  -36.08%
Calmar ratio                              0.28                -0.05                     0.45
Win rate (days in market)               50.43%               48.15%                   52.87%
Win rate (round trips)                 100.00%               26.67%                   57.56%
Position changes                             1                   59                      542
Round trips                                  1                   30                      271
Total turnover (x equity)                 1.00                59.00                   542.00
Annualised turnover (x equity)            0.12                 6.79                    62.37
Average exposure                        99.95%               57.95%                   56.39%
Total costs (sum, % of equity)           0.03%                1.77%                   16.26%
Cost drag (gross - net cumulative)       0.07%                1.41%                   64.57%
```

## Validation (used for model choices) (2019-01-29 to 2022-01-12)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2019-01-29           2019-01-29               2019-01-29
End date                            2022-01-12           2022-01-12               2022-01-12
Trading days                               730                  730                      730
Cumulative return (net)                112.25%               64.29%                   40.17%
Cumulative return (gross)              112.32%               65.28%                   46.05%
CAGR (net)                              29.67%               18.69%                   12.36%
Annualised volatility                   34.77%               21.38%                   28.97%
Sharpe ratio (net)                        0.92                 0.91                     0.55
Sharpe ratio (gross)                      0.92                 0.92                     0.60
Maximum drawdown                       -45.09%              -32.06%                  -47.78%
Calmar ratio                              0.66                 0.58                     0.26
Win rate (days in market)               52.40%               52.74%                   51.69%
Win rate (round trips)                 100.00%               60.00%                   52.17%
Position changes                             1                   20                      137
Round trips                                  1                   10                       69
Total turnover (x equity)                 1.00                20.00                   137.00
Annualised turnover (x equity)            0.35                 6.90                    47.29
Average exposure                        99.86%               57.40%                   48.49%
Total costs (sum, % of equity)           0.03%                0.60%                    4.11%
Cost drag (gross - net cumulative)       0.06%                0.99%                    5.88%
```

## Out-of-sample (untouched test period) (2022-01-13 to 2024-12-31)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2022-01-13           2022-01-13               2022-01-13
End date                            2024-12-31           2024-12-31               2024-12-31
Trading days                               731                  731                      731
Cumulative return (net)                  4.92%                1.18%                   30.98%
Cumulative return (gross)                4.96%                1.73%                   38.70%
CAGR (net)                               1.67%                0.41%                    9.75%
Annualised volatility                   22.57%               15.42%                   17.71%
Sharpe ratio (net)                        0.19                 0.10                     0.61
Sharpe ratio (gross)                      0.19                 0.11                     0.73
Maximum drawdown                       -24.46%              -24.41%                  -16.59%
Calmar ratio                              0.07                 0.02                     0.59
Win rate (days in market)               50.68%               49.32%                   52.63%
Win rate (round trips)                 100.00%               22.22%                   61.46%
Position changes                             1                   18                      191
Round trips                                  1                    9                       96
Total turnover (x equity)                 1.00                18.00                   191.00
Annualised turnover (x equity)            0.34                 6.21                    65.84
Average exposure                        99.86%               49.93%                   54.58%
Total costs (sum, % of equity)           0.03%                0.54%                    5.73%
Cost drag (gross - net cumulative)       0.03%                0.55%                    7.72%
```

## Walk-forward out-of-sample (46 folds)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2013-04-03           2013-04-03               2013-04-03
End date                            2024-12-30           2024-12-30               2024-12-30
Trading days                             2,894                2,894                    2,894
Cumulative return (net)                645.35%               68.57%                  147.53%
Cumulative return (gross)              645.57%               72.56%                  212.79%
CAGR (net)                              19.11%                4.65%                    8.21%
Annualised volatility                   27.52%               19.40%                   20.22%
Sharpe ratio (net)                        0.77                 0.33                     0.49
Sharpe ratio (gross)                      0.77                 0.34                     0.59
Maximum drawdown                       -45.09%              -43.74%                  -54.05%
Calmar ratio                              0.42                 0.11                     0.15
Win rate (days in market)               51.81%               50.15%                   50.87%
Win rate (round trips)                 100.00%               38.46%                   51.03%
Position changes                             1                   78                      780
Round trips                                  1                   39                      390
Total turnover (x equity)                 1.00                78.00                   780.00
Annualised turnover (x equity)            0.09                 6.79                    67.92
Average exposure                        99.97%               58.36%                   45.58%
Total costs (sum, % of equity)           0.03%                2.34%                   23.40%
Cost drag (gross - net cumulative)       0.23%                3.99%                   65.26%
```

## Checks
- Label purge: 1 row(s) removed before each split boundary; no training label uses prices from a later period.
- Look-ahead audit PASSED: features recomputed on truncated history at 25 dates (max abs diff 0.00e+00).
- Preprocessing (standardisation) is inside the model pipeline and fitted on training rows only.

## Warnings
- none
