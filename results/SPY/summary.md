# Research run: SPY

- Data: Yahoo Finance via yfinance (2010-01-04 to 2024-12-31, 3774 rows); retrieved 2026-10-20T12:39:57.045761+00:00
- Price adjustment: Adjusted for splits and dividends: raw Open/High/Low/Close multiplied by Adj Close / Close (applied once).
- Execution: next_open; costs: 1.0 bps fees + 2.0 bps slippage per unit turnover
- Model: logistic_regression {'C': 0.001, 'max_iter': 2000, 'class_weight': None}; threshold 0.50

## In-sample (training period) (2010-03-16 to 2019-01-29)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2010-03-16           2010-03-16               2010-03-16
End date                            2019-01-29           2019-01-29               2019-01-29
Trading days                             2,234                2,234                    2,234
Cumulative return (net)                170.34%               70.34%                  171.89%
Cumulative return (gross)              170.42%               72.81%                  172.30%
CAGR (net)                              11.87%                6.19%                   11.94%
Annualised volatility                   14.94%               10.81%                   14.92%
Sharpe ratio (net)                        0.83                 0.61                     0.83
Sharpe ratio (gross)                      0.83                 0.63                     0.83
Maximum drawdown                       -19.35%              -23.74%                  -19.35%
Calmar ratio                              0.61                 0.26                     0.62
Win rate (days in market)               54.99%               54.28%                   55.04%
Win rate (round trips)                 100.00%               58.33%                   66.67%
Position changes                             1                   48                        5
Round trips                                  1                   24                        3
Total turnover (x equity)                 1.00                48.00                     5.00
Annualised turnover (x equity)            0.11                 5.41                     0.56
Average exposure                        99.96%               71.08%                   99.87%
Total costs (sum, % of equity)           0.03%                1.44%                    0.15%
Cost drag (gross - net cumulative)       0.08%                2.47%                    0.41%
```

## Validation (used for model choices) (2019-01-30 to 2022-01-10)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2019-01-30           2019-01-30               2019-01-30
End date                            2022-01-10           2022-01-10               2022-01-10
Trading days                               744                  744                      744
Cumulative return (net)                 82.98%               62.66%                   82.23%
Cumulative return (gross)               83.03%               63.20%                   82.39%
CAGR (net)                              22.71%               17.91%                   22.54%
Annualised volatility                   21.92%               14.02%                   21.93%
Sharpe ratio (net)                        1.04                 1.25                     1.04
Sharpe ratio (gross)                      1.04                 1.25                     1.04
Maximum drawdown                       -33.72%              -12.44%                  -33.72%
Calmar ratio                              0.67                 1.44                     0.67
Win rate (days in market)               57.60%               57.67%                   57.68%
Win rate (round trips)                 100.00%               83.33%                  100.00%
Position changes                             1                   11                        3
Round trips                                  1                    6                        2
Total turnover (x equity)                 1.00                11.00                     3.00
Annualised turnover (x equity)            0.34                 3.73                     1.02
Average exposure                        99.87%               84.14%                   99.73%
Total costs (sum, % of equity)           0.03%                0.33%                    0.09%
Cost drag (gross - net cumulative)       0.05%                0.54%                    0.16%
```

## Out-of-sample (untouched test period) (2022-01-11 to 2024-12-31)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2022-01-11           2022-01-11               2022-01-11
End date                            2024-12-31           2024-12-31               2024-12-31
Trading days                               747                  747                      747
Cumulative return (net)                 29.82%               -1.62%                   29.82%
Cumulative return (gross)               29.86%               -1.12%                   29.86%
CAGR (net)                               9.20%               -0.55%                    9.20%
Annualised volatility                   17.52%               11.68%                   17.52%
Sharpe ratio (net)                        0.59                 0.01                     0.59
Sharpe ratio (gross)                      0.59                 0.03                     0.59
Maximum drawdown                       -23.54%              -27.11%                  -23.54%
Calmar ratio                              0.39                -0.02                     0.39
Win rate (days in market)               52.82%               52.35%                   52.82%
Win rate (round trips)                 100.00%               44.44%                  100.00%
Position changes                             1                   17                        1
Round trips                                  1                    9                        1
Total turnover (x equity)                 1.00                17.00                     1.00
Annualised turnover (x equity)            0.34                 5.73                     0.34
Average exposure                        99.87%               62.65%                   99.87%
Total costs (sum, % of equity)           0.03%                0.51%                    0.03%
Cost drag (gross - net cumulative)       0.04%                0.50%                    0.04%
```

## Walk-forward out-of-sample (48 folds)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2013-03-18           2013-03-18               2013-03-18
End date                            2024-12-30           2024-12-30               2024-12-30
Trading days                             2,968                2,968                    2,968
Cumulative return (net)                366.70%              139.42%                  295.58%
Cumulative return (gross)              366.84%              143.70%                  330.61%
CAGR (net)                              13.97%                7.69%                   12.39%
Annualised volatility                   16.80%               11.31%                   15.98%
Sharpe ratio (net)                        0.86                 0.71                     0.81
Sharpe ratio (gross)                      0.86                 0.73                     0.86
Maximum drawdown                       -33.72%              -28.02%                  -33.03%
Calmar ratio                              0.41                 0.27                     0.38
Win rate (days in market)               55.17%               54.84%                   55.37%
Win rate (round trips)                 100.00%               63.33%                   70.42%
Position changes                             1                   59                      283
Round trips                                  1                   30                      142
Total turnover (x equity)                 1.00                59.00                   283.00
Annualised turnover (x equity)            0.08                 5.01                    24.03
Average exposure                        99.97%               73.05%                   93.53%
Total costs (sum, % of equity)           0.03%                1.77%                    8.49%
Cost drag (gross - net cumulative)       0.14%                4.27%                   35.03%
```

## Checks
- Label purge: 1 row(s) removed before each split boundary; no training label uses prices from a later period.
- Look-ahead audit PASSED: features recomputed on truncated history at 25 dates (max abs diff 0.00e+00).
- Preprocessing (standardisation) is inside the model pipeline and fitted on training rows only.

## Warnings
- none
