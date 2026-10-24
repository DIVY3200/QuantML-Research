# Research run: AAPL

- Data: Yahoo Finance via yfinance (2010-01-04 to 2024-12-31, 3774 rows); retrieved 2026-10-24T14:33:43.193126+00:00
- Price adjustment: Adjusted for splits and dividends: raw Open/High/Low/Close multiplied by Adj Close / Close (applied once).
- Execution: next_open; costs: 1.0 bps fees + 2.0 bps slippage per unit turnover
- Model: logistic_regression {'C': 0.01, 'max_iter': 2000, 'class_weight': None}; threshold 0.50

## In-sample (training period) (2010-03-16 to 2019-01-29)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2010-03-16           2010-03-16               2010-03-16
End date                            2019-01-29           2019-01-29               2019-01-29
Trading days                             2,234                2,234                    2,234
Cumulative return (net)                444.96%              218.56%                  243.55%
Cumulative return (gross)              445.13%              223.75%                  288.18%
CAGR (net)                              21.08%               13.96%                   14.94%
Annualised volatility                   25.84%               18.97%                   22.26%
Sharpe ratio (net)                        0.87                 0.78                     0.74
Sharpe ratio (gross)                      0.87                 0.79                     0.80
Maximum drawdown                       -43.80%              -24.20%                  -40.16%
Calmar ratio                              0.48                 0.58                     0.37
Win rate (days in market)               52.31%               51.45%                   52.80%
Win rate (round trips)                 100.00%               51.85%                   49.51%
Position changes                             1                   54                      407
Round trips                                  1                   27                      204
Total turnover (x equity)                 1.00                54.00                   407.00
Annualised turnover (x equity)            0.11                 6.09                    45.91
Average exposure                        99.96%               64.82%                   82.99%
Total costs (sum, % of equity)           0.03%                1.62%                   12.21%
Cost drag (gross - net cumulative)       0.16%                5.19%                   44.63%
```

## Validation (used for model choices) (2019-01-30 to 2022-01-10)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2019-01-30           2019-01-30               2019-01-30
End date                            2022-01-10           2022-01-10               2022-01-10
Trading days                               744                  744                      744
Cumulative return (net)                326.99%              158.77%                  209.00%
Cumulative return (gross)              327.12%              159.79%                  224.79%
CAGR (net)                              63.50%               37.99%                   46.54%
Annualised volatility                   33.50%               26.05%                   24.25%
Sharpe ratio (net)                        1.64                 1.37                     1.70
Sharpe ratio (gross)                      1.64                 1.37                     1.77
Maximum drawdown                       -31.43%              -21.20%                  -20.05%
Calmar ratio                              2.02                 1.79                     2.32
Win rate (days in market)               55.05%               55.83%                   56.58%
Win rate (round trips)                 100.00%               85.71%                   56.63%
Position changes                             1                   13                      166
Round trips                                  1                    7                       83
Total turnover (x equity)                 1.00                13.00                   166.00
Annualised turnover (x equity)            0.34                 4.40                    56.23
Average exposure                        99.87%               76.08%                   78.63%
Total costs (sum, % of equity)           0.03%                0.39%                    4.98%
Cost drag (gross - net cumulative)       0.13%                1.02%                   15.79%
```

## Out-of-sample (untouched test period) (2022-01-11 to 2024-12-31)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2022-01-11           2022-01-11               2022-01-11
End date                            2024-12-31           2024-12-31               2024-12-31
Trading days                               747                  747                      747
Cumulative return (net)                 44.47%               15.39%                   21.65%
Cumulative return (gross)               44.52%               16.04%                   27.52%
CAGR (net)                              13.21%                4.95%                    6.83%
Annualised volatility                   27.09%               17.71%                   23.63%
Sharpe ratio (net)                        0.59                 0.36                     0.40
Sharpe ratio (gross)                      0.59                 0.37                     0.47
Maximum drawdown                       -29.82%              -28.10%                  -23.56%
Calmar ratio                              0.44                 0.18                     0.29
Win rate (days in market)               52.95%               52.05%                   50.99%
Win rate (round trips)                 100.00%               40.00%                   44.30%
Position changes                             1                   19                      157
Round trips                                  1                   10                       79
Total turnover (x equity)                 1.00                19.00                   157.00
Annualised turnover (x equity)            0.34                 6.41                    52.96
Average exposure                        99.87%               55.56%                   81.39%
Total costs (sum, % of equity)           0.03%                0.57%                    4.71%
Cost drag (gross - net cumulative)       0.04%                0.66%                    5.87%
```

## Walk-forward out-of-sample (48 folds)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2013-03-18           2013-03-18               2013-03-18
End date                            2024-12-30           2024-12-30               2024-12-30
Trading days                             2,968                2,968                    2,968
Cumulative return (net)               1694.79%              355.50%                 1056.44%
Cumulative return (gross)             1695.33%              365.30%                 1310.17%
CAGR (net)                              27.78%               13.74%                   23.10%
Annualised volatility                   27.69%               19.85%                   22.63%
Sharpe ratio (net)                        1.02                 0.75                     1.03
Sharpe ratio (gross)                      1.02                 0.76                     1.11
Maximum drawdown                       -38.52%              -28.10%                  -30.97%
Calmar ratio                              0.72                 0.49                     0.75
Win rate (days in market)               53.22%               52.22%                   52.98%
Win rate (round trips)                 100.00%               52.78%                   53.78%
Position changes                             1                   71                      661
Round trips                                  1                   36                      331
Total turnover (x equity)                 1.00                71.00                   661.00
Annualised turnover (x equity)            0.08                 6.03                    56.12
Average exposure                        99.97%               65.36%                   73.52%
Total costs (sum, % of equity)           0.03%                2.13%                   19.83%
Cost drag (gross - net cumulative)       0.54%                9.79%                  253.73%
```

## Checks
- Label purge: 1 row(s) removed before each split boundary; no training label uses prices from a later period.
- Look-ahead audit PASSED: features recomputed on truncated history at 25 dates (max abs diff 0.00e+00).
- Preprocessing (standardisation) is inside the model pipeline and fitted on training rows only.

## Warnings
- none
