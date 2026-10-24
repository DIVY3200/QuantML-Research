# Research run: MSFT

- Data: Yahoo Finance via yfinance (2010-01-04 to 2024-12-31, 3774 rows); retrieved 2026-10-24T14:33:43.867840+00:00
- Price adjustment: Adjusted for splits and dividends: raw Open/High/Low/Close multiplied by Adj Close / Close (applied once).
- Execution: next_open; costs: 1.0 bps fees + 2.0 bps slippage per unit turnover
- Model: logistic_regression {'C': 0.001, 'max_iter': 2000, 'class_weight': None}; threshold 0.50

## In-sample (training period) (2010-03-16 to 2019-01-29)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2010-03-16           2010-03-16               2010-03-16
End date                            2019-01-29           2019-01-29               2019-01-29
Trading days                             2,234                2,234                    2,234
Cumulative return (net)                334.71%              116.30%                  696.39%
Cumulative return (gross)              334.84%              119.43%                  806.24%
CAGR (net)                              18.03%                9.09%                   26.37%
Annualised volatility                   23.14%               17.83%                   20.24%
Sharpe ratio (net)                        0.83                 0.58                     1.26
Sharpe ratio (gross)                      0.83                 0.59                     1.33
Maximum drawdown                       -26.37%              -27.25%                  -25.38%
Calmar ratio                              0.68                 0.33                     1.04
Win rate (days in market)               51.32%               51.28%                   52.04%
Win rate (round trips)                 100.00%               50.00%                   70.83%
Position changes                             1                   48                      431
Round trips                                  1                   24                      216
Total turnover (x equity)                 1.00                48.00                   431.00
Annualised turnover (x equity)            0.11                 5.41                    48.62
Average exposure                        99.96%               66.34%                   78.87%
Total costs (sum, % of equity)           0.03%                1.44%                   12.93%
Cost drag (gross - net cumulative)       0.13%                3.13%                  109.85%
```

## Validation (used for model choices) (2019-01-30 to 2022-01-10)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2019-01-30           2019-01-30               2019-01-30
End date                            2022-01-10           2022-01-10               2022-01-10
Trading days                               744                  744                      744
Cumulative return (net)                213.01%               88.81%                  231.67%
Cumulative return (gross)              213.10%               89.84%                  249.11%
CAGR (net)                              47.18%               24.02%                   50.10%
Annualised volatility                   30.27%               26.38%                   24.38%
Sharpe ratio (net)                        1.43                 0.95                     1.79
Sharpe ratio (gross)                      1.43                 0.96                     1.86
Maximum drawdown                       -28.04%              -27.49%                  -14.44%
Calmar ratio                              1.68                 0.87                     3.47
Win rate (days in market)               55.99%               55.80%                   55.41%
Win rate (round trips)                 100.00%               66.67%                   77.91%
Position changes                             1                   18                      171
Round trips                                  1                    9                       86
Total turnover (x equity)                 1.00                18.00                   171.00
Annualised turnover (x equity)            0.34                 6.10                    57.92
Average exposure                        99.87%               85.75%                   69.62%
Total costs (sum, % of equity)           0.03%                0.54%                    5.13%
Cost drag (gross - net cumulative)       0.09%                1.03%                   17.44%
```

## Out-of-sample (untouched test period) (2022-01-11 to 2024-12-31)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2022-01-11           2022-01-11               2022-01-11
End date                            2024-12-31           2024-12-31               2024-12-31
Trading days                               747                  747                      747
Cumulative return (net)                 35.23%               -6.33%                   21.68%
Cumulative return (gross)               35.27%               -5.85%                   28.32%
CAGR (net)                              10.72%               -2.18%                    6.84%
Annualised volatility                   27.55%               18.21%                   23.79%
Sharpe ratio (net)                        0.51                -0.03                     0.40
Sharpe ratio (gross)                      0.51                -0.02                     0.47
Maximum drawdown                       -32.56%              -34.38%                  -30.73%
Calmar ratio                              0.33                -0.06                     0.22
Win rate (days in market)               51.61%               49.64%                   51.82%
Win rate (round trips)                 100.00%               33.33%                   56.18%
Position changes                             1                   17                      177
Round trips                                  1                    9                       89
Total turnover (x equity)                 1.00                17.00                   177.00
Annualised turnover (x equity)            0.34                 5.73                    59.71
Average exposure                        99.87%               56.09%                   73.63%
Total costs (sum, % of equity)           0.03%                0.51%                    5.31%
Cost drag (gross - net cumulative)       0.04%                0.48%                    6.64%
```

## Walk-forward out-of-sample (48 folds)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2013-03-18           2013-03-18               2013-03-18
End date                            2024-12-30           2024-12-30               2024-12-30
Trading days                             2,968                2,968                    2,968
Cumulative return (net)               1746.42%              320.99%                  524.40%
Cumulative return (gross)             1746.97%              329.02%                  739.51%
CAGR (net)                              28.09%               12.98%                   16.83%
Annualised volatility                   26.52%               20.88%                   20.40%
Sharpe ratio (net)                        1.07                 0.69                     0.86
Sharpe ratio (gross)                      1.07                 0.70                     0.99
Maximum drawdown                       -37.15%              -39.94%                  -39.02%
Calmar ratio                              0.76                 0.33                     0.43
Win rate (days in market)               53.35%               52.93%                   52.33%
Win rate (round trips)                 100.00%               56.25%                   56.28%
Position changes                             1                   63                      987
Round trips                                  1                   32                      494
Total turnover (x equity)                 1.00                63.00                   987.00
Annualised turnover (x equity)            0.08                 5.35                    83.80
Average exposure                        99.97%               71.87%                   64.45%
Total costs (sum, % of equity)           0.03%                1.89%                   29.61%
Cost drag (gross - net cumulative)       0.55%                8.03%                  215.11%
```

## Checks
- Label purge: 1 row(s) removed before each split boundary; no training label uses prices from a later period.
- Look-ahead audit PASSED: features recomputed on truncated history at 25 dates (max abs diff 0.00e+00).
- Preprocessing (standardisation) is inside the model pipeline and fitted on training rows only.

## Warnings
- none
