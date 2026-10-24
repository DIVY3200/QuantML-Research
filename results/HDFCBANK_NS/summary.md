# Research run: HDFCBANK.NS

- Data: Yahoo Finance via yfinance (2010-01-04 to 2024-12-31, 3700 rows); retrieved 2026-10-24T14:33:44.845089+00:00
- Price adjustment: Adjusted for splits and dividends: raw Open/High/Low/Close multiplied by Adj Close / Close (applied once).
- Execution: next_open; costs: 1.0 bps fees + 2.0 bps slippage per unit turnover
- Model: logistic_regression {'C': 0.001, 'max_iter': 2000, 'class_weight': None}; threshold 0.50

## In-sample (training period) (2010-03-17 to 2019-01-28)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2010-03-17           2010-03-17               2010-03-17
End date                            2019-01-28           2019-01-28               2019-01-28
Trading days                             2,190                2,190                    2,190
Cumulative return (net)                514.36%              240.22%                  503.61%
Cumulative return (gross)              514.54%              244.43%                  566.26%
CAGR (net)                              23.23%               15.13%                   22.98%
Annualised volatility                   21.02%               15.43%                   19.76%
Sharpe ratio (net)                        1.10                 0.99                     1.15
Sharpe ratio (gross)                      1.10                 1.00                     1.20
Maximum drawdown                       -21.87%              -22.98%                  -22.35%
Calmar ratio                              1.06                 0.66                     1.03
Win rate (days in market)               51.30%               51.93%                   51.77%
Win rate (round trips)                 100.00%               66.67%                   60.00%
Position changes                             1                   41                      329
Round trips                                  1                   21                      165
Total turnover (x equity)                 1.00                41.00                   329.00
Annualised turnover (x equity)            0.12                 4.72                    37.86
Average exposure                        99.95%               67.44%                   85.02%
Total costs (sum, % of equity)           0.03%                1.23%                    9.87%
Cost drag (gross - net cumulative)       0.19%                4.20%                   62.65%
```

## Validation (used for model choices) (2019-01-29 to 2022-01-12)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2019-01-29           2019-01-29               2019-01-29
End date                            2022-01-12           2022-01-12               2022-01-12
Trading days                               730                  730                      730
Cumulative return (net)                 53.23%               39.48%                   38.21%
Cumulative return (gross)               53.27%               40.15%                   42.72%
CAGR (net)                              15.87%               12.17%                   11.82%
Annualised volatility                   30.08%               18.65%                   23.80%
Sharpe ratio (net)                        0.64                 0.71                     0.59
Sharpe ratio (gross)                      0.64                 0.72                     0.64
Maximum drawdown                       -41.05%              -16.10%                  -39.55%
Calmar ratio                              0.39                 0.76                     0.30
Win rate (days in market)               51.71%               52.11%                   51.42%
Win rate (round trips)                 100.00%               50.00%                   68.52%
Position changes                             1                   16                      107
Round trips                                  1                    8                       54
Total turnover (x equity)                 1.00                16.00                   107.00
Annualised turnover (x equity)            0.35                 5.52                    36.94
Average exposure                        99.86%               64.93%                   72.19%
Total costs (sum, % of equity)           0.03%                0.48%                    3.21%
Cost drag (gross - net cumulative)       0.05%                0.67%                    4.51%
```

## Out-of-sample (untouched test period) (2022-01-13 to 2024-12-31)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2022-01-13           2022-01-13               2022-01-13
End date                            2024-12-31           2024-12-31               2024-12-31
Trading days                               731                  731                      731
Cumulative return (net)                 20.15%              -19.93%                   38.32%
Cumulative return (gross)               20.18%              -19.32%                   41.64%
CAGR (net)                               6.53%               -7.37%                   11.83%
Annualised volatility                   21.87%               16.77%                   20.04%
Sharpe ratio (net)                        0.40                -0.37                     0.66
Sharpe ratio (gross)                      0.40                -0.36                     0.70
Maximum drawdown                       -21.76%              -34.91%                  -20.77%
Calmar ratio                              0.30                -0.21                     0.57
Win rate (days in market)               52.74%               49.89%                   52.15%
Win rate (round trips)                 100.00%               30.77%                   50.00%
Position changes                             1                   25                       79
Round trips                                  1                   13                       40
Total turnover (x equity)                 1.00                25.00                    79.00
Annualised turnover (x equity)            0.34                 8.62                    27.23
Average exposure                        99.86%               59.51%                   89.19%
Total costs (sum, % of equity)           0.03%                0.75%                    2.37%
Cost drag (gross - net cumulative)       0.04%                0.60%                    3.32%
```

## Walk-forward out-of-sample (46 folds)

```
                                    Buy & Hold MA Crossover (20/50) ML (Logistic Regression)
Start date                          2013-04-03           2013-04-03               2013-04-03
End date                            2024-12-30           2024-12-30               2024-12-30
Trading days                             2,894                2,894                    2,894
Cumulative return (net)                531.49%              189.73%                  223.07%
Cumulative return (gross)              531.68%              195.43%                  309.83%
CAGR (net)                              17.41%                9.71%                   10.75%
Annualised volatility                   23.22%               16.38%                   19.12%
Sharpe ratio (net)                        0.81                 0.65                     0.63
Sharpe ratio (gross)                      0.81                 0.66                     0.74
Maximum drawdown                       -41.05%              -42.17%                  -37.39%
Calmar ratio                              0.42                 0.23                     0.29
Win rate (days in market)               51.75%               51.55%                   51.01%
Win rate (round trips)                 100.00%               48.48%                   57.43%
Position changes                             1                   65                      793
Round trips                                  1                   33                      397
Total turnover (x equity)                 1.00                65.00                   793.00
Annualised turnover (x equity)            0.09                 5.66                    69.05
Average exposure                        99.97%               65.96%                   70.18%
Total costs (sum, % of equity)           0.03%                1.95%                   23.79%
Cost drag (gross - net cumulative)       0.19%                5.70%                   86.76%
```

## Checks
- Label purge: 1 row(s) removed before each split boundary; no training label uses prices from a later period.
- Look-ahead audit PASSED: features recomputed on truncated history at 25 dates (max abs diff 0.00e+00).
- Preprocessing (standardisation) is inside the model pipeline and fitted on training rows only.

## Warnings
- none
