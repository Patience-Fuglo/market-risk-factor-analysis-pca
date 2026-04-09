# 📉 Financial Market Risk & Factor Analysis Using Statistical Distributions and PCA

An end-to-end quantitative research pipeline spanning statistical inference, linear algebra, and machine learning — applied to 60+ years of S&P 500 price history across 10 major stocks to model return distributions, decompose market risk structure, and extract latent factors using Principal Component Analysis.

---

## 🧩 Project Overview

This project sits at the intersection of three disciplines: **statistics** (distribution analysis, hypothesis testing, confidence intervals), **linear algebra** (vectors, matrices, eigendecomposition), and **machine learning** (PCA as a factor model). It treats stock returns as random variables and uses the full toolkit of quantitative finance to interrogate whether classical statistical assumptions hold — and what the actual risk structure of the market looks like when they don't.

The dataset spans from IBM's 1962 listing through August 2023, with the analysis window restricted to the period where all 10 tickers have concurrent data (2010–2023, 3,297 trading days). A companion research report documents the findings in full.

---

## 🎯 What It Investigates

| Section | Analysis |
|---|---|
| **1. Data Pipeline** | Multi-file ingestion via `globals()`, long-to-wide pivot, missing value audit |
| **2. Random Variables** | Daily returns as RVs; simple return formula |
| **3. Distribution Analysis** | Histogram + KDE vs fitted normal curve (fat tail visualisation) |
| **4. Descriptive Statistics** | Mean, variance, skewness, excess kurtosis — full 10-asset summary |
| **5. Confidence Intervals** | 95% CI on GOOG mean return using t-distribution and SEM |
| **6. Hypothesis Testing** | One-sample t-test (H₀: μ = 0), two-sample t-test (GOOG vs MSFT A/B test) |
| **7. Why Assumptions Fail** | Volatility clustering, fat tails, non-stationarity, correlation breakdown |
| **8. Linear Algebra** | Returns as vectors and T×N matrices |
| **9. Covariance Matrix** | Full 10×10 covariance heatmap; variance decomposition |
| **10. Eigendecomposition** | `np.linalg.eig()` on the covariance matrix — eigenvalues and eigenvectors |
| **11. PCA Factor Analysis** | Standardised PCA; explained variance ratio; factor score extraction |
| **12. Feature Stability** | 60-day rolling covariance and rolling correlation |
| **13. Noise Removal** | `VarianceThreshold` filtering; 3-component PCA reconstruction |

---

## 📦 Dataset

- **Source:** S&P 500 Stock Prices (Kaggle) — individual OHLCV CSVs per ticker
- **Tickers:** AAPL, AMD, AMZN, CSCO, GOOG, IBM, MSFT, NFLX, SBUX, TSLA
- **Raw coverage:** 1962–2023 (82,880 total rows across all tickers combined)
- **Analysis window:** June 2010 – August 2023 (3,297 concurrent trading days after `dropna`)
- **Missing values before alignment:** TSLA 12,206 rows, GOOG 10,731 — all pre-IPO; IBM is complete from 1962

---

## 🗂️ Project Structure

```
market-risk-factor-analysis-pca/
│
├── Financial_Market_Risk_Factor_Analysis_Using_Statistical_Distributions_and_PCA.ipynb
├── quant_market_risk_research_report.pdf
└── README.md
```

---

## 🔧 Technical Stack

| Library | Purpose |
|---|---|
| `pandas` | CSV ingestion, pivot table, rolling statistics |
| `numpy` | Return vectors, matrix representation, eigendecomposition |
| `scipy.stats` | Distribution fitting, confidence intervals, t-tests |
| `sklearn.decomposition.PCA` | Factor extraction, explained variance, noise reduction |
| `sklearn.preprocessing.StandardScaler` | Return standardisation before PCA |
| `sklearn.feature_selection.VarianceThreshold` | Low-variance feature removal |
| `matplotlib` / `seaborn` | Distribution plots, heatmaps, scree plot |

---

## 📐 Methodology

### Data Architecture

Raw CSVs are loaded into named DataFrames via `globals()`, then concatenated into a single long-format DataFrame (82,880 rows). A `pivot_table` with `Date` as index and `Ticker` as columns converts this to a wide price matrix — the natural structure for multi-asset analysis. After `dropna()` to restrict to dates where all 10 tickers traded concurrently, the working dataset is 3,297 × 10.

### Return Computation

Daily simple returns are computed across all 10 assets simultaneously:
```python
returns = prices.pct_change().dropna()
```
The resulting T×N matrix (3,296 × 10) forms the basis for all statistical and linear algebra operations that follow.

### Distribution Analysis

Google returns are compared against a fitted normal distribution with the same mean and standard deviation. The histogram shows visible **fat tails** — extreme daily moves occur more frequently than the normal curve predicts. This is not an artefact of the data; it is a structural property of equity returns.

### Descriptive Statistics (Full Portfolio)

| Ticker | Mean Return | Variance | Skewness | Excess Kurtosis |
|---|---|---|---|---|
| AAPL | 0.001082 | 0.000318 | −0.061 | **5.32** |
| AMD | 0.001457 | 0.001282 | +1.024 | **16.91** |
| AMZN | 0.001177 | 0.000432 | +0.293 | **6.64** |
| CSCO | 0.000413 | 0.000279 | −0.419 | **15.71** |
| GOOG | 0.000887 | 0.000297 | +0.492 | **8.57** |

All 10 assets have excess kurtosis far above zero (normal distribution = 0). AMD's kurtosis of 16.91 and CSCO's 15.71 indicate that extreme daily moves in these names occur at a rate that normal models would dramatically underestimate. AMD also has the highest variance — the most volatile asset in the basket across the full 13-year window.

### Confidence Intervals

95% CI for GOOG mean daily return (t-distribution, SEM-based):
```
CI: (0.000299, 0.001475)
```
Zero is outside this interval — which is consistent with the one-sample t-test result below. This is a longer sample (3,296 days) than Project 4's one-year window, giving more statistical power to detect a non-zero mean.

### Hypothesis Testing

**One-sample t-test, GOOG (H₀: μ = 0):**
- t-statistic: 2.957, p-value: 0.003
- Reject H₀ — GOOG's mean daily return is statistically distinguishable from zero over the 13-year sample

This contrasts with Project 4's one-year window where AMZN's mean was not distinguishable from zero (p = 0.80). Sample size is the key difference: with 3,296 observations, even a mean return of 0.09% per day generates enough signal to detect.

**Two-sample t-test, GOOG vs MSFT (H₀: μ_GOOG = μ_MSFT):**
- t-statistic: −0.124, p-value: 0.901
- Fail to reject — no statistically significant difference between GOOG and MSFT mean returns across this period

### Covariance Matrix & Eigendecomposition

The 10×10 covariance matrix shows predominantly positive off-diagonal entries — all 10 technology names are positively correlated. AMD has the highest diagonal value (variance = 0.001282), over 4× higher than IBM (0.000101), reflecting the difference between a high-growth semiconductor stock and a mature enterprise IT company.

Eigendecomposition via `np.linalg.eig()`:
```
λ₁ = 0.002540  (largest — market factor)
λ₂ = 0.000899
λ₃ = 0.000815
...
λ₁₀ = 0.000111  (smallest — noise)
```
The first eigenvalue is nearly 3× the second, indicating a dominant common factor in the data. The eigenvectors associated with the largest eigenvalue show positive loadings across all stocks — this is the market factor, the direction in which all assets tend to move together.

### PCA Factor Analysis

After standardising returns with `StandardScaler`, PCA is run on the full 3,296 × 10 matrix:

**Explained variance by component:**
| PC | Variance Explained | Cumulative |
|---|---|---|
| PC1 | **46.27%** | 46.27% |
| PC2 | 10.24% | 56.51% |
| PC3 | 7.69% | 64.20% |
| PC4 | 6.91% | 71.12% |
| PC5 | 6.45% | 77.57% |

**PC1 explains 46.27% of all return variance** — this is the market factor. On any given day, roughly half of the variation in all 10 stock returns is explained by a single underlying dimension: broad market direction. The remaining components capture sector-specific and idiosyncratic influences.

The scree plot (explained variance vs component number) shows a pronounced elbow after PC1, confirming that a single dominant factor drives the market and subsequent components have diminishing explanatory power.

The 3-component PCA reconstruction (`n_components=3`) retains ~64% of total variance while discarding the remaining 36% as noise — a practical dimensionality reduction for downstream modelling.

### Feature Stability (60-Day Rolling Windows)

Rolling covariance and rolling correlation over 60-day windows (approximately 3 trading months) reveal that asset relationships are not constant. During periods of market stress, correlations increase across the basket — a well-documented empirical phenomenon where diversification breaks down precisely when it is most needed. During calmer periods, correlations fall and stocks behave more independently.

---

## 📊 Key Findings

- **Returns are fat-tailed.** All 10 assets have excess kurtosis well above zero — AMD at 16.91, CSCO at 15.71. Extreme daily moves happen far more often than normal models predict.
- **One dominant market factor exists.** PC1 explains 46.27% of all return variance. Three components together explain 64.2%. The market factor is not theoretical — it is directly measurable from the data.
- **GOOG's mean return is statistically non-zero over 13 years** (p = 0.003), but not distinguishable from MSFT's over the same period (p = 0.901).
- **Covariance is not stable.** 60-day rolling windows show significant regime changes in cross-asset relationships — any risk model calibrated on one period may be unreliable in another.
- **AMD is the highest-risk, highest-return asset** in the basket across the sample period: highest variance, highest mean return, highest kurtosis.
- **IBM provides the best diversification benefit** — lowest variance, lowest cross-asset covariance, oldest and most complete history in the dataset.

---

## 🧠 Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Random variables in finance | Returns modelled as RVs; full statistical characterisation |
| Fat-tailed distributions | Empirical vs fitted normal comparison; excess kurtosis analysis |
| Statistical moments | Mean, variance, skewness, kurtosis — full 10-asset DataFrame |
| Confidence intervals | t-distribution CI using `stats.t.interval()` and SEM |
| Hypothesis testing | One-sample and two-sample t-tests with interpretation |
| Vector & matrix representation | Returns as T×N matrix via `returns.values` |
| Covariance matrix | `returns.cov()` — full computation and heatmap |
| Eigendecomposition | `np.linalg.eig()` on covariance matrix — eigenvalues and eigenvectors |
| PCA factor analysis | `sklearn.PCA` — explained variance, factor scores, scree plot |
| Feature stability | 60-day rolling covariance and rolling correlation |
| Noise removal | `VarianceThreshold` + low-component PCA reconstruction |
| Long-to-wide data transformation | `pivot_table` for multi-ticker time series alignment |

---

## 🚀 How to Run

**Requirements:**
```
pandas numpy matplotlib seaborn scipy scikit-learn
```

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

1. Clone the repo and open the notebook in Jupyter or Google Colab
2. Mount Google Drive or update `path` to your local data directory
3. Place the 10 ticker CSVs from the Kaggle S&P 500 dataset in the data folder
4. Run all cells top to bottom

The companion research report (`quant_market_risk_research_report.pdf`) documents methodology, findings, and investment implications in full.

---

## 📄 Research Report

A written research report accompanies this notebook, covering:
- Market risk structure analysis
- Volatility and covariance regime analysis
- PCA factor interpretation
- Business and investment implications for portfolio managers
- Conclusions on statistical model limitations in financial markets

---

## 📌 Context

Capstone project of a Python for Quantitative Finance programme. Integrates all prior project themes — data pipelines, statistical testing, time series analysis — and extends into linear algebra and unsupervised machine learning. The use of PCA as a factor model reflects standard practice in quantitative asset management: APT (Arbitrage Pricing Theory), risk factor models (Barra, Axioma), and statistical arbitrage strategies all rely on eigendecomposition of the return covariance matrix to identify systematic risk exposures.
