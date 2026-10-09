# MISY331: NFT Price Prediction

A reproducible extension of MISY331 Assignment 5 comparing basic machine learning and PyTorch neural networks on Crypto Coven NFT sale amounts.

**Result:** none of the learned models beat the training-window mean baseline on the later-sale test window. Random forest performs best among learned models; the two-hidden-layer MLP performs worse than the one-hidden-layer MLP. These are results from one historical snapshot and one chronological holdout.

[Executed notebook](notebooks/01_model_comparison.ipynb) · [HTML report](notebooks/01_model_comparison.html) · [Experiment record](results/experiment.json) · [Data provenance](data/README.md)

![Model comparison](results/model_comparison.png)

## 中文说明

这个项目从 MISY331 的 NFT 价格预测课程作业出发，比较线性模型、树模型和基础神经网络。原作业使用随机划分；本次扩展按成交时间划分训练、验证和测试窗口，排除持有人信息、累计交易次数等可能混入快照时点信息的字段，保留 32 个 NFT 属性。

实际结果中，均值基准的测试 RMSLE 为 **0.2590**，随机森林为 **0.3293**，浅层 MLP 为 **0.3781**，深层 MLP 为 **0.4140**。训练、验证、测试窗口的成交价格水平差异很大；神经网络训练误差继续下降，验证误差却没有持续改善。这提示属性与价格的关系难以跨时间窗口泛化，但单个项目不能证明“深度学习一定不适合金融”，也不能单独识别信噪比、过拟合和市场变化各自造成了多少误差。

仓库保留真实结果、候选参数、随机种子和学习曲线。它是课程作业的后续实验，不代表原作业已经包含这些扩展，也不构成收益预测或交易回测。

## Question and data

Can static NFT attributes predict a later group of sale amounts, and does a small neural network improve on simpler estimators?

The dataset is Harry Wang's [Crypto Coven dataset](https://www.kaggle.com/datasets/harrywang/crypto-coven), collected on **21 April 2022**. The author's [public mirror](https://huggingface.co/datasets/harrywang/crypto-coven) declares the MIT license. The included data contain 9,764 rows and 35 columns, stripped to traits, token ID, sale amount, and sale time. After dropping 4,868 rows without a sale amount and deduplicating one priced token, 4,895 unique sold NFTs remain.

The response is `last_sale.total_price / 10**18`, in ETH-equivalent units. The expanded source export was checked: priced sales use ETH or WETH with 18 decimals. We avoid snapshot-dependent USD conversions. Only the six numeric and 26 categorical traits in `src/data.py` are predictors; token ID is used for deduplication and timestamp only for the split.

| Window (UTC) | Rows | Date range | Mean ETH | Median ETH |
|---|---:|---|---:|---:|
| train | 2,901 | 2021-11-07 to 2022-01-22 | 0.582 | 0.279 |
| validation | 998 | 2022-01-23 to 2022-02-09 | 2.740 | 2.500 |
| test | 996 | 2022-02-10 to 2022-04-21 | 1.400 | 1.200 |

Split boundaries approximate 60/20/20 by sorted records and keep each UTC calendar day together. These irregular observations have different window lengths. Each NFT appears once. This is a latest-sale snapshot, so chronological separation **does not eliminate snapshot selection bias** or recreate a transaction-by-transaction investment setting.

## Models and evaluation

| Method | Basic idea | Small validation search |
|---|---|---|
| Mean / median | Constant prediction from fitting-window raw ETH prices | No tuning |
| Linear regression | Ordinary least squares on log sale amount | No tuning |
| Ridge | L2-regularized linear regression | alpha: 0.1, 1, 10, 100 |
| SGD regression | L2 linear regression fitted by stochastic gradients | alpha: 0.0001, 0.001 |
| Random forest | Averaged regression trees | (depth, leaf): (8, 2), (16, 2), (unlimited, 5) |
| Gradient boosting | Histogram-based boosted trees | (leaves, L2): (7, 1), (15, 1), (15, 10) |
| Shallow MLP | One hidden layer with ReLU | 32 units/no dropout; 64 units/0.1 dropout |
| Deep MLP | Two hidden layers with ReLU | 64 -> 32 units; dropout 0 or 0.1 |

1. Fit numeric median imputation and standardization on the fitting window. Fill missing categories with an explicit absence label, one-hot encode, pool categories occurring fewer than five times, and ignore unseen categories. Some missing traits may be unknown rather than absent; the dataset does not distinguish them reliably.
2. Fit learned estimators on `log1p(price_eth)` with squared error. Neural targets are additionally standardized using fitting-window statistics. Invert using `expm1` and floor negative predictions at zero. Inverting a log-space prediction does not produce an unbiased estimate of the raw-price conditional mean.
3. Choose candidates using **earlier validation RMSLE**, with selection seed 42. MLPs use Adam (learning rate 0.001, weight decay 0.0001), batch size 128, at most 200 epochs and patience 20. Their epoch is selected by validation log MSE before the candidate is compared by clipped RMSLE. Forests use 160 trees; boosting uses 160 iterations at learning rate 0.05. Search budgets are small and differ by model; this is a practical baseline comparison rather than exhaustive tuning.
4. Refit each selected configuration on training plus validation. Freeze the MLP epoch count before final fitting. Report the later test window without using it for candidate selection or early stopping. Constant baselines use the same combined fitting window in raw ETH units.
5. Run stochastic estimators with seeds **42, 7, 2026**; deterministic estimators and baselines run once. Report RMSLE, ETH RMSE, ETH MAE, and R2. Seed spread measures optimization variability, **not** sampling uncertainty or a confidence interval. Boosting has identical outcomes across these seeds under these settings.

## Held-out results

| Model | Runs | RMSLE | RMSE (ETH) | MAE (ETH) | R2 |
|---|---:|---:|---:|---:|---:|
| Mean baseline | 1 | 0.2590 | 0.7651 | 0.4520 | -0.137 |
| Random forest | 3 | 0.3293 | 0.8948 | 0.5891 | -0.556 |
| Gradient boosting | 3 | 0.3311 | 0.8955 | 0.5854 | -0.558 |
| Ridge | 1 | 0.3368 | 0.9004 | 0.5969 | -0.575 |
| SGD regression | 3 | 0.3537 | 0.9185 | 0.6242 | -0.640 |
| Linear regression | 1 | 0.3688 | 0.9345 | 0.6480 | -0.697 |
| Shallow MLP | 3 | 0.3781 | 0.9461 | 0.6640 | -0.739 |
| Deep MLP | 3 | 0.4140 | 0.9992 | 0.7191 | -0.940 |
| Median baseline | 1 | 0.4725 | 1.1126 | 0.8527 | -1.405 |

Exact per-run results are in [test_metrics.csv](results/test_metrics.csv), candidate results in [validation_candidates.csv](results/validation_candidates.csv), and training histories in [learning_curves.csv](results/learning_curves.csv). All learned models also have negative test R2. R2 compares with the observed test-window mean (an evaluation reference unavailable to a deployed model), whereas the reported mean baseline is estimated only from earlier fitting data.

![MLP learning curves](results/mlp_learning_curves.png)

The training-window median is 0.279 ETH; the validation median is 2.500 ETH and the test median is 1.200 ETH. This observed price distribution shift makes static-trait extrapolation difficult. The MLP curves show a growing training/validation gap, consistent with weak generalization. They cannot separate memorization from temporal distribution shift. On the validation window, the training-only mean baseline also achieves RMSLE 0.8615, better than the best learned candidate, Deep MLP at 0.8921. The notebook computes that comparison directly.

The deeper MLP being best among learned validation candidates and worse on the later test window shows why architecture choice alone is insufficient. No claim of profitable signals, causal trait effects, universal tree-model superiority, or measured financial signal-to-noise follows from this experiment. Testing multiple temporal windows and collecting genuine transaction history would be sensible subsequent research; they are outside this baseline project.

## Reproduce

Use **Python 3.12** on a 64-bit Windows or Linux machine. The saved run used Python 3.12.14 and CPU PyTorch; exact package versions are pinned in `requirements.txt` and recorded in `results/experiment.json`.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.train
python -m pytest -q --basetemp .test-tmp-local
```

The data needed for the default command are included. The dependency file uses PyTorch's CPU wheel index; these pins target the tested Windows/Linux CPU environment. Package availability and numerical precision may differ on other platforms.

Open `notebooks/01_model_comparison.ipynb` in Jupyter or VS Code. The notebook loads the saved experiment results by default and executes the data audit and interpretation; set `RUN_TRAINING = True` to regenerate all results. The checked-in notebook includes executed outputs and an HTML export. The Python module is the single training implementation; the notebook does not duplicate it.

To fetch the unmodified, pinned public source instead:

```bash
python download_data.py
python -m src.train --data data/witches.csv
```

## Repository contents

```text
data/          Curated traits, attribution, source revision
src/           Data preparation, model definitions, training, figures
notebooks/     Executed report notebook and HTML export
results/       Aggregate metrics, chosen parameters, learning curves, figures
tests/         Units/deduplication, temporal split, preprocessing isolation,
               metric units, seeded neural fitting
```

The original assignment materials, raw owner fields, local environment and token-level prediction export are excluded. Code is licensed under MIT; dataset attribution and its author's license declaration are documented separately in `data/README.md`.
