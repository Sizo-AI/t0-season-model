# t0-season-model

Season demand forecast for a fashion product colour: a LoRA adapter on top of
[t0-gamma](https://huggingface.co/theforecastingcompany/t0-gamma) (The Forecasting Company, Apache-2.0).
It forecasts weekly units (all sizes together) for the weeks of a season; the season forecast is their sum.

## What is in each folder

```
t0-season-model/
├── model/                  the model the app uses today (forecast on the first day of the season)
│   ├── lora_adapter.pt     its weights: LoRA matrices, 11.0M parameters, 44 MB (Git LFS)
│   └── config.json         its settings: inputs, readout, rank, base model
├── t0_season_lora.py       code to load model/ and forecast a season: load(), forecast_season()
├── rounds/                 newer rounds that forecast 0–26 weeks before the season
│   ├── 14T/                ★ round 14T, the best round: weights (lora_adapter.pt, Git LFS) + config.json
│   ├── 11/                 round 11: + release week, new colours in training (weights + config)
│   ├── 8b/                 round 8b: first season-only round (weights + config)
│   ├── results/            per-colour forecasts of 14T, 11, 8b and untrained t0-gamma on the full test
│   │                       (stores A–G, anonymous colour ids, no product names)
│   ├── scripts/            training script, its helpers, and the exact 14T training flags
│   ├── score_full_test.py  re-scores any results file: error, bias, by lead and by store
│   └── README.md           details of the rounds and the results columns
└── paper/                  the research paper (a living document, updated as tests continue)
```

**Where are the weights?** Each model folder holds one `lora_adapter.pt` (the
trained add-on) and one `config.json`. They are stored with Git LFS, so run
`git lfs pull` after cloning. The base t0-gamma model is not stored here; it
downloads from Hugging Face when you load an adapter.

## Two models, two tests

| | `model/` (used by the app today) | `rounds/14T/` (newest) |
|---|---|---|
| Forecasts | on the first day of the season | 0, 4, 8, 13 or 26 weeks before the season |
| History it sees | up to 104 weeks, 9 input groups | last year's same season + the 8 weeks before the forecast date, + release week |
| New colours | not covered | covered (stand-in from siblings / category) |
| Tested on | 136 old good sellers, 2026 | every colour in stock ≥ half the season: 4,826 colour seasons, Spring + Summer 2026 |
| Error | 41.9% (−4%) | 72% (−25%); untrained t0-gamma 108% (+23%) |

The two errors can't be compared: the full test is far harder (56% new colours,
forecasts up to half a year ahead). On old good sellers round 14T scores 50%.
Details and all rounds: [`rounds/README.md`](rounds/README.md) and the
[paper](paper/t0gamma_lora_season_forecast.md).

## Use

```bash
pip install torch==2.8.0 "tfc-t0[torch]==0.6.0" numpy
git lfs pull
```

```python
import t0_season_lora as M
model, cfg = M.load("model")                          # base t0-gamma from the hub + this adapter
total = M.forecast_season(model, cfg, history, horizon=13)
```

`history` holds the product colour's weekly arrays (see the docstring in `t0_season_lora.py`). Run on CPU to
reproduce the numbers below.

## `model/` inputs (9 groups, chosen by backward elimination)

| Group | Kind | Over the season |
|---|---|---|
| sold-out flag (in-stock share < 50%) | known ahead | 0 = in stock |
| key-sizes-out flag | known ahead | 0 = in stock |
| week of year (sin / cos) | known ahead | calendar |
| product age (weeks since first sale) | known ahead | keeps counting |
| stock on hand | history only | — |
| restocks | history only | — |
| sibling colours' units | history only | — |
| discount share + depth | history only | — |
| category units | history only | — |

Stock-out rows carry real values in training and "fully in stock" at forecast time, so the output is demand,
not sales capped by stock. Removed by the elimination: the in-stock share (0–1) and a holiday/sale-window flag.

## `model/` training

LoRA rank 32 (alpha 64) on the 96 linear layers of t0-gamma's 24 transformer layers; base weights frozen.
Loss = t0's weekly pinball over 21 quantile levels + a season-total pinball term. 375 steps, batch 32,
lr 3e-4, seed 0, on ~51k windows (≤ 104 weeks of history + 13 weeks ahead) whose targets end before
2026-01-01. Readout = mean of the 21 quantiles × 0.9 (factor chosen on the 2025 seasons).

## `model/` results (season total vs stock-out-adjusted demand; error = Σ|f − a| ÷ Σa)

The input set was chosen on the 2025 seasons with a model trained only on data before 2025 (39.6%); the final
model was then retrained through 2025 and scored once on 136 seasons of 2026:

| Model | 2026 error | bias |
|---|---|---|
| **this model** | **41.9%** | −4% |
| t0-gamma, no training | 45.9% | −23% |
| TimesFM 3.0 | 45.8% | −14% |
| Timestral-209M | 47.2% | −15% |
| LightGBM | 49.2% | −21% |

Private research artifact of Sizo AI.
