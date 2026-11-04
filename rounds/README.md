# Rounds: season-only t0-gamma LoRA (forecast 0–26 weeks ahead)

Later rounds of the t0-gamma LoRA, trained to forecast a colour's season total
0, 4, 8, 13 or 26 weeks before the season, on every colour (new ones included).
The paper is in [`paper/t0gamma_lora_season_forecast.md`](../paper/t0gamma_lora_season_forecast.md).

`model/` at the repo root is still the model the app uses (forecast from the
first day of the season). Nothing here replaces it.

| Path | |
|---|---|
| `14T/` | **round 14T, the chosen round** — LoRA adapter (Git LFS) + config |
| `11/` | round 11: + release week, new colours in training |
| `8b/` | round 8b: season-only input |
| `scripts/t0lora_lead.py` | training and evaluation script (round 14T flags) |
| `scripts/t0gamma_bwe.py` | helpers it imports (series building, LoRA, save / load) |
| `scripts/train_round_14T.sh` | the exact environment flags of the 14T training run |
| `results/full_test_*.csv` | per-colour forecasts on the full test: 14T, 11, 8b, untrained t0-gamma (same input as 14T), LightGBM, TimesFM 2.5, Timestral |
| `score_full_test.py` | scores one results file |

## Full test (Spring + Summer 2026, 4,826 colour seasons, all leads)

| Round | All colours | Old colours | New colours |
|---|---|---|---|
| t0-gamma, no training (same input as 14T) | 106% (+23%) | 144% (+94%) | 90% (−6%) |
| 8b | 80% (−48%) | 80% (−23%) | 79% (−58%) |
| 11 | 74% (−27%) | 78% (−27%) | 73% (−27%) |
| **14T** | **72% (−25%)** | **78% (−22%)** | **70% (−27%)** |
| LightGBM (trained on 2025 seasons) | 69% (−40%) | 77% (−33%) | 66% (−43%) |
| TimesFM 2.5 (no training) | 82% (−43%) | 82% (−37%) | 83% (−45%) |
| Timestral 209M (no training) | 86% (−32%) | 85% (−32%) | 86% (−32%) |

Error = Σ|forecast − demand| ÷ Σ demand; bias in brackets. Demand = units ÷
weekly in-stock share (never below 0.3).

```bash
python rounds/score_full_test.py rounds/results/full_test_14T.csv
```

## Results files

One row per colour season and forecast lead. Stores are **A–G**; colours are
**anonymous ids** (`c00001` …, the same id across all files). No product
names, product ids or store names are included.

| Column | |
|---|---|
| `store` | A–G |
| `colour` | anonymous colour id |
| `season` | Spring 2026 / Summer 2026 |
| `lead_weeks` | weeks before the season the forecast was made |
| `season_weeks` | length of the season in weeks |
| `new_colour` | first sold in the 26 weeks before the season, or during it |
| `in_stock_share` | share of the season the colour was in stock |
| `in_test` | in stock ≥ 50% of the season (the scored set) |
| `actual_demand` | stock-corrected season demand |
| `actual_units` | units sold |
| `forecast` | the round's season-total forecast |

## Training round 14T

`scripts/train_round_14T.sh` holds the flags (`CTXL=36 PROXY=style KREAD=1.0
MASK=1 MEANLOSS=1 CLOSS=1 CLOSS_SEASON=1 CLOSSW=0.5 DROP=1 NEWCOL=1 TYPEFILL=1`,
1,500 steps). Training needs the stores' weekly data, which is not in this
repo. Store names in the scripts are replaced by `store-a` … `store-g`, and two
store-specific title words were removed from the product-type word list.
