"""Load the t0-gamma + LoRA season-demand model and forecast one product colour's season.

    import numpy as np, t0_season_lora as M
    model, cfg = M.load("model")
    total = M.forecast_season(model, cfg, history, horizon=13)

`history` is a dict of equal-length weekly arrays (oldest first, last entry = the week before the forecast starts):
  units        units sold per week (all sizes)                                        — required
  soldout      1 if the colour's demand-weighted in-stock share < 0.5, else 0          — required
  keyout       1 if the sizes making up the top half of demand are < 20% in stock      — required
  stock        units on hand at week end                                               — required
  restock      units added by restocks that week                                       — required
  sib          weekly units of the other colours of the same model                     — required (zeros if none)
  disc_share   share of units sold on discount                                         — required
  disc_depth   Σ discount ÷ Σ list price (0..1)                                        — required
  cat          weekly units of the other colours of the same product type              — required
and calendar arrays covering history + horizon:
  woy          ISO week number of every week (history and season)
  age          weeks since the colour's first sale for every week (history and season)
Over the season, the stock-out rows are set to "fully in stock" (soldout = 0, keyout = 0), so the output is demand.

Requires torch (tested 2.8.0, CPU) and tfc-t0==0.6.0. The base t0-gamma weights are downloaded from the Hugging Face hub.
"""
import json
import math

import numpy as np

MODEL = "theforecastingcompany/t0-gamma"
PAST = {"stock": ["stock"], "restocks": ["restock"], "siblings": ["sib"], "discount": ["disc_share", "disc_depth"], "category": ["cat"]}
KNOWN = {"soldout": ["so"], "keyout": ["kf"], "season": ["woy_sin", "woy_cos"], "age": ["age"]}
FUTURE_FILL = {"so": 0.0, "kf": 0.0}


def add_lora(model, rank, alpha):
    """Wrap every nn.Linear of the transformer layers with a rank-r adapter (same layout as training)."""
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.a = nn.Parameter(torch.zeros(rank, base.in_features))
            self.b = nn.Parameter(torch.zeros(base.out_features, rank))
            self.s = alpha / rank

        def forward(self, x):
            return self.base(x) + self.s * ((x @ self.a.t()) @ self.b.t())

    for p in model.parameters():
        p.requires_grad = False
    for layer in model.transformer.layers:
        for mod in list(layer.modules()):
            for name, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(mod, name, LoRALinear(child))


def load(path="model", device="cpu"):
    """Base t0-gamma + the saved LoRA adapter → (model in eval mode, config)."""
    import torch
    from t0 import T0Forecaster
    cfg = json.load(open(f"{path}/config.json"))
    model = T0Forecaster.from_pretrained(cfg.get("base_model", MODEL)).to(device).eval()
    add_lora(model, cfg["rank"], cfg["alpha"])
    res = model.load_state_dict(torch.load(f"{path}/lora_adapter.pt", map_location=device), strict=False)
    assert not res.unexpected_keys and not any(k.endswith((".a", ".b")) for k in res.missing_keys), res
    return model.eval(), cfg


def _rows(cfg, h, horizon):
    T = len(h["units"])
    woy = np.asarray(h["woy"], np.float32)
    calc = {"so": h["soldout"], "kf": h["keyout"], "woy_sin": np.sin(2 * np.pi * woy / 52.18), "woy_cos": np.cos(2 * np.pi * woy / 52.18),
            "age": h["age"]}
    past = [np.asarray(h[n], np.float32) for g in cfg["groups"] for n in PAST.get(g, [])]
    known = []
    for g in cfg["groups"]:
        for n in KNOWN.get(g, []):
            x = np.asarray(calc[n], np.float32)
            known.append(np.concatenate([x[:T], np.full(horizon, FUTURE_FILL[n], np.float32)]) if n in FUTURE_FILL else x[:T + horizon])
    return np.asarray(h["units"], np.float32), past, known


def forecast_season(model, cfg, history, horizon=13, weekly=False):
    """Season demand (Σ of weekly forecasts, readout as trained: mean of the 21 quantiles × k)."""
    import torch
    from t0 import TimeSeries
    from t0.types import MaskType, VariateType
    units, past, known = _rows(cfg, history, horizon)
    T, W = len(units), len(units) + horizon
    vals = [np.concatenate([units, np.zeros(horizon, np.float32)])] + \
           [np.concatenate([p, np.zeros(horizon, np.float32)]) for p in past] + known
    mask = [np.concatenate([np.full(T, MaskType.VALID), np.full(horizon, MaskType.WITHHELD)])] + \
           [np.concatenate([np.full(T, MaskType.VALID), np.full(horizon, MaskType.MISSING)]) for _ in past] + \
           [np.full(W, MaskType.VALID) for _ in known]
    vt = [VariateType.TARGET] + [VariateType.HISTORICAL] * len(past) + [VariateType.FUTURE] * len(known)
    ts = TimeSeries(variates=torch.tensor(np.stack(vals), dtype=torch.float32),
                    mask=torch.tensor(np.stack(mask).astype(np.int8)),
                    group_ids=torch.zeros((len(vals), W), dtype=torch.int64),
                    variate_type=torch.tensor(np.repeat(np.array(vt, np.int64)[:, None], W, 1)))
    levels = cfg["levels"]
    with torch.inference_mode():
        q = model.predict(ts, horizon, levels).quantiles.float().cpu().numpy()[0]  # [horizon, 21]
    lv = np.array(levels)
    w = np.diff(np.concatenate([[0], (lv[1:] + lv[:-1]) / 2, [1]]))
    col, k = cfg["readout"]
    week = k * (q @ w if col == "mean" else q[:, levels.index(0.5)])
    return week if weekly else float(week.sum())
