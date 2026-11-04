"""Backward feature elimination for t0-gamma + LoRA (2026-10-25, owner-approved plan) — the t0 twin of
timestral_lora_bwe.py, run on Azure CPU VMs.

Target = raw weekly units of the product colour (all sizes). Same per-colour arrays as the Timestral job
(results/timestral_bwe_arrays.pkl), so both models see identical data. Eleven droppable groups (no fixed base), + `cens` from round 2:
  stock-out signals (known rows; real in training, "fully in stock" over the season at forecast time):
    soldout  1 if weighted availability < 0.5 (t0's best zero-shot input), season = 0
    avail    weighted availability 0–1 (size shares learned before the cutoff), season = 1
    keyout   1 if the key sizes (top half of demand) are out, season = 0
  features:
    stock     past   units on hand at week end
    restocks  past   units added by restocks
    season    known  week of the year as sin / cos
    holiday   known  1 inside a shop sale/holiday window (stores B, C)
    siblings  past   weekly units of the other colours of the same title + type
    discount  past   discount share, discount depth
    category  past   weekly units of the other colours of the same type
    age       known  weeks since the colour's first sale
  (dropped before the run: category-last-year and season strength, see GROUPS)
t0 input: one TimeSeries per sample — TARGET row (units), HISTORICAL rows (past features, not forecast), FUTURE rows
(known features, context + season). Left padding is MaskType.PAD.
LoRA r=32 lr=3e-4 on the 96 linears of the 24 transformer layers (attention wQKV / wO + MLP), base frozen.
Loss = weekly pinball over t0's 21 levels + season-total pinball (Σ over the 13 target weeks of each level vs Σ of
the actual), both ÷ the window's mean context units.
Each run: eval every 125 steps (max 500 — 11 s/step with all groups on 16 vCPU — patience 2); readout median or mean × k chosen on 2025.
Elimination (stage A: every target week before 2025-01-01, choose on the 2025 cases): round 1 = ALL + each group left
out; drop the group whose removal lowers 2025 error most; repeat; stop when no removal helps. STOP RULE: if round-1
ALL does not beat zero-shot t0 on 2025 (47.9% median, and zero-shot with its own 2025-chosen readout), stop there.
Add-back: each dropped group back once; the best is kept if it lowers 2025 error.
Final (stage B): chosen set retrained, same steps, on every window whose targets end before 2026-01-01 (store A
capped at 15,000); 2026 scored once.

    python python/timesfm/t0gamma_bwe.py prep                      (laptop, .venv-lgbm: reads the Timestral arrays pickle)
    python t0gamma_bwe.py one --groups a,b,c --tag NAME [--stage A|B] [--fixed-steps N] [--smoke]   (VM)
    python t0gamma_bwe.py zs                                       (VM: zero-shot references)
    python python/timesfm/t0gamma_bwe.py score                     (laptop)
"""
import argparse
import gzip
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = f"{HERE}/results"
OUT = f"{R}/t0bwe"
DATE = "2026-10-25"
L, H, STEP = 104, 13, 2
RANK, LR, BS = 32, 3e-4, 32
MODEL = "theforecastingcompany/t0-gamma"
LEVELS = [0.01] + [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)] + [0.99]
GRID_K = np.round(np.arange(0.3, 1.81, 0.05), 2)
GROUPS = {"soldout": ([], ["so"]), "avail": ([], ["av"]), "keyout": ([], ["kf"]),
          "stock": (["stock"], []), "restocks": (["restock"], []), "season": ([], ["woy_sin", "woy_cos"]),
          "holiday": ([], ["holiday"]), "siblings": (["sib"], []), "discount": (["disc_share", "disc_depth"], []),
          "category": (["cat"], []), "age": ([], ["age"]), "cens": ([], []),
          "topsize": ([], ["tso", "tsh"])}
# topsize (owner, 2026-10-25, from round 2; FINDINGS "Size stock-outs and demand": losing the best size leaves only
# 22–29% of sales, not 1 − its share): tso = 1 when the most-demanded size is < 20% in stock while the other sizes
# average ≥ 70% (season: 0, in stock); tsh = that size's share of demand (constant). Shares as `avail` (before cutoff).
# cens (owner, 2026-10-25, added from round 2): no extra row — history weeks with the sold-out flag get t0's own
# MaskType.CENSORED mark on the units row ("this value is a lower bound"). Tested zero-shot on 2026-10-25 (lost on 2025).
# Not used (owner, 2026-10-25): "same week last year" rows — cat_ly and season strength (sstr) — are constant 1 in
# every stage-A training window (data starts 2024-01, so targets before 2025 have no last year), yet vary in the
# 2025/2026 cases: the model could never learn them. Both are still built (prep) for a later stage-B-only test.
FUTURE_FILL = {"so": 0.0, "av": 1.0, "kf": 0.0, "tso": 0.0}  # stock-out rows over the forecast weeks: fully in stock
PAST_ALL = ["stock", "restock", "sib", "disc_share", "disc_depth", "cat"]
KNOWN_ALL = ["so", "av", "kf", "woy_sin", "woy_cos", "holiday", "cat_ly", "age", "sstr", "tso", "tsh"]


def err(f, a):
    f, a = np.asarray(f, float), np.asarray(a, float)
    return abs(f - a).sum() / a.sum(), (f - a).sum() / a.sum()


def mean_w(levels):
    lv = np.array(levels)
    return np.diff(np.concatenate([[0], (lv[1:] + lv[:-1]) / 2, [1]]))


# ---------------------------------------------------------------- prep (laptop): windows + cases from the shared arrays
def sstr_row(S, k_i, f, t0, t1, ce):
    """season strength over [t0, t1) as seen at context end ce (see module doc)."""
    u, out = S["units"][k_i], np.ones(t1 - t0, np.float32)
    own = ce - f >= 104
    base = u[max(0, ce - 52):ce].mean() if ce > 0 else 0
    if own and base > 0:
        for j, t in enumerate(range(t0, t1)):
            ly = t - 52
            out[j] = u[max(0, ly - 2):ly + 3].mean() / base if ly >= 0 else 1.0
    else:
        from timestral_lora_bwe import cat_ly
        out = cat_ly(S, k_i, t0, t1, ce)
    return np.nan_to_num(out, nan=1.0).clip(0, 5).astype(np.float32)


def topsize_rows(S, k_i, cut):
    """(top size out flag over the shop timeline, top size share) with size shares from in-stock weeks < cut (as avail_rows)."""
    key = S["keys"][k_i]
    inst = S["inst"].loc[key].fillna(0).values
    su = S["size_units"].loc[key].values
    past = S["weeks"] < cut
    ok = inst[:, past] >= 0.3
    rate = (su[:, past] * ok).sum(1) / np.maximum((inst[:, past] * ok).sum(1), 1e-9)
    sh = rate / rate.sum() if rate.sum() > 0 else np.full(len(rate), 1 / len(rate))
    if len(sh) < 2:
        return np.zeros(inst.shape[1], np.float32), 1.0
    top = int(np.argmax(sh))
    others = np.delete(inst, top, axis=0).mean(0)
    return ((inst[top] < 0.2) & (others >= 0.7)).astype(np.float32), float(sh[top])


def make_windows(A, boundary, cap, rng):
    from timestral_lora_bwe import avail_rows, cat_ly
    W = {n: [] for n in ["units", *PAST_ALL, *KNOWN_ALL]}
    meta = []
    for shop, S in A.items():
        b = int(np.searchsorted(S["weeks"], boundary))
        n0 = len(meta)
        for k_i in range(len(S["keys"])):
            u = S["units"][k_i]
            sold = np.nonzero(u[:b] > 0)[0]
            if not len(sold) or u[:b].sum() < 5:
                continue
            f = sold[0]
            if b - f < 8 + H:
                continue
            av, kf = avail_rows(S, k_i, boundary)
            tso, tsh = topsize_rows(S, k_i, boundary)
            for c in range(f + 8, b - H + 1, STEP):
                lo = max(f, c - L)
                if u[lo:c].sum() < 3:
                    continue
                pad = L - (c - lo)
                padn = np.full(pad, np.nan, np.float32)
                sl = lambda x: np.concatenate([padn, x[lo:c + H]])
                for n in ["units", *PAST_ALL]:
                    W[n].append(sl(S[n][k_i]))
                for n in ("woy_sin", "woy_cos", "holiday"):
                    W[n].append(sl(S[n]))
                W["av"].append(sl(av)), W["kf"].append(sl(kf)), W["so"].append(sl((av < 0.5).astype(np.float32)))
                W["cat_ly"].append(np.concatenate([padn, cat_ly(S, k_i, lo, c + H, c)]))
                W["age"].append(np.concatenate([padn, np.arange(lo - f, c + H - f, dtype=np.float32)]))
                W["sstr"].append(np.concatenate([padn, sstr_row(S, k_i, f, lo, c + H, c)]))
                W["tso"].append(sl(tso))
                W["tsh"].append(np.concatenate([padn, np.full(c + H - lo, tsh, np.float32)]))
                meta.append(shop)
        idx = list(range(n0, len(meta)))
        if shop in cap and len(idx) > cap[shop]:
            drop = set(rng.choice(idx, len(idx) - cap[shop], replace=False).tolist())
            keep = [i for i in range(len(meta)) if i not in drop]
            W = {n: [v[i] for i in keep] for n, v in W.items()}
            meta = [meta[i] for i in keep]
        print(f"    {boundary.date()} {shop}: {sum(1 for m in meta if m == shop)} windows", flush=True)
    return {n: np.stack(v).astype(np.float32) for n, v in W.items()}, np.array(meta)


def make_cases(A):
    import pandas as pd
    from timestral_lora_bwe import avail_rows, cat_ly, holiday
    cases = []
    for shop, S in A.items():
        st = pd.read_csv(f"{HERE}/features/static_{shop}.csv", dtype={"case_id": str, "product_id": str, "scope": str}, keep_default_na=False)
        st["actual_demand"] = pd.to_numeric(st.actual_demand, errors="coerce")
        kx = {k: i for i, k in enumerate(S["keys"])}
        for c in st.dropna(subset=["actual_demand"]).itertuples():
            k_i = kx.get(f"{c.product_id}|{c.scope}")
            if k_i is None:
                continue
            cut = pd.Timestamp(c.cutoff)
            ci = int(np.searchsorted(S["weeks"], cut))
            u = S["units"][k_i]
            sold = np.nonzero(u[:ci] > 0)[0]
            if not len(sold) or ci - sold[0] < 3:
                continue
            f = sold[0]
            Hc = math.ceil(int(c.days) / 7)
            lo = max(f, ci - L)
            av, kf = avail_rows(S, k_i, cut)
            tso, tsh = topsize_rows(S, k_i, cut)
            wk = pd.date_range(S["weeks"][lo], periods=ci - lo + Hc, freq="W-MON")
            woy = np.array([t.isocalendar()[1] for t in wk], np.float32)
            row = {"case_id": c.case_id, "shop": shop, "year": cut.year, "H": Hc, "actual_demand": float(c.actual_demand),
                   "units": u[lo:ci], "av": av[lo:ci], "kf": kf[lo:ci], "so": (av[lo:ci] < 0.5).astype(np.float32),
                   "woy_sin": np.sin(2 * np.pi * woy / 52.18).astype(np.float32), "woy_cos": np.cos(2 * np.pi * woy / 52.18).astype(np.float32),
                   "holiday": np.array([holiday(shop, t) for t in wk], np.float32), "cat_ly": cat_ly(S, k_i, lo, ci + Hc, ci),
                   "age": np.arange(lo - f, ci + Hc - f, dtype=np.float32), "sstr": sstr_row(S, k_i, f, lo, ci + Hc, ci),
                   "tso": tso[lo:ci], "tsh": np.full(ci + Hc - lo, tsh, np.float32)}
            for n in PAST_ALL:
                row[n] = S[n][k_i][lo:ci]
            cases.append({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in row.items()})
    return cases


def prep():
    import pandas as pd
    sys.path.insert(0, HERE)
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(0)
    A = pd.read_pickle(f"{R}/timestral_bwe_arrays.pkl")
    for stage, boundary, cap in (("A", "2025-01-01", {}), ("B", "2026-01-01", {"store-a": 15000})):
        Wd, meta = make_windows(A, pd.Timestamp(boundary), cap, rng)
        np.savez_compressed(f"{OUT}/windows_{stage}.npz", meta=meta, **Wd)
        print(f"stage {stage}: {len(meta)} windows (targets before {boundary})", flush=True)
    cases = make_cases(A)
    with gzip.open(f"{OUT}/cases.json.gz", "wt") as fh:
        json.dump(cases, fh)
    print(f"{len(cases)} cases (2025: {sum(c['year'] == 2025 for c in cases)}, 2026: {sum(c['year'] == 2026 for c in cases)})")


# ---------------------------------------------------------------- t0 inputs
def rows_for(groups):
    past, known = [], []
    for g in groups:
        p, k = GROUPS[g]
        past += p
        known += k
    return past, known


def build_series(units, past, known, horizon, torch, T0, cens=None):
    """units [B, Tc] (NaN = left pad), past [B, P, Tc], known [B, F, Tc + horizon] → TimeSeries, targets first.
    cens [B, Tc] (optional): 1 where the units value is only a lower bound (MaskType.CENSORED)."""
    from t0.types import MaskType, VariateType
    B, Tc = units.shape
    P, F = past.shape[1], known.shape[1]
    W = Tc + horizon
    pad = np.isnan(units)  # left padding only (units have no gaps)

    def ctx_rows(x):  # [B, n, Tc] → values, mask over [Tc + horizon]
        v = np.concatenate([np.nan_to_num(x), np.zeros(x.shape[:2] + (horizon,), np.float32)], -1)
        m = np.where(np.concatenate([np.broadcast_to(pad[:, None], x.shape), np.zeros(x.shape[:2] + (horizon,), bool)], -1),
                     int(MaskType.PAD), int(MaskType.VALID)).astype(np.int8)
        return v, m

    tv, tm = ctx_rows(units[:, None])
    tm[:, :, Tc:] = int(MaskType.WITHHELD)
    if cens is not None:
        c = np.nan_to_num(cens) > 0.5
        tm[:, 0, :Tc] = np.where(c & (tm[:, 0, :Tc] == int(MaskType.VALID)), int(MaskType.CENSORED), tm[:, 0, :Tc])
    hv, hm = ctx_rows(past) if P else (np.zeros((B, 0, W), np.float32), np.zeros((B, 0, W), np.int8))
    if P:
        hm[:, :, Tc:] = int(MaskType.MISSING)  # history rows stop at the cutoff (the rollout buffer withholds them anyway)
    fpad = np.concatenate([pad, np.zeros((B, horizon), bool)], -1)
    fv = np.nan_to_num(known)
    fm = np.where(np.broadcast_to(fpad[:, None], known.shape), int(MaskType.PAD), int(MaskType.VALID)).astype(np.int8)
    vals = np.concatenate([tv[:, 0], hv.reshape(-1, W), fv.reshape(-1, W)]).astype(np.float32)
    mask = np.concatenate([tm[:, 0], hm.reshape(-1, W), fm.reshape(-1, W)])
    gid = np.concatenate([np.arange(B), np.repeat(np.arange(B), P), np.repeat(np.arange(B), F)])
    vt = np.concatenate([np.full(B, int(VariateType.TARGET)), np.full(B * P, int(VariateType.HISTORICAL)),
                         np.full(B * F, int(VariateType.FUTURE))])
    gid_m = np.where(mask == int(MaskType.PAD), -1, gid[:, None]).astype(np.int64)  # PAD cells: -1 sentinels (as TimeSeries.batch)
    vt_m = np.where(mask == int(MaskType.PAD), -1, vt[:, None]).astype(np.int64)
    return T0(variates=torch.from_numpy(vals), mask=torch.from_numpy(mask.astype(np.int8)),
              group_ids=torch.from_numpy(gid_m), variate_type=torch.from_numpy(vt_m))


def batch_arrays(Wd, idx, groups, train=True):
    """→ units, past, known, target, cens flags (None unless the cens group is on)."""
    past, known = rows_for(groups)
    u = Wd["units"][idx][:, :L]
    cens = Wd["so"][idx][:, :L] if "cens" in groups else None
    p = np.stack([Wd[n][idx][:, :L] for n in past], 1) if past else np.zeros((len(idx), 0, L), np.float32)
    k = np.stack([Wd[n][idx] for n in known], 1) if known else np.zeros((len(idx), 0, L + H), np.float32)
    return u, p, k, Wd["units"][idx][:, L:], cens


def case_arrays(chunk, groups):
    """left-pad a chunk of same-H cases to one width; stock-out rows set to 'in stock' over the season."""
    past, known = rows_for(groups)
    Hc = chunk[0]["H"]
    T = max(len(c["units"]) for c in chunk)
    padl = lambda a, t: np.concatenate([np.full(t - len(a), np.nan, np.float32), np.asarray(a, np.float32)])
    u = np.stack([padl(c["units"], T) for c in chunk])
    p = np.stack([np.stack([padl(c[n], T) for n in past]) for c in chunk]) if past else np.zeros((len(chunk), 0, T), np.float32)
    ks = []
    for c in chunk:
        rows = []
        for n in known:
            if n in FUTURE_FILL:
                rows.append(padl(np.concatenate([c[n], np.full(Hc, FUTURE_FILL[n], np.float32)]), T + Hc))
            else:
                rows.append(padl(c[n], T + Hc))
        ks.append(np.stack(rows) if rows else np.zeros((0, T + Hc), np.float32))
    cens = np.stack([padl(c["so"], T) for c in chunk]) if "cens" in groups else None
    return u, p, np.stack(ks), Hc, cens


# ---------------------------------------------------------------- model
def load(device):
    import torch
    from t0 import T0Forecaster
    return T0Forecaster.from_pretrained(MODEL).to(device).eval()


def add_lora(model, rank, alpha):
    import torch
    from torch import nn

    class LoRALinear(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.a = nn.Parameter(torch.randn(rank, base.in_features) / math.sqrt(base.in_features))
            self.b = nn.Parameter(torch.zeros(base.out_features, rank))
            self.s = alpha / rank

        @property
        def weight(self):
            return self.base.weight + self.s * self.b @ self.a

        def forward(self, x):
            return self.base(x) + self.s * ((x @ self.a.t()) @ self.b.t())

    for p in model.parameters():
        p.requires_grad = False
    n = 0
    for layer in model.transformer.layers:
        for mod in list(layer.modules()):
            for name, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    setattr(mod, name, LoRALinear(child))
                    n += 1
    return n, [p for p in model.parameters() if p.requires_grad]


def evaluate(model, cases, groups, torch):
    from t0 import TimeSeries
    w = mean_w(LEVELS)
    rows, by_h = [], {}
    for c in cases:
        by_h.setdefault(c["H"], []).append(c)
    for Hc, grp in by_h.items():
        for i in range(0, len(grp), 32):
            chunk = grp[i:i + 32]
            u, p, k, _, cens = case_arrays(chunk, groups)
            ts = build_series(u, p, k, Hc, torch, TimeSeries, cens)
            with torch.inference_mode():
                q = model.predict(ts, Hc, LEVELS).quantiles.float().cpu().numpy()  # [B, Hc, 21]
            qs = q.sum(1)
            for c, s in zip(chunk, qs):
                rows.append({"case_id": c["case_id"], "shop": c["shop"], "year": c["year"], "actual_demand": c["actual_demand"],
                             "median": float(s[LEVELS.index(0.5)]), "mean": float(s @ w)})
    import pandas as pd
    return pd.DataFrame(rows)


def select(df):
    g = df[df.year == 2025]
    return min(((err(kk * g[col], g.actual_demand)[0], col, float(kk)) for col, kk in [("median", 1.0)] + [("mean", kk) for kk in GRID_K]))


def lora_state(model):
    """copy of the LoRA adapter tensors (the a / b matrices of every wrapped linear)."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def save_lora(sd, path, meta, torch):
    """LoRA adapter + what is needed to rebuild and read it out."""
    os.makedirs(path, exist_ok=True)
    torch.save(sd, f"{path}/lora_adapter.pt")
    json.dump({**meta, "base_model": MODEL, "rank": RANK, "alpha": 2 * RANK, "n_tensors": len(sd),
               "n_params": int(sum(v.numel() for v in sd.values())), "torch": torch.__version__,
               "levels": LEVELS, "L": L, "H": H}, open(f"{path}/config.json", "w"), indent=1)
    print(f"saved LoRA adapter → {path} ({len(sd)} tensors)", flush=True)


def load_lora(path, device="cpu"):
    """Rebuild the trained model: base t0-gamma + LoRA wrappers + saved adapter."""
    import torch
    cfg = json.load(open(f"{path}/config.json"))
    model = load(device)
    add_lora(model, cfg["rank"], cfg["alpha"])
    missing = model.load_state_dict(torch.load(f"{path}/lora_adapter.pt"), strict=False)
    # only the frozen base weights may be "missing" (they come from the hub); every adapter tensor must load
    assert not missing.unexpected_keys and not any(k.endswith((".a", ".b")) for k in missing.missing_keys), missing
    return model.eval(), cfg


def one(groups, tag, stage, fixed_steps, smoke, max_steps=1000, every=125, patience=2, data=HERE, save=None, readout=None):
    import torch
    from t0 import TimeSeries
    from t0.rollout import prepare_rollout_buffer
    from t0.types import round_up
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count())
    z = np.load(f"{data}/windows_{stage}.npz", allow_pickle=False)
    Wd = {n: z[n] for n in z.files if n != "meta"}
    cases = json.load(gzip.open(f"{data}/cases.json.gz", "rt"))
    if smoke:
        cases = [c for c in cases if c["year"] == 2025][:6] + [c for c in cases if c["year"] == 2026][:6]
        max_steps, every, patience = 4, 2, 1
        if fixed_steps:
            fixed_steps = 2
    model = load("cpu")
    n_lin, params = add_lora(model, RANK, 2 * RANK)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    lv = torch.tensor(LEVELS)
    gen = torch.Generator().manual_seed(0)
    hz = round_up(H, model.patch_size)
    cw = round_up(L, model.patch_size)
    t0, losses, step, bad = time.time(), [], 0, 0
    print(f"[{tag}] stage {stage} · {len(Wd['units'])} windows · {len(cases)} cases · groups {groups} · LoRA on {n_lin} linears", flush=True)
    best = None
    if fixed_steps is None:
        model.eval()
        df = evaluate(model, cases, groups, torch)
        best = (select(df), 0, df)
        best_sd = lora_state(model)
        print(f"[{tag}] step 0 (zero-shot, these inputs): 2025 {best[0][0]:.4f} ({best[0][1]}×{best[0][2]}) · {time.time() - t0:.0f}s", flush=True)
    target = fixed_steps or max_steps
    while step < target and bad < patience:
        for idx in torch.randperm(len(Wd["units"]), generator=gen).split(BS):
            u, p, k, y, cens = batch_arrays(Wd, idx.numpy(), groups)
            ts = build_series(u, p, k, H, torch, TimeSeries, cens)
            model.train()
            buf = prepare_rollout_buffer(ts, H, L, model.patch_size)
            q = model._predict_step(buf.time_slice(0, cw + hz), hz)[: len(idx), :H]  # [B, H, 21], data space
            yt = torch.from_numpy(y)
            s = torch.from_numpy(np.nanmean(u, 1)).clamp(min=0.5)
            d = yt[..., None] - q
            weekly = torch.maximum(lv * d, (lv - 1) * d).mean(dim=(1, 2))
            ds = yt.sum(1)[..., None] - q.sum(1)  # season total, each level
            season = torch.maximum(lv * ds, (lv - 1) * ds).mean(1) / H
            loss = ((weekly + season) / s).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
            if step <= 3:
                print(f"[{tag}]   step {step}: loss {losses[-1]:.4f} · {time.time() - t0:.1f}s", flush=True)
            if fixed_steps is None and (step % every == 0 or step >= target):
                model.eval()
                df = evaluate(model, cases, groups, torch)
                sel = select(df)
                print(f"[{tag}] step {step}: loss {np.mean(losses[-every:]):.4f} · 2025 {sel[0]:.4f} ({sel[1]}×{sel[2]}) · {time.time() - t0:.0f}s", flush=True)
                if sel[0] < best[0][0] - 1e-4:
                    best, bad = (sel, step, df), 0
                    best_sd = lora_state(model)
                else:
                    bad += 1
            if step >= target or bad >= patience:
                break
    if fixed_steps is not None:
        model.eval()
        df = evaluate(model, cases, groups, torch)
        best = (select(df), step, df)
        best_sd = lora_state(model)
    (e25, col, kk), steps, df = best
    if save:  # the adapter of the step whose forecasts are written below
        save_lora(best_sd, save, {"tag": tag, "groups": groups, "stage": stage, "steps": steps,
                                  "readout": readout or [col, kk]}, torch)
    os.makedirs("results", exist_ok=True)
    df.assign(run=tag, readout=col, k=kk).to_csv(f"results/raw_{tag}.csv", index=False)
    res = {"tag": tag, "groups": groups, "stage": stage, "e25": e25, "readout": col, "k": kk, "steps": steps,
           "min": round((time.time() - t0) / 60, 1), "sec_per_step": round((time.time() - t0) / max(step, 1), 2)}
    json.dump(res, open(f"results/res_{tag}.json", "w"))
    print(f"== [{tag}] 2025 {e25:.4f} at step {steps} ({col}×{kk}) · {res['min']} min", flush=True)


def zs():
    """zero-shot references: soldout only (= F1) and ALL 12, readout median and chosen on 2025."""
    import torch
    cases = json.load(gzip.open(f"{HERE}/cases.json.gz", "rt"))
    model = load("cpu")
    torch.set_num_threads(os.cpu_count())
    out = {}
    for name, groups in (("F1", ["soldout"]), ("ALL", list(GROUPS))):
        df = evaluate(model, cases, groups, torch)
        g = df[df.year == 2025]
        out[name] = {"median": err(g["median"], g.actual_demand)[0], "chosen": select(df)}
        df.assign(run=f"zero-shot {name}").to_csv(f"results/raw_zs_{name}.csv", index=False)
        print(f"zero-shot {name}: 2025 median {out[name]['median']:.4f} · chosen {out[name]['chosen']}", flush=True)
    json.dump(out, open("results/zs.json", "w"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "one", "zs"])
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--stage", default="A")
    ap.add_argument("--fixed-steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--data", default=HERE, help="folder with windows_A/B.npz and cases.json.gz")
    ap.add_argument("--save", default=None, help="folder to save the LoRA adapter + config into")
    ap.add_argument("--readout", default=None, help="readout to store with the weights, e.g. mean,0.9 (chosen in stage A)")
    ap.add_argument("--max-steps", type=int, default=500, help="owner, 2026-10-25: 500 (11 s/step with all groups on 16 vCPU)")
    a = ap.parse_args()
    if a.step == "prep":
        prep()
    elif a.step == "one":
        ro = [a.readout.split(",")[0], float(a.readout.split(",")[1])] if a.readout else None
        one([g for g in a.groups.split(",") if g], a.tag, a.stage, a.fixed_steps, a.smoke, max_steps=a.max_steps,
            data=a.data, save=a.save, readout=ro)
    else:
        zs()
