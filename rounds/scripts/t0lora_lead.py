"""t0 LoRA lead-time + last-year-season experiment (owner-approved 2026-10-26, research only, no live change).

Problem: the trained t0 LoRA (Sizo-AI/t0-season-model) was trained and scored only on seasons forecast FROM
their first day (13 weeks ahead). The buying plan forecasts seasons weeks/months ahead, and there the model
flattens the season (store A swimwear: winter forecast ~18× last winter; spring ~40% too low). Its
week-of-year input moves a forecast by < 10%; the "same weeks last year" rows were never trained (the
research's stage-A windows had no last year).

Arms (each a fresh LoRA on base t0-gamma, rank 32, lr 3e-4, batch 32, 375 steps, readout mean×0.9 —
the current model's recipe):
  base     the current model (model repo 548e630), evaluated only
  lead     the 9 current groups, trained on forecasts made 0/4/8/13/20/26 weeks before the 13 target weeks
  catly    + the category's units in the same weeks last year (known over the forecast; ÷ its mean over the
           52 weeks before the origin; research cat_ly)
  ownly    + catly + the colour's own units in the same weeks last year (same scaling; 1 when it did not exist)
Training targets end before 2025-12-01, so Winter 25/26, Spring 26 and Summer 26 are unseen by the new arms
(the base model trained through 2025-12-31 — it has seen December 2025; that favours it).

Test: EVERY colour of the 7 research shops that sold ≥ 5 units in the 52 weeks before the origin, for each
shop's Settings seasons starting 2025-12-01 … 2026-06-30, forecast 0/4/8/13/26 weeks before the season.
Actual = stock-corrected season demand (shares from before the season start), also raw units.

    python t0lora_lead.py prep                       # laptop: windows_lead.npz + test_cases.pkl.gz
    python t0lora_lead.py train --arm catly          # VM
    python t0lora_lead.py eval --arm catly           # VM (base: --arm base --model-dir <t0-season-model>/model)
    python t0lora_lead.py score                      # laptop
"""
import argparse
import gzip
import json
import math
import os
import pickle
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t0gamma_bwe as B  # noqa: E402  (the research's build_series / LoRA / save-load, unchanged)

L, H, BS, RANK, LR = B.L, B.H, B.BS, B.RANK, B.LR
L = int(os.environ.get("CTXL", L))  # season-only arm: context = last year's season + the recent weeks (≤ 36)
LEVELS = B.LEVELS
LEADS_TRAIN = [0, 4, 8, 13, 20, 26]
LEADS_TEST = [0, 4, 8, 13, 26]
MAXLEAD = 26
SPAN = L + MAXLEAD + H  # stored window: context + the longest lead + the target season
BOUNDARY = os.environ.get("BOUNDARY", "2025-12-01")  # training targets end before this
TAG = os.environ.get("TAG", "")  # file suffix for a second window/test set
TEST_FROM, TEST_TO = os.environ.get("TEST_FROM", "2025-12-01"), os.environ.get("TEST_TO", "2026-06-30")
STEPS = int(os.environ.get("STEPS", "375"))
K = float(os.environ.get("KREAD", "0.9"))  # readout mean × K (fix run: 1.0)
# owner 2026-10-28 low-bias fix: learn only from target weeks WITH stock (58% had none), and match the season total
MASK = os.environ.get("MASK") == "1"
MEANLOSS = os.environ.get("MEANLOSS") == "1"
PAST = ["stock", "restock", "sib", "disc_share", "disc_depth", "cat"]
KNOWN_BASE = ["so", "kf", "woy_sin", "woy_cos", "age"]
SEASON_ROWS = ["s_winter", "s_spring", "s_summer", "s_fall"]
ARMS = {"base": KNOWN_BASE, "lead": KNOWN_BASE, "catly": KNOWN_BASE + ["cat_ly"], "ownly": KNOWN_BASE + ["cat_ly", "own_ly"],
        # owner 2026-10-27: flag each week with the shop's season and how much the product's CATEGORY sells in that season
        "season": KNOWN_BASE + ["cat_ly"] + SEASON_ROWS + ["cat_sidx"],
        # + "season change from now": the category season index ÷ its mean over the 13 weeks before the origin
        "season2": KNOWN_BASE + ["cat_ly"] + SEASON_ROWS + ["cat_sidx", "cat_srel"]}
# owner 2026-10-28: siblings = the 10 products in the shop with the most similar CURVE (not the same title).
#   csales  sales-over-the-year curve (52 weeks before the anchor, 13 four-week buckets, shares)
#   csize   size curve (size shares from in-stock weeks before the anchor, 16 points along the ladder)
# rows: past "csib_x" = siblings' units ÷ each one's 52-week mean; known "csib_x_ly" = the same 52 weeks earlier
# (the siblings' season ahead). No siblings (no curve yet) → the category instead.
ARMS["csales"] = ARMS["season2"] + ["csib_s_ly"]
ARMS["csize"] = ARMS["season2"] + ["csib_z_ly"]
ARM_PAST = {"csales": ["csib_s"], "csize": ["csib_z"]}
CSIB_ROWS = ["csib_s", "csib_s_ly", "csib_z", "csib_z_ly"]
ANCHOR, NSIB = 13, 10
# round 7 (owner 2026-10-28): DEMAND=1 builds the cat / sib / category-last-year / sibling rows from stock-corrected
# demand (units ÷ max(availability, 0.3), the test's own "actual"), not raw units; CENS=1 marks the units of sold-out
# weeks MaskType.CENSORED ("a lower bound"). Curve siblings that did not exist last year are left out of last year.
DEMAND = os.environ.get("DEMAND") == "1"
CENS = os.environ.get("CENS") == "1"
# round 9 (owner 2026-10-29): all history, and a sold-out target week teaches "demand was AT LEAST what sold"
# (one-sided loss) instead of being dropped; the season rows tell the model whether that week is a strong or a weak one
CLOSS = os.environ.get("CLOSS") == "1"
CLOSSW = float(os.environ.get("CLOSSW", "1.0"))  # weight of sold-out target weeks under CLOSS
# round 11 (owner 2026-10-31): DROP=1 adds a "release" row (the weeks a colour goes on sale: its first sale, or the first
# sale after 3+ quiet weeks with stock received in the 8 weeks before) — known ahead, the merchant plans launches/drops.
# CLOSS_SEASON=1: the one-sided sold-out loss only where the category sells above its average then (catrel >= 1);
# a sold-out week off-season counts as real low demand (normal loss). NEWCOL=1: training windows also for colours
# not yet on sale at the forecast date that launch within the target season.
DROP = os.environ.get("DROP") == "1"
CLOSS_SEASON = os.environ.get("CLOSS_SEASON") == "1"
NEWCOL = os.environ.get("NEWCOL") == "1"
# round 12 (owner 2026-10-31): LAUNCH=1 — a colour not on sale yet gets SIBLINGS as its stand-in, not the category ÷
# colours: the same style's other colours (their average colour), else "launch siblings" = the same category's colours
# that launched in last year's same season (their average colour, week by week).
LAUNCH = os.environ.get("LAUNCH") == "1"
# round 13 (owner 2026-11-01): BOOST=k draws training windows with a release week in the target or a colour not on sale
# yet (all-stand-in context) k× as often; SODEM=1 puts stock-corrected demand (not the few units sold) in the colour's
# own sold-out context weeks — the sold-out flag row stays.
BOOST = float(os.environ.get("BOOST", "1"))
SODEM = os.environ.get("SODEM") == "1"
# round 14 (owner 2026-11-04): TYPEFILL=1 fills an empty Shopify product type from the title (dress/top/tee/pants…),
# mapped to the shop's own type that most often carries that word; non-clothing (cards, perfume, samples) stays empty.
# STYLE=1: a colour with no sales yet whose style (same title + type) already has colours on sale gets the style's
# average colour as its stand-in (not the category ÷ its number of colours); no other launch stand-in.
TYPEFILL = os.environ.get("TYPEFILL") == "1"
STYLE = os.environ.get("STYLE") == "1"
TYPE_WORDS = [  # (class, title words) — first match wins, so tee before top and jumpsuit before suit
    ("skip", ["כרטיס ברכה", "גלוי", "edp", "gift card", "שובר", "סאמפל", "משלוח", "test", "candle", "soap", "lotion",
              "cream", "mist", "balm", "shopping bag", "סבון"]),
    ("swim", ["bikini", "swim", "one piece", "בגד ים", "ביקיני"]),
    ("dress", ["dress", "שמלה", "שמלת"]),
    ("jumpsuit", ["jumpsuit", "overall", "אוברול", "חליפ"]),
    ("tee", ["tee", "t-shirt", "tshirt", "טי "]),
    ("top", ["top", "blouse", "tank", "halter", "cami", "bodysuit", "shirt", "חולצ", "גופי", "טופ"]),
    ("skirt", ["skirt", "חצאית"]),
    ("shorts", ["shorts", "bermuda", "שורט", "ברמודה"]),
    ("pants", ["pants", "trousers", "leggings", "jeans", "מכנס", "טייץ", "ג'ינס"]),
    ("jacket", ["jacket", "coat", "blazer", "ז'קט", "מעיל", "בלייזר"]),
    ("knit", ["sweater", "cardigan", "knit", "סוודר", "קרדיגן"]),
]


def title_class(title):
    t = f" {title.lower()} "
    for c, words in TYPE_WORDS:
        if any(w in t for w in words):
            return c
    return ""


def fill_types(titles, typ):
    """empty types → the shop's type whose titles most often fall in the same class (≥ 3 of them), else '~class'.
    A shop that left most types empty (store-d: 94% of units) has product-level types, not categories: there every
    product with a class gets '~class'."""
    cls = np.array([title_class(t) for t in titles])
    if np.mean(typ == "") > 0.5:
        return np.array([f"~{c}" if c not in ("", "skip") else x for x, c in zip(typ, cls)], dtype=object)
    best = {}
    for c in set(cls) - {"", "skip"}:
        have = [x for x, cc in zip(typ, cls) if cc == c and x]
        if len(have) >= 3:
            v, n = np.unique(have, return_counts=True)
            best[c] = v[np.argmax(n)]
    out = typ.copy()
    for i, (x, c) in enumerate(zip(typ, cls)):
        if not x and c not in ("", "skip"):
            out[i] = best.get(c, f"~{c}")
    return out
RUN = os.environ.get("RUN", "")  # adapter/eval suffix for a second run of the same arm
INIT = os.environ.get("INIT", "")  # warm start from this adapter dir (else a fresh LoRA on base t0-gamma)
OVERSAMPLE = float(os.environ.get("OVERSAMPLE", "0"))  # extra weight on windows whose target season differs from now
ALL_KNOWN = KNOWN_BASE + ["cat_ly", "own_ly"] + SEASON_ROWS + ["cat_sidx"]
FUTURE_FILL = {"so": 0.0, "kf": 0.0}
SEASONS = {  # each shop's Settings seasons (backtest.py SEASONS)
    "store-a": [("Spring", "03-01", "05-31"), ("Summer", "06-01", "08-31"), ("Fall", "09-01", "11-30"), ("Winter", "12-01", "02-28")],
    "store-b": [("Spring", "02-01", "03-30"), ("Summer", "04-01", "08-31"), ("Fall", "09-01", "11-30"), ("Winter", "12-01", "02-28")],
    "store-e": [("Summer", "03-01", "08-31"), ("Winter", "09-01", "02-28")],
    "store-d": [("Spring", "03-01", "05-31"), ("Summer", "06-01", "08-31")],
    "store-c": [("Spring", "03-01", "05-31"), ("Summer", "06-01", "08-31"), ("Fall", "09-01", "11-30"), ("Winter", "12-01", "02-28")],
    "store-g": [("Spring", "03-01", "05-31"), ("Summer", "06-01", "09-14"), ("Fall", "09-15", "11-30"), ("Winter", "12-01", "02-28")],
    "store-f": [("Spring", "03-01", "05-31"), ("Summer", "06-01", "08-31"), ("Fall", "09-01", "11-30"), ("Winter", "12-01", "02-28")],
}


# ---------------------------------------------------------------- research definitions (timestral_lora_bwe.py)
def avail_rows(S, k_i, cut):
    """weighted availability + key-sizes-out flag over the shop timeline, size shares from in-stock weeks < cut."""
    key = S["keys"][k_i]
    inst = S["inst"].loc[key].fillna(0)
    su = S["size_units"].loc[key]
    past = S["weeks"] < cut
    ok = inst.values[:, past] >= 0.3
    rate = (su.values[:, past] * ok).sum(1) / np.maximum((inst.values[:, past] * ok).sum(1), 1e-9)
    sh = rate / rate.sum() if rate.sum() > 0 else np.full(len(rate), 1 / len(rate))
    av = (inst.values * sh[:, None]).sum(0)
    order = np.argsort(-sh)
    top = order[: int(np.searchsorted(np.cumsum(sh[order]), 0.5) + 1)]
    kav = (inst.values[top] * sh[top, None]).sum(0) / max(sh[top].sum(), 1e-9)
    return av.astype(np.float32), (kav < 0.2).astype(np.float32)


def ly_row(x, t0, t1, c_end, first=0):
    """x 52 weeks earlier on [t0, t1), ÷ x's mean over the 52 weeks before c_end; 1 where there is no last year
    (before the series' first week) or no base; clipped 0..10. With x = category units this is research cat_ly."""
    ly = np.array([x[t - 52] if t - 52 >= first else np.nan for t in range(t0, t1)], np.float32)
    lo = max(first, c_end - 52)
    base = x[lo:c_end].mean() if c_end > lo else 0
    return np.nan_to_num(ly / base if base > 0 else ly * np.nan, nan=1.0).clip(0, 10).astype(np.float32)


def season_names(shop, weeks):
    """The shop's Settings season name (lower case) of every Monday: the season containing the week's Thursday."""
    import pandas as pd
    out = []
    for w in weeks:
        d = pd.Timestamp(w) + pd.Timedelta(days=3)
        name = ""
        for n, st, en in SEASONS[shop]:
            for y in (d.year - 1, d.year):
                a = pd.Timestamp(f"{y}-{st}")
                b = pd.Timestamp(f"{y + (1 if en < st else 0)}-{en}") + pd.Timedelta(days=1)
                if a <= d < b:
                    name = n.lower()
        out.append(name)
    return np.array(out)


def season_rows(names, t0, t1):
    """One-hot rows (winter, spring, summer, fall) over [t0, t1) of the shop timeline."""
    return {r: (names[t0:t1] == r[2:]).astype(np.float32) for r in SEASON_ROWS}


def season_index(x, names, t0, t1, c_end):
    """The category's mean weekly units in the same named season, over every week before c_end, ÷ its mean
    over the 52 weeks before c_end; 1 without history or base; clipped 0..10."""
    base = x[max(0, c_end - 52):c_end].mean() if c_end > 0 else 0
    out = np.ones(t1 - t0, np.float32)
    if base <= 0:
        return out
    past_names = names[:c_end]
    means = {n: x[:c_end][past_names == n].mean() for n in set(names[t0:t1]) if n and (past_names == n).any()}
    for i, t in enumerate(range(t0, t1)):
        n = names[t] if t < len(names) else ""
        if n in means:
            out[i] = min(10.0, max(0.0, means[n] / base))
    return out


def srel_rows(sidx, T):
    """[B, W] category season index → ÷ its mean over the 13 weeks before column T (the origin); clipped 0..20."""
    recent = np.nanmean(sidx[:, max(0, T - 13):T], axis=1, keepdims=True)
    recent = np.where(np.isfinite(recent), recent, 1.0)
    return np.clip(sidx / np.maximum(recent, 0.05), 0, 20).astype(np.float32)


def rows_for_arm(arm):
    if arm == "sonly":
        return ["stock", "disc_share"], ARMS[arm]
    return PAST + ARM_PAST.get(arm, []), ARMS[arm]


# ---------------------------------------------------------------- round 8: season-only input (owner 2026-10-28)
# Planning winter, the summer weeks do not help. The context is ONLY last year's same season (weeks before the colour
# existed = the category's demand × the colour's share of its category, flagged proxy=1) + the RECENT weeks before the
# forecast date (the current level). The horizon is the season itself. Rows: units; past stock, disc_share; known
# so / kf (own weeks; 0 on proxy weeks and over the season), proxy, blk (-1 last year, 0 recent, 1 season) and catrel
# (the category's demand ÷ its 52-week mean; over the season = the same weeks last year).
ARMS["sonly"] = ["so", "kf", "proxy", "blk", "catrel"]
FUTURE_FILL["proxy"] = 0.0
RECENT, CTX_SONLY = int(os.environ.get("RECENT", "8")), 36
# round 8b (PROXY=style): the stand-in for weeks before the colour existed is the CLOSEST group that sold last year:
# the same style's other colours → the same-category size-curve siblings → the whole category (proxy row 1/3, 2/3, 1).
PROXY = os.environ.get("PROXY", "category")
if DROP:
    ARMS["sonly"] = ARMS["sonly"] + ["drop"]


def release_weeks(u, restock):
    """1 in the weeks a colour goes on sale: its first sale, or a sale after 3+ weeks without one when stock was
    received in the 8 weeks before (a drop / relaunch / back in stock)."""
    r = np.zeros(len(u), np.float32)
    sold = np.nonzero(u > 0)[0]
    if len(sold):
        r[sold[0]] = 1.0
    for t in sold[1:]:
        if u[max(0, t - 3):t].sum() == 0 and restock[max(0, t - 8):t + 1].sum() > 0:
            r[t] = 1.0
    return r


def launch_stand_in(S, k_i, o, si):
    """round 12: (series, level) for a colour with no sales before o, or None."""
    D, first = S["demand"], S["first"]
    sty = np.flatnonzero((S["model"] == S["model"][k_i]) & (first < o) & (np.arange(len(first)) != k_i))
    if len(sty):
        return D[sty].mean(0), 1 / 3
    ly = si - 52
    ls = np.flatnonzero((S["type"] == S["type"][k_i]) & (first >= ly - 4) & (first < ly + 13) & (np.arange(len(first)) != k_i))
    if len(ls) >= 3:
        return D[ls].mean(0), 2 / 3
    return None


def stand_in(S, k_i, o, tabs, need, si=None):
    """(series, level): the closest group with sales over the `need` weeks; each ÷ the colour's share of it."""
    if STYLE and S["units"][k_i][:o].sum() <= 0:
        D, first = S["demand"], S["first"]
        sty = np.flatnonzero((S["model"] == S["model"][k_i]) & (first < o) & (np.arange(len(first)) != k_i))
        if len(sty):
            return D[sty].mean(0), 1 / 3
    if LAUNCH and si is not None and S["units"][k_i][:o].sum() <= 0:
        r = launch_stand_in(S, k_i, o, si)
        if r is not None:
            return r
    groups = []
    if PROXY == "style":
        groups.append((S["sib"][k_i], 1 / 3))
        T = len(S["weeks"])
        a = max([x for x in range(52, T + 1, ANCHOR) if x <= o], default=None)
        sibs = tabs["z"].get(a, {}).get(k_i, []) if tabs and a is not None else []
        if sibs:
            groups.append((S["demand"][sibs].sum(0), 2 / 3))
    groups.append((S["cat"][k_i], 1.0))
    u = S["units"][k_i]
    for x, lvl in groups:
        if x[need].sum() <= 0 and lvl < 1:
            continue
        cs, us = x[o - RECENT:o].sum(), u[o - RECENT:o].sum()
        if cs <= 0 or us <= 0:
            cs, us = x[max(0, o - 52):o].sum(), u[max(0, o - 52):o].sum()
        if (cs <= 0 or us <= 0) and lvl < 1:
            continue
        if us <= 0 and lvl >= 1:
            # round 9: a colour with no sales of its own yet gets an average colour's share of its category
            same = S["type"] == S["type"][k_i]
            n = int((S["demand"][same][:, max(0, o - 52):o].sum(1) > 0).sum())
            return x / max(n, 1), lvl
        return x * (us / cs if cs > 0 else 0.0), lvl
    return S["cat"][k_i] * 0.0, 1.0


def sonly_rows(S, k_i, si, o, Hs, av, kf, tabs=None):
    u, st, ds = S["units"][k_i], S["stock"][k_i], S["disc_share"][k_i]
    catd = S["cat"][k_i] + S["demand"][k_i]
    sold = np.nonzero(u > 0)[0]
    f = int(sold[0]) if len(sold) else o
    ctx = list(range(si - 52, si - 52 + Hs)) + list(range(o - RECENT, o))
    blk = [-1.0] * Hs + [0.0] * RECENT
    own = np.array([t >= f for t in ctx])
    proxy_x, lvl = stand_in(S, k_i, o, tabs, [t for t, ok in zip(ctx, own) if not ok] or [ctx[0]], si)
    base = catd[max(0, o - 52):o].mean()
    rel = lambda t: float(np.clip(catd[t] / base, 0, 10)) if base > 0 else 1.0
    return {
        "units": np.array([(S["demand"][k_i][t] if SODEM and av[t] < 0.5 else u[t]) if ok else proxy_x[t]
                           for t, ok in zip(ctx, own)], np.float32),
        "proxy": ((~own) * lvl).astype(np.float32),
        "so": np.array([float(av[t] < 0.5) if ok else 0.0 for t, ok in zip(ctx, own)], np.float32),
        "kf": np.array([kf[t] if ok else 0.0 for t, ok in zip(ctx, own)], np.float32),
        "stock": np.array([st[t] for t in ctx], np.float32),
        "disc_share": np.array([ds[t] for t in ctx], np.float32),
        "blk": np.array(blk + [1.0] * Hs, np.float32),
        "catrel": np.array([rel(t) for t in ctx] + [rel(t - 52) for t in range(si, si + Hs)], np.float32),
        **({"drop": np.concatenate([[rw[t] if ok else 0.0 for t, ok in zip(ctx, own)], rw[si:si + Hs]]).astype(np.float32)}
           if DROP and (rw := release_weeks(u, S["restock"][k_i])) is not None else {}),
    }


def prep_sonly():
    """training windows: every colour × every Settings season whose first 13 weeks end before BOUNDARY and whose
    last-year season is in the data, forecast from each training lead; the target = the season's first H weeks."""
    import pandas as pd
    A = demandize(pd.read_pickle(os.environ.get("ARRAYS", f"{HERE}/timestral_bwe_arrays.pkl")))
    rng = np.random.default_rng(0)
    names = ["units", "stock", "disc_share", "so", "kf", "proxy", "blk", "catrel"] + (["drop"] if DROP else [])
    W = {n: [] for n in names}
    lead_of, shop_of = [], []
    bnd = pd.Timestamp(BOUNDARY)
    for shop, S in A.items():
        weeks, b, n0 = S["weeks"], int(np.searchsorted(S["weeks"], bnd)), len(lead_of)
        tabs = sib_tables(S) if PROXY == "style" else None
        for name, st, en in SEASONS[shop]:
            for year in range(weeks[0].year, bnd.year + 1):
                s0 = monday_on_or_after(f"{year}-{st}")
                si = int(np.searchsorted(weeks, s0))
                s1 = monday_on_or_after(pd.Timestamp(f"{year + (1 if en < st else 0)}-{en}") + pd.Timedelta(days=1))
                Hs = (s1 - s0).days // 7
                if si - 52 < 0 or Hs < H or si + H > b or si >= len(weeks):
                    continue
                for k_i in range(len(S["keys"])):
                    u = S["units"][k_i]
                    for lead in LEADS_TRAIN:
                        o = si - lead
                        if o - RECENT < 0:
                            continue
                        if u[:o].sum() < 5:  # not on sale yet: only NEWCOL, and only if it launches in the target weeks
                            if not (NEWCOL and u[:o].sum() == 0 and u[si:si + H].sum() >= 3):
                                continue
                        av, kf = avail_rows(S, k_i, weeks[o])
                        r = sonly_rows(S, k_i, si, o, Hs, av, kf, tabs)
                        n = len(r["units"])
                        padl = np.full(L - n, np.nan, np.float32)
                        tgt = lambda x: np.asarray(x[si:si + H], np.float32)
                        W["units"].append(np.concatenate([padl, r["units"], tgt(u)]))
                        for m in ["stock", "disc_share", "proxy"]:
                            W[m].append(np.concatenate([padl, r[m], np.full(H, np.nan, np.float32)]))
                        W["so"].append(np.concatenate([padl, r["so"], (av[si:si + H] < 0.5).astype(np.float32)]))
                        W["kf"].append(np.concatenate([padl, r["kf"], np.zeros(H, np.float32)]))
                        W["blk"].append(np.concatenate([padl, r["blk"][:n + H]]))
                        W["catrel"].append(np.concatenate([padl, r["catrel"][:n + H]]))
                        if DROP:
                            W["drop"].append(np.concatenate([padl, r["drop"][:n + H]]))
                        lead_of.append(0)
                        shop_of.append(shop)
        idx = list(range(n0, len(lead_of)))
        if len(idx) > 15000:
            keep_s = set(rng.choice(idx, 15000, replace=False).tolist())
            keep = [i for i in range(len(lead_of)) if i < n0 or i in keep_s]
            W = {m: [v[i] for i in keep] for m, v in W.items()}
            lead_of, shop_of = [lead_of[i] for i in keep], [shop_of[i] for i in keep]
        print(f"  {shop}: {sum(1 for x in shop_of if x == shop)} windows", flush=True)
    np.savez_compressed(f"{HERE}/windows_lead{TAG}.npz", lead=np.array(lead_of), shop=np.array(shop_of),
                        **{m: np.stack(v).astype(np.float32) for m, v in W.items()})
    print(f"windows: {len(lead_of)}")


def sonly_cases(src, dst):
    """the same test cases (shop, colour, season, lead, actual) with the season-only input."""
    import pandas as pd
    A = demandize(pd.read_pickle(os.environ.get("ARRAYS", f"{HERE}/timestral_bwe_arrays.pkl")))
    out, kx, tb = [], {}, {}
    for c in pickle.load(gzip.open(src, "rb")):
        S = A[c["shop"]]
        if c["shop"] not in kx:
            kx[c["shop"]] = {k: i for i, k in enumerate(S["keys"])}
            tb[c["shop"]] = sib_tables(S) if PROXY == "style" else None
        k_i = kx[c["shop"]][c["key"]]
        name, year = c["season"].split()
        st = {n: x for n, x, e in SEASONS[c["shop"]]}[name]
        si = int(np.searchsorted(S["weeks"], monday_on_or_after(f"{year}-{st}")))
        o = si - c["lead"]
        av, kf = avail_rows(S, k_i, S["weeks"][o])
        r = sonly_rows(S, k_i, si, o, c["Hs"], av, kf, tb[c["shop"]])
        out.append({**{k: c[k] for k in ["shop", "key", "season", "lead", "Hs", "actual", "actual_units"]},
                    "hz": c["Hs"], "sum_from": 0, **r})
    with gzip.open(dst, "wb") as fh:
        pickle.dump(out, fh)
    print(f"season-only cases: {len(out)} → {dst}")


def sonly_test_v2(dst):
    """round 11 full test (owner 2026-10-31): the old test's colours PLUS every colour first sold from 26 weeks before
    the season to its end (new colours, the 2026 collections) — the old test only had colours already selling
    26 weeks before. Same seasons, leads, actual (stock-corrected demand); season-only input; "new" marks them."""
    import pandas as pd
    A = demandize(pd.read_pickle(os.environ.get("ARRAYS", f"{HERE}/timestral_bwe_arrays.pkl")))
    lo_s, hi_s = pd.Timestamp(TEST_FROM), pd.Timestamp(TEST_TO)
    out = []
    for shop, S in A.items():
        weeks = S["weeks"]
        tabs = sib_tables(S) if PROXY == "style" else None
        for name, st, en in SEASONS[shop]:
            for year in (2025, 2026):
                start = pd.Timestamp(f"{year}-{st}")
                if not (lo_s <= start <= hi_s):
                    continue
                end = pd.Timestamp(f"{year + (1 if en < st else 0)}-{en}") + pd.Timedelta(days=1)
                s0, s1 = monday_on_or_after(start), monday_on_or_after(end)
                if s1 > weeks[-1]:
                    continue
                si, Hs = int(np.searchsorted(weeks, s0)), (s1 - s0).days // 7
                o26 = si - MAXLEAD
                for k_i, key in enumerate(S["keys"]):
                    u = S["units"][k_i]
                    sold = np.nonzero(u > 0)[0]
                    if not len(sold):
                        continue
                    f = int(sold[0])
                    old = f < o26 and o26 - f >= 3 and u[max(0, o26 - 52):o26].sum() >= 5
                    new = o26 <= f < si + Hs
                    if not (old or new):
                        continue
                    av_s, _ = avail_rows(S, k_i, s0)
                    actual = float((u[si:si + Hs] / np.maximum(av_s[si:si + Hs], 0.3)).sum())
                    for lead in LEADS_TEST:
                        o = si - lead
                        av, kf = avail_rows(S, k_i, weeks[o])
                        r = sonly_rows(S, k_i, si, o, Hs, av, kf, tabs)
                        out.append({"shop": shop, "key": key, "season": f"{name} {year}", "lead": lead, "Hs": Hs, "hz": Hs,
                                    "actual": actual, "actual_units": float(u[si:si + Hs].sum()), "sum_from": 0,
                                    "new": bool(new), **r})
        print(f"  test v2 {shop}: {len(out)} so far", flush=True)
    with gzip.open(dst, "wb") as fh:
        pickle.dump(out, fh)
    print(f"test v2 cases: {len(out)} ({sum(c['new'] for c in out)} new-colour) → {dst}")


LADDER = {"XXXS": 0, "XXS": 1, "2XS": 1, "XS": 2, "S": 3, "M": 4, "L": 5, "XL": 6, "XXL": 7, "2XL": 7, "XXXL": 8, "3XL": 8, "4XL": 9}


def size_pos(z):
    """canonical ladder position of a size label, None when it is not a size (colour, one size)."""
    z = str(z).strip().upper().replace(" ", "")
    if z in LADDER:
        return float(LADDER[z])
    for sep in ("/", "-"):
        if sep in z:
            a, b = z.split(sep, 1)
            if a in LADDER and b in LADDER:
                return (LADDER[a] + LADDER[b]) / 2
    try:
        return 100.0 + float(z)
    except ValueError:
        return None


def size_vec(S, key, cut_i):
    """the size curve as 16 points along the ladder (app/engine/curveVector method), L2-normalized; None if < 2 sizes."""
    if key not in S["size_units"].index.get_level_values(0):
        return None
    su, inst = S["size_units"].loc[key], S["inst"].loc[key].fillna(0)
    pos = [size_pos(z) for z in su.index]
    if len(pos) < 2 or any(p is None for p in pos):
        return None
    ok = inst.values[:, :cut_i] >= 0.3
    rate = (su.values[:, :cut_i] * ok).sum(1) / np.maximum((inst.values[:, :cut_i] * ok).sum(1), 1e-9)
    if rate.sum() <= 0:
        return None
    sh = (rate / rate.sum())[np.argsort(pos)]
    v = np.interp(np.linspace(0, 1, 16), np.linspace(0, 1, len(sh)), sh)
    return v / max(np.linalg.norm(v), 1e-9)


def sales_vec(u, cut_i):
    """the sales curve over the 52 weeks before cut_i: 13 four-week buckets as shares, L2-normalized; None if < 10 units."""
    if cut_i < 52 or u[cut_i - 52:cut_i].sum() < 10:
        return None
    v = u[cut_i - 52:cut_i].reshape(13, 4).sum(1)
    v = v / v.sum()
    return v / np.linalg.norm(v)


def demandize(A):
    """DEMAND=1: per colour demand = units ÷ max(availability, 0.3) (size shares from before BOUNDARY), and the
    sib (same title + type) / cat (same type) rows rebuilt from it, minus the colour itself."""
    import pandas as pd
    R = os.environ.get("RESEARCH", "")
    bnd = pd.Timestamp(BOUNDARY)
    for shop, S in A.items():
        dd = "data" if shop == "store-a" else f"data_{shop}"
        st = pd.read_csv(f"{R}/{dd}/series.csv", dtype={"product_id": str, "scope": str}, keep_default_na=False)
        meta = {f"{r.product_id}|{r.scope}": (str(r.title).strip().lower(), str(r.product_type).strip().lower()) for r in st.itertuples()}
        keys = list(S["keys"])
        D = np.zeros_like(S["units"], dtype=np.float32)
        for k_i in range(len(keys)):
            av, _ = avail_rows(S, k_i, bnd)
            D[k_i] = S["units"][k_i] / np.maximum(av, 0.3)
        def grp(lbl):
            out = np.zeros_like(D)
            for g in set(lbl):
                m = lbl == g
                out[m] = D[m].sum(0) - D[m]
            return out
        typ = np.array([meta.get(k, ("", ""))[1] for k in keys], dtype=object)
        if TYPEFILL:
            nt = typ == ""
            typ = fill_types([meta.get(k, ("", ""))[0] for k in keys], typ)
            print(f"  typefill {shop}: {int(nt.sum())} empty → {int((typ[nt] != '').sum())} filled", flush=True)
        typ = typ.astype(str)
        mod = np.array([f"{meta.get(k, ('', ''))[0]}|{t}" for k, t in zip(keys, typ)])
        S["demand"], S["cat"], S["sib"], S["type"], S["model"] = D, grp(typ), grp(mod), typ, mod
        S["first"] = np.array([int(np.nonzero(r > 0)[0][0]) if (r > 0).any() else 10 ** 6 for r in S["units"]])
        print(f"  demand rows {shop}: units {S['units'].sum():,.0f} → demand {D.sum():,.0f}", flush=True)
    return A


def sib_tables(S):
    """{kind: {anchor: {k_i: [sibling k_i]}}}: the NSIB most similar curves at every ANCHOR-week anchor (data before it)."""
    T = len(S["weeks"])
    out = {"s": {}, "z": {}}
    for a in range(52, T + 1, ANCHOR):
        for kind in out:
            idx, vecs = [], []
            for k_i, key in enumerate(S["keys"]):
                u = S.get("demand", S["units"])[k_i]
                if S["units"][k_i][:a].sum() < 10:
                    continue
                v = sales_vec(u, a) if kind == "s" else size_vec(S, key, a)
                if v is not None:
                    idx.append(k_i)
                    vecs.append(v)
            if len(idx) < 2:
                out[kind][a] = {}
                continue
            M = np.stack(vecs)
            sim = M @ M.T
            np.fill_diagonal(sim, -9)
            if kind == "z" and "type" in S:
                # owner 2026-10-28: a size curve says who buys it, not when — size siblings only from the SAME category
                # (no type, or < 3 others in it → no siblings, so the category row stands in)
                ty = S["type"][np.array(idx)]
                same = (ty[:, None] == ty[None, :]) & (ty[:, None] != "")
                sim = np.where(same, sim, -9)
            top = np.argsort(-sim, axis=1)[:, :NSIB]
            res = {}
            for i in range(len(idx)):
                js = [idx[j] for j in top[i] if sim[i, j] > -9]
                if kind != "z" or len(js) >= 3:
                    res[idx[i]] = js
            out[kind][a] = res
    return out


def csib_rows(S, tabs, k_i, lo, end, c):
    """past + known sibling rows over [lo, end) for origin c (both kinds); category when there are no siblings."""
    T = len(S["weeks"])
    a = max([x for x in range(52, T + 1, ANCHOR) if x <= c], default=None)
    own = S["units"][k_i]
    res = {}
    for kind in ("s", "z"):
        sibs = tabs[kind].get(a, {}).get(k_i, []) if a is not None else []
        U = S.get("demand", S["units"])
        series = [U[j] for j in sibs] or [S["cat"][k_i] + U[k_i]]
        past, ly = [], []
        for x in series:
            base = x[max(0, c - 52):c].mean() if c > 0 else 0
            if base <= 0:
                continue
            past.append(x[lo:c] / base)
            sold = np.nonzero(x > 0)[0]
            first = int(sold[0]) if len(sold) else 0
            # last year only where the sibling already existed (not launched yet ≠ sold nothing)
            ly.append(np.array([x[t - 52] / base if t - 52 >= first else np.nan for t in range(lo, end)], np.float32))
        n = c - lo
        res[f"csib_{kind}"] = (np.mean(past, 0) if past else np.ones(n)).clip(0, 20).astype(np.float32)
        with np.errstate(all="ignore"):
            lym = np.nanmean(np.stack(ly), 0) if ly else np.full(end - lo, np.nan)
        res[f"csib_{kind}_ly"] = np.nan_to_num(lym, nan=1.0).clip(0, 10).astype(np.float32)
    return res


def augment(src, dst):
    """add the curve-sibling rows to an existing test-case file (same cases, same order)."""
    import pandas as pd
    A = pd.read_pickle(os.environ.get("ARRAYS", f"{HERE}/timestral_bwe_arrays.pkl"))
    if DEMAND:
        A = demandize(A)
    names_of = {}
    cases = pickle.load(gzip.open(src, "rb"))
    tabs, kx = {}, {}
    for c in cases:
        S = A[c["shop"]]
        if c["shop"] not in tabs:
            tabs[c["shop"]] = sib_tables(S)
            kx[c["shop"]] = {k: i for i, k in enumerate(S["keys"])}
            print(f"  sibling tables {c['shop']}", flush=True)
        name, year = c["season"].split()
        st = {n: x for n, x, e in SEASONS[c["shop"]]}[name]
        si = int(np.searchsorted(S["weeks"], monday_on_or_after(f"{year}-{st}")))
        oi = si - c["lead"]
        lo = oi - len(c["units"])
        assert np.allclose(S["units"][kx[c["shop"]][c["key"]]][lo:oi], c["units"])
        k_i = kx[c["shop"]][c["key"]]
        c.update(csib_rows(S, tabs[c["shop"]], k_i, lo, oi + c["hz"], oi))
        if DEMAND:
            if c["shop"] not in names_of:
                names_of[c["shop"]] = season_names(c["shop"], S["weeks"])
            x = S["cat"][k_i] + S["demand"][k_i]
            c["cat"] = S["cat"][k_i][lo:oi].astype(np.float32)
            c["sib"] = S["sib"][k_i][lo:oi].astype(np.float32)
            c["cat_ly"] = ly_row(x, lo, oi + c["hz"], oi)
            c["cat_sidx"] = season_index(x, names_of[c["shop"]], lo, oi + c["hz"], oi)
            c.pop("cat_srel", None)
    with gzip.open(dst, "wb") as fh:
        pickle.dump(cases, fh)
    print(f"augmented {len(cases)} cases → {dst}")


# ---------------------------------------------------------------- prep (laptop)
def prep():
    import pandas as pd
    A = pd.read_pickle(os.environ.get("ARRAYS", f"{HERE}/timestral_bwe_arrays.pkl"))
    if DEMAND:
        A = demandize(A)
    rng = np.random.default_rng(0)
    W = {n: [] for n in ["units", *PAST, *ALL_KNOWN, *CSIB_ROWS]}
    lead_of, shop_of = [], []
    cap = {"store-a": 15000}
    bnd = pd.Timestamp(BOUNDARY)
    for shop, S in A.items():
        b = int(np.searchsorted(S["weeks"], bnd))
        n0 = len(lead_of)
        names = season_names(shop, S["weeks"])
        tabs = sib_tables(S)
        for k_i in range(len(S["keys"])):
            u = S["units"][k_i]
            sold = np.nonzero(u[:b] > 0)[0]
            if not len(sold) or u[:b].sum() < 5:
                continue
            f = sold[0]
            av, kf = avail_rows(S, k_i, bnd)
            for c in range(f + 8, b - H + 1, 2):
                lead = int(rng.choice(LEADS_TRAIN))
                if c + lead + H > b:
                    lead = max(x for x in LEADS_TRAIN if c + x + H <= b)
                lo = max(f, c - L)
                if u[lo:c].sum() < 3:
                    continue
                end = c + lead + H
                padl = np.full(L - (c - lo), np.nan, np.float32)
                padr = np.full(SPAN - (L + lead + H), np.nan, np.float32)
                full = lambda x: np.concatenate([padl, np.asarray(x[lo:end], np.float32), padr])
                W["units"].append(full(u))
                for n in PAST:
                    W[n].append(full(S[n][k_i]))
                W["so"].append(full((av < 0.5).astype(np.float32)))
                W["kf"].append(full(kf))
                W["woy_sin"].append(full(S["woy_sin"]))
                W["woy_cos"].append(full(S["woy_cos"]))
                W["age"].append(np.concatenate([padl, np.arange(lo - f, end - f, dtype=np.float32), padr]))
                own = S.get("demand", S["units"])[k_i]
                W["cat_ly"].append(np.concatenate([padl, ly_row(S["cat"][k_i] + own, lo, end, c), padr]))
                W["own_ly"].append(np.concatenate([padl, ly_row(u, lo, end, c, first=f), padr]))
                for r, v in season_rows(names, lo, end).items():
                    W[r].append(np.concatenate([padl, v, padr]))
                W["cat_sidx"].append(np.concatenate([padl, season_index(S["cat"][k_i] + own, names, lo, end, c), padr]))
                cr = csib_rows(S, tabs, k_i, lo, end, c)
                for kind in ("s", "z"):
                    W[f"csib_{kind}"].append(np.concatenate([padl, cr[f"csib_{kind}"], np.full(end - c, np.nan, np.float32), padr]))
                    W[f"csib_{kind}_ly"].append(np.concatenate([padl, cr[f"csib_{kind}_ly"], padr]))
                lead_of.append(lead)
                shop_of.append(shop)
        idx = list(range(n0, len(lead_of)))
        if shop in cap and len(idx) > cap[shop]:
            drop = set(rng.choice(idx, len(idx) - cap[shop], replace=False).tolist())
            keep = [i for i in range(len(lead_of)) if i not in drop]
            W = {n: [v[i] for i in keep] for n, v in W.items()}
            lead_of = [lead_of[i] for i in keep]
            shop_of = [shop_of[i] for i in keep]
        print(f"  {shop}: {sum(1 for s in shop_of if s == shop)} windows", flush=True)
    np.savez_compressed(f"{HERE}/windows_lead{TAG}.npz", lead=np.array(lead_of), shop=np.array(shop_of),
                        **{n: np.stack(v).astype(np.float32) for n, v in W.items()})
    print(f"windows: {len(lead_of)} (targets before {BOUNDARY}); leads {dict(zip(*np.unique(lead_of, return_counts=True)))}")
    if os.environ.get("SKIP_TEST"):
        return
    cases = make_test(A)
    with gzip.open(f"{HERE}/test_cases{TAG}.pkl.gz", "wb") as fh:
        pickle.dump(cases, fh)
    print(f"test cases: {len(cases)}")


def monday_on_or_after(d):
    import pandas as pd
    d = pd.Timestamp(d).normalize()
    return d + pd.Timedelta(days=(7 - d.weekday()) % 7)


def make_test(A):
    import pandas as pd
    cases = []
    lo_s, hi_s = pd.Timestamp(TEST_FROM), pd.Timestamp(TEST_TO)
    for shop, S in A.items():
        weeks = S["weeks"]
        names = season_names(shop, weeks)
        for name, st, en in SEASONS[shop]:
            for year in (2025, 2026):
                start = pd.Timestamp(f"{year}-{st}")
                if not (lo_s <= start <= hi_s):
                    continue
                end = pd.Timestamp(f"{year + (1 if en < st else 0)}-{en}") + pd.Timedelta(days=1)
                s0, s1 = monday_on_or_after(start), monday_on_or_after(end)
                if s1 > weeks[-1]:
                    continue
                si, Hs = int(np.searchsorted(weeks, s0)), (s1 - s0).days // 7
                for k_i, key in enumerate(S["keys"]):
                    u = S["units"][k_i]
                    sold = np.nonzero(u[:si - MAXLEAD] > 0)[0]  # alive at the earliest origin: same colours at every lead
                    if not len(sold):
                        continue
                    f = sold[0]
                    o26 = si - MAXLEAD
                    if o26 - f < 3 or u[max(0, o26 - 52):o26].sum() < 5:
                        continue
                    av_s, _ = avail_rows(S, k_i, s0)
                    actual = float((u[si:si + Hs] / np.maximum(av_s[si:si + Hs], 0.3)).sum())
                    for lead in LEADS_TEST:
                        oi = si - lead
                        lo = max(f, oi - L)
                        av, kf = avail_rows(S, k_i, weeks[oi])
                        hz = lead + Hs
                        c = {"shop": shop, "key": key, "season": f"{name} {year}", "lead": lead, "Hs": Hs, "hz": hz,
                             "actual": actual, "actual_units": float(u[si:si + Hs].sum()),
                             "units": u[lo:oi].astype(np.float32),
                             "so": (av[lo:oi] < 0.5).astype(np.float32), "kf": kf[lo:oi],
                             "woy_sin": S["woy_sin"][lo:oi + hz], "woy_cos": S["woy_cos"][lo:oi + hz],
                             "age": np.arange(lo - f, oi + hz - f, dtype=np.float32),
                             "cat_ly": ly_row(S["cat"][k_i] + u, lo, oi + hz, oi),
                             "own_ly": ly_row(u, lo, oi + hz, oi, first=f),
                             **season_rows(names, lo, oi + hz),
                             "cat_sidx": season_index(S["cat"][k_i] + u, names, lo, oi + hz, oi)}
                        for n in PAST:
                            c[n] = S[n][k_i][lo:oi].astype(np.float32)
                        cases.append(c)
        print(f"  test {shop}: {len(cases)} so far", flush=True)
    return cases


# ---------------------------------------------------------------- model helpers (VM)
def fresh_model():
    model = B.load("cpu")
    _, params = B.add_lora(model, RANK, 2 * RANK)
    return model, params


def batch_series(units, past, known, hz, torch, cens=None):
    from t0 import TimeSeries
    return B.build_series(units, past, known, hz, torch, TimeSeries, cens=cens if CENS else None)


def train(arm):
    import torch
    from t0.rollout import prepare_rollout_buffer
    from t0.types import round_up
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count())
    z = np.load(f"{HERE}/windows_lead{TAG}.npz", allow_pickle=False)
    past_n, known_n = rows_for_arm(arm)
    lead = z["lead"]
    Wd = {n: z[n] for n in ["units", *past_n, *known_n] if n != "cat_srel"}
    if "cat_srel" in known_n:
        Wd["cat_srel"] = srel_rows(z["cat_sidx"], L)
    if INIT:
        model, _ = B.load_lora(INIT)
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        model, params = fresh_model()
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    lv = torch.tensor(LEVELS)
    rng = np.random.default_rng(0)
    buckets = {x: np.flatnonzero(lead == x) for x in LEADS_TRAIN}
    if MASK and not CLOSS:  # only windows whose target season was in stock at least half the weeks (like the test)
        so = z["so"]
        instock = np.array([np.nanmean(so[i, L + lead[i]:L + lead[i] + H] < 0.5) for i in range(len(lead))])
        buckets = {x: b[instock[b] >= 0.5] for x, b in buckets.items()}
        print(f"mask: {sum(len(b) for b in buckets.values())} of {len(lead)} windows kept", flush=True)
    wmean = torch.tensor(B.mean_w(LEVELS), dtype=torch.float32)
    weight = {x: None for x in LEADS_TRAIN}
    if OVERSAMPLE:
        sidx = z["cat_sidx"]
        recent = np.nanmean(sidx[:, L - 13:L], axis=1)
        for x in LEADS_TRAIN:
            idx = buckets[x]
            tgt = np.nanmean(sidx[idx, L + x:L + x + H], axis=1)
            ch = np.abs(np.log(np.maximum(tgt, 0.02) / np.maximum(recent[idx], 0.02)))
            w = 1 + OVERSAMPLE * np.minimum(np.nan_to_num(ch), 3) / 3
            weight[x] = w / w.sum()
        print(f"oversampling ×{OVERSAMPLE}: share of draws on windows whose season changes ≥2× = "
              f"{np.mean([weight[x][np.nan_to_num(np.abs(np.log(np.maximum(np.nanmean(sidx[buckets[x], L + x:L + x + H], 1), 0.02) / np.maximum(recent[buckets[x]], 0.02)))) > 0.69].sum() for x in LEADS_TRAIN]):.0%}", flush=True)
    if BOOST != 1 and "drop" in z:
        hit = (np.nansum(z["drop"][:, L:L + H], 1) > 0) | (np.nan_to_num(z["proxy"][:, L - 1]) > 0)
        for x in LEADS_TRAIN:
            if not len(buckets[x]):
                continue
            w = (weight[x] if weight[x] is not None else np.full(len(buckets[x]), 1 / len(buckets[x]))) * (1 + (BOOST - 1) * hit[buckets[x]])
            weight[x] = w / w.sum()
        print(f"boost ×{BOOST}: {hit.mean():.0%} of windows are drop/new-colour; their share of draws = "
              f"{np.mean([weight[x][hit[buckets[x]]].sum() for x in LEADS_TRAIN if len(buckets[x])]):.0%}", flush=True)
    p_b = np.array([len(buckets[x]) for x in LEADS_TRAIN], float)
    p_b /= p_b.sum()
    cw = round_up(L, model.patch_size)
    t0, losses = time.time(), []
    print(f"[{arm}{RUN}] {len(lead)} windows · known {known_n} · {STEPS} steps · init {INIT or 'base t0-gamma'}", flush=True)
    for step in range(1, STEPS + 1):
        ld = int(rng.choice(LEADS_TRAIN, p=p_b))
        idx = rng.choice(buckets[ld], BS, replace=False, p=weight[ld])
        hz = ld + H
        u = Wd["units"][idx][:, :L]
        y = Wd["units"][idx][:, L:L + hz]
        p = np.stack([Wd[n][idx][:, :L] for n in past_n], 1)
        k = np.stack([Wd[n][idx][:, :L + hz] for n in known_n], 1)
        for i, n in enumerate(known_n):
            if n in FUTURE_FILL:
                k[:, i, L:] = FUTURE_FILL[n]
        ts = batch_series(u, p, k, hz, torch, cens=Wd["so"][idx][:, :L])
        model.train()
        buf = prepare_rollout_buffer(ts, hz, L, model.patch_size)
        hp = round_up(hz, model.patch_size)
        q = model._predict_step(buf.time_slice(0, cw + hp), hp)[:BS, :hz]
        yt = torch.from_numpy(y)
        s = torch.from_numpy(np.nanmean(u, 1)).clamp(min=0.5)
        if MASK:  # a week with no stock says nothing about demand: leave it out of the loss
            m = torch.from_numpy((Wd["so"][idx][:, L:L + hz] < 0.5).astype(np.float32))
        else:
            m = torch.ones_like(yt)
        d = yt[..., None] - q
        if CLOSS:  # in-stock weeks: normal loss; sold-out weeks: only a forecast BELOW what sold is wrong
            out = torch.from_numpy((Wd["so"][idx][:, L:L + hz] >= 0.5).astype(np.float32)) * CLOSSW
            if CLOSS_SEASON:  # off-season (the category sells below its average then): sold out = real low demand
                ins = torch.from_numpy((np.nan_to_num(Wd["catrel"][idx][:, L:L + hz], nan=1.0) >= 1.0).astype(np.float32))
                m = m + out * (1 - ins) / max(CLOSSW, 1e-9)
                out = out * ins
            cl = (lv * d.clamp(min=0)).mean(2)
            weekly = ((torch.maximum(lv * d, (lv - 1) * d).mean(2) * m) + cl * out).sum(1) / (m + out).sum(1).clamp(min=1)
        else:
            weekly = (torch.maximum(lv * d, (lv - 1) * d).mean(2) * m).sum(1) / m.sum(1).clamp(min=1)
        mt = m[:, ld:]
        ds = (yt[:, ld:] * mt).sum(1)[..., None] - (q[:, ld:] * mt[..., None]).sum(1)  # the target season, in-stock weeks
        season = torch.maximum(lv * ds, (lv - 1) * ds).mean(1) / H
        loss = (weekly + season) / s
        if MEANLOSS:  # the season total of the readout (mean × K) should equal what sold
            mean_sum = K * ((q[:, ld:] @ wmean) * mt).sum(1)
            loss = loss + torch.nn.functional.huber_loss(mean_sum / (s * H), (yt[:, ld:] * mt).sum(1) / (s * H), reduction="none", delta=1.0)
        loss = loss.mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
        if step <= 3 or step % 25 == 0:
            print(f"[{arm}] step {step} (lead {ld}): loss {np.mean(losses[-25:]):.4f} · {time.time() - t0:.0f}s", flush=True)
    B.save_lora(B.lora_state(model), f"{HERE}/adapter_{arm}{TAG}{RUN}", {"tag": f"lead_{arm}", "groups": known_n + past_n, "stage": "lead",
                                                                "steps": STEPS, "readout": ["mean", K]}, torch)


def evaluate(arm, model_dir=None):
    import pandas as pd
    import torch
    torch.set_num_threads(os.cpu_count())
    if arm == "base":
        model, _ = B.load_lora(model_dir)
    else:
        model, _ = B.load_lora(f"{HERE}/adapter_{arm}{TAG}{RUN}")
    past_n, known_n = rows_for_arm(arm)
    cases = pickle.load(gzip.open(f"{HERE}/test_cases{TAG}.pkl.gz", "rb"))
    if os.environ.get("EVAL_LIMIT"):  # smoke test: every n-th case
        cases = cases[:: max(1, len(cases) // int(os.environ["EVAL_LIMIT"]))]
    w = B.mean_w(LEVELS)
    by = {}
    for i, c in enumerate(cases):
        by.setdefault(c["hz"], []).append(i)
        if "cat_srel" in known_n and "cat_srel" not in c:
            c["cat_srel"] = srel_rows(np.asarray(c["cat_sidx"], np.float32)[None], len(c["units"]))[0]
    out, t0, done = [], time.time(), 0
    for hz, ids in sorted(by.items()):
        for j in range(0, len(ids), 64):
            chunk = [cases[i] for i in ids[j:j + 64]]
            T = max(len(c["units"]) for c in chunk)
            padl = lambda a, t: np.concatenate([np.full(t - len(a), np.nan, np.float32), np.asarray(a, np.float32)])
            u = np.stack([padl(c["units"], T) for c in chunk])
            p = np.stack([np.stack([padl(c[n], T) for n in past_n]) for c in chunk])
            ks = []
            for c in chunk:
                rows = []
                for n in known_n:
                    x = np.concatenate([c[n], np.full(hz, FUTURE_FILL[n], np.float32)]) if n in FUTURE_FILL else c[n]
                    rows.append(padl(x, T + hz))
                ks.append(np.stack(rows))
            ts = batch_series(u, p, np.stack(ks), hz, torch, cens=np.stack([padl(c["so"], T) for c in chunk]))
            with torch.inference_mode():
                flags = dict(symmetric_averaging=True, clip_nonnegative=True, snap_integers=True, trim_leading_missing=True) if os.environ.get("FLAGS") == "1" else {}  # the docs' recommended settings
                q = model.predict(ts, hz, LEVELS, **flags).quantiles.float().cpu().numpy()  # [B, hz, 21]
            wk = K * (q @ w)
            for c, row in zip(chunk, wk):
                out.append({"shop": c["shop"], "key": c["key"], "season": c["season"], "lead": c["lead"], "Hs": c["Hs"],
                            "actual": c["actual"], "actual_units": c["actual_units"], "new": c.get("new", False), "forecast": float(row[c.get("sum_from", c["lead"]):].sum())})
            done += len(chunk)
        print(f"[eval {arm}] hz {hz}: {done}/{len(cases)} · {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(out).to_csv(f"{HERE}/eval_{arm}{TAG}{RUN}{os.environ.get('EVAL_TAG', '')}.csv", index=False)
    print(f"[eval {arm}] done · {time.time() - t0:.0f}s", flush=True)


def score():
    import pandas as pd
    A = pd.read_pickle(os.environ.get("ARRAYS", f"{HERE}/timestral_bwe_arrays.pkl"))
    types = {}
    for shop, S in A.items():
        st = None
        for d in ("data" if shop == "store-a" else f"data_{shop}",):
            p = os.environ.get("RESEARCH", "") + f"/{d}/series.csv"
            if os.path.exists(p):
                st = pd.read_csv(p, dtype={"product_id": str, "scope": str}, keep_default_na=False)
        if st is not None:
            for r in st.itertuples():
                types[(shop, f"{r.product_id}|{r.scope}")] = str(r.product_type).strip().lower()
    arms = [a for a in ARMS if os.path.exists(f"{HERE}/eval_{a}.csv")]
    e = lambda f, a: f"{(f - a).abs().sum() / a.sum():.1%} ({(f - a).sum() / a.sum():+.0%})"
    d = None
    for a in arms:
        x = pd.read_csv(f"{HERE}/eval_{a}.csv").rename(columns={"forecast": f"f_{a}"})
        d = x if d is None else d.merge(x[["shop", "key", "season", "lead", f"f_{a}"]], on=["shop", "key", "season", "lead"])
    d["type"] = [types.get((s, k), "") for s, k in zip(d.shop, d.key)]
    lines = [f"# t0 LoRA lead-time experiment — {len(d)} forecasts, {d.groupby(['shop', 'key', 'season']).ngroups} colour seasons", ""]
    for title, g in [("All shops, all seasons", d), ("Winter 25/26", d[d.season.str.startswith("Winter")]),
                     ("Spring 26", d[d.season.str.startswith("Spring")]), ("Summer 26", d[d.season.str.startswith("Summer")]),
                     ("Swimwear (store A), all seasons", d[d.type == "swimwear"]),
                     ("Swimwear, Winter 25/26", d[(d.type == "swimwear") & d.season.str.startswith("Winter")]),
                     ("Swimwear, Spring 26", d[(d.type == "swimwear") & d.season.str.startswith("Spring")])]:
        if not len(g):
            continue
        lines += [f"## {title} — {g.groupby(['shop', 'key', 'season']).ngroups} colour seasons, actual {g[g.lead == 0].actual.sum():,.0f}", "",
                  "| weeks before | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
        for ld in LEADS_TEST:
            h = g[g.lead == ld]
            lines.append(f"| {ld} | " + " | ".join(e(h[f"f_{a}"], h.actual) for a in arms) + " |")
        lines.append("")
    text = "\n".join(lines)
    open(f"{HERE}/lead_report.md", "w").write(text + "\n")
    d.to_csv(f"{HERE}/lead_all.csv", index=False)
    print(text)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "eval", "score", "augment", "prep_sonly", "sonly_cases", "test_v2"])
    ap.add_argument("--arm", default="catly", choices=list(ARMS))
    ap.add_argument("--model-dir", default="")
    ap.add_argument("--src", default="")
    ap.add_argument("--dst", default="")
    a = ap.parse_args()
    if a.cmd == "prep":
        prep()
    elif a.cmd == "train":
        train(a.arm)
    elif a.cmd == "prep_sonly":
        prep_sonly()
    elif a.cmd == "sonly_cases":
        sonly_cases(a.src, a.dst)
    elif a.cmd == "test_v2":
        sonly_test_v2(a.dst)
    elif a.cmd == "augment":
        augment(a.src, a.dst)
    elif a.cmd == "eval":
        evaluate(a.arm, a.model_dir)
    else:
        score()
