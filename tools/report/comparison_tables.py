"""The head-to-head tables (the final model against three references) and the
full method x baseline matrix for the appendix, from the cache. Run from the
repo root with PYTHONPATH=. ; prints markdown tables."""
import pandas as pd
from src.step1_problem import Config
from src.step5_evaluate import _read_cached
from src.scoring import score_folds
L = dict(fold_step=1, n_folds=358)
NAME = {"seasonal_naive": "This day last week (seasonal naive)", "arma": "ARMA, no seasonality, no inputs", "arima_plain": "ARIMA, no holiday or SNAP inputs",
        "arima": "ARIMA", "ets": "Exponential smoothing (ETS)", "moving_average_28": "28-day moving average", "xgboost": "XGBoost, one per store",
        "xgboost_rel": "XGBoost, one per store, level-relative", "xgboost@pooled": "XGBoost, pooled", "xgboost_rel@pooled": "XGBoost, pooled, level-relative (final)"}
def load(item, key):
    m, pooled = (key.split("@") + [None])[:2]
    cfg = Config(item_ids=(item,), pool_by="item_id" if pooled else None, **L)
    f = _read_cached(cfg, m); assert f is not None, (item, key); return f
def per_week(f):
    sc = score_folds(f.assign(method="m")); return sc.set_index(["id", "origin"])["rmsse"]
def weekly_mae(f):
    g = f[~f["closure"].astype(bool)].groupby(["id", "origin"]).agg(fc=("forecast", "sum"), ac=("actual", "sum"))
    return float((g.fc - g.ac).abs().mean())
def pair(a, b):
    """b against a, paired on store-week: weeks won, median gain, stores won (per-store RMSSE)."""
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    ok = j.a > 0
    gain = ((j.a - j.b) / j.a * 100)[ok]
    win = float((j.b < j.a).mean())
    ps = j.groupby(level=0).mean()
    return win, float(gain.median()), int((ps.b < ps.a).sum())
def head_to_head(item, refs, final="xgboost_rel@pooled"):
    """One row per reference: its weekly error and RMSSE against the final model's, and the final model's wins."""
    fr = {k: load(item, k) for k in refs + [final]}; pw = {k: per_week(fr[k]) for k in fr}
    fe, fr_ = weekly_mae(fr[final]), float(pw[final].mean())
    rows = []
    for k in refs:
        w, _, s = pair(pw[k], pw[final])
        rows.append({"XGBoost (final) against": NAME[k], "weekly error, units": f"{weekly_mae(fr[k]):.1f} → {fe:.1f}",
                     "RMSSE": f"{float(pw[k].mean()):.2f} → {fr_:.2f}", "weeks XGBoost won": f"{w:.0%}", "stores XGBoost won": f"{s} of 10"})
    return pd.DataFrame(rows)
def matrix(item, methods, baselines, pw_all):
    rows = []
    for k in methods:
        r = {"method": NAME[k]}
        for b in baselines:
            short = {"seasonal_naive": "seasonal naive", "arma": "ARMA", "arima": "ARIMA", "moving_average_28": "28-day average"}[b]
            if b == k:
                r[f"weeks won vs: {short}"] = ""; r[f"median gain vs: {short}"] = ""; continue
            w, g, _ = pair(pw_all[b], pw_all[k])
            r[f"weeks won vs: {short}"] = f"{w:.0%}"; r[f"median gain vs: {short}"] = f"{g:+.0f}%".replace("+0%", "0%").replace("-0%", "0%")
        rows.append(r)
    return pd.DataFrame(rows)
def md(df):
    cols = list(df.columns); out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows(): out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)

print("## HEAD TO HEAD fast\n" + md(head_to_head("FOODS_3_586", ["seasonal_naive", "arma", "arima"])))
print("\n## HEAD TO HEAD slow\n" + md(head_to_head("FOODS_1_021", ["seasonal_naive", "moving_average_28", "arima"])))
all_fast = ["xgboost_rel@pooled", "xgboost@pooled", "xgboost_rel", "arima", "arima_plain", "xgboost", "ets", "arma", "moving_average_28", "seasonal_naive"]
all_slow = ["moving_average_28", "ets", "xgboost_rel@pooled", "xgboost_rel", "arima_plain", "arma", "arima", "xgboost@pooled", "xgboost", "seasonal_naive"]
pwf_all = {k: per_week(load("FOODS_3_586", k)) for k in all_fast}
pws_all = {k: per_week(load("FOODS_1_021", k)) for k in all_slow}
print("\n## MATRIX fast\n" + md(matrix("FOODS_3_586", all_fast, ["seasonal_naive", "arma", "arima"], pwf_all)))
print("\n## MATRIX slow\n" + md(matrix("FOODS_1_021", all_slow, ["seasonal_naive", "moving_average_28", "arima"], pws_all)))
