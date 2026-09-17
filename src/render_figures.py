"""
Render every step-3 figure to `figures/` as PNG, for review outside a notebook.

    python -m src.render_figures

The notebook is where the figures are *read*; this is where they are
*regenerated* on demand - after a change to `plots.py`, or to look at them all
side by side. `figures/` is gitignored: the report embeds its own copies.

The last figure is on synthetic data whose pattern is known by construction,
so the plotting tools have a right answer to be checked against - the ACF of a
pure weekly sine is a cosine, and that is what should appear.
"""

from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")  # file output only; never opens a window

import matplotlib.pyplot as plt  # noqa: E402

from src import plots  # noqa: E402
from src.step1_problem import PROJECT_ROOT, STUDY_ITEMS, Config  # noqa: E402
from src.step2_data import load_panel  # noqa: E402
from src.step3_explore import classification_cutoff, series_stats  # noqa: E402
from src.validate import _synthetic_panel  # noqa: E402

OUT = PROJECT_ROOT / "figures"


def render_all(cfg: Config | None = None) -> list[str]:
    """Render the full set. Returns the filenames written."""
    cfg = cfg or Config(item_ids=STUDY_ITEMS)
    OUT.mkdir(exist_ok=True)
    plots.use_style()

    item = cfg.item_ids[0]
    df = load_panel(cfg, verbose=False)
    stats = series_stats(df, cfg)
    cutoff = classification_cutoff(df, cfg)
    top_id, top_store = stats.iloc[0]["id"], stats.iloc[0]["store_id"]
    written: list[str] = []

    def save(fig: plt.Figure, name: str) -> None:
        fig.savefig(OUT / name, bbox_inches="tight")
        plt.close(fig)
        written.append(name)

    # 1. every store, with the held-out window marked and outliers flagged
    save(
        plots.plot_store_grid(df, item, stats, holdout_start=cutoff, flag_outliers=True),
        "1_grid.png",
    )

    # 2. demand classification
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    plots.plot_demand_class_map(stats, highlight_item=item, item_id=item, ax=ax)
    fig.tight_layout()
    save(fig, "2_classmap.png")

    # 3a. seasonality at the busiest store, all years
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.9))
    plots.plot_seasonality(df, top_id, axes=axes)
    fig.suptitle(f"{item} at {top_store}", y=1.05, fontsize=11)
    fig.tight_layout()
    save(fig, "3a_season_one_store.png")

    # 3b. seasonality across every store, each indexed to its own mean
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 2.9))
    plots.plot_seasonality_by_store(df, item, axes=axes)
    fig.suptitle(f"{item} — seasonal shape across stores", y=1.05, fontsize=11)
    fig.tight_layout()
    save(fig, "3b_season_all_stores.png")

    # 4. autocorrelation and year-on-year overlay
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
    plots.plot_acf(
        df, top_id, ax=axes[0], title=f"{item} at {top_store} — autocorrelation"
    )
    plots.plot_year_overlay(df, top_id, ax=axes[1])
    fig.tight_layout()
    save(fig, "4_acf_overlay.png")

    # 5. relationships between variables
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.4))
    plots.plot_price_relationship(df, item, ax=axes[0])
    plots.plot_event_effect(df, item, ax=axes[1])
    plots.plot_zero_rate(stats, item_id=item, ax=axes[2])
    fig.tight_layout()
    save(fig, "5_relationships.png")

    # 6. SNAP benefit days
    fig, ax = plt.subplots(figsize=(9, 3))
    plots.plot_snap_effect(df, item, stats, ax=ax)
    fig.tight_layout()
    save(fig, "6_snap.png")

    # 7. synthetic - the tools checked against a known answer
    syn = _synthetic_panel(n_series=3)
    sid = syn["id"].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.4))
    plots.plot_series(
        syn,
        sid,
        ax=axes[0],
        title="synthetic: level + weekly sine + slow trend + N(0, 2)",
    )
    plots.plot_acf(
        syn, sid, ax=axes[1], title="synthetic ACF — a pure weekly sine gives a cosine"
    )
    fig.tight_layout()
    save(fig, "7_synthetic.png")

    return written


def render_evaluation(cfg: Config | None = None) -> list[str]:
    """
    The step-5 figures. Runs the walk-forward (about a minute once the
    feature matrices are cached) and draws FPP's evaluation plots from it.
    """
    from src.step5_evaluate import run_walk_forward, score_folds

    cfg = cfg or Config(item_ids=STUDY_ITEMS)
    OUT.mkdir(exist_ok=True)
    plots.use_style()

    item = cfg.item_ids[0]
    df = load_panel(cfg, verbose=False)
    stats = series_stats(df, cfg)
    cutoff = classification_cutoff(df, cfg)
    origins = cfg.fold_origins(df["date"].max())
    top_id, top_store = stats.iloc[0]["id"], stats.iloc[0]["store_id"]

    predictions = run_walk_forward(df, cfg, progress=False)
    scores = score_folds(predictions)
    written: list[str] = []

    def save(fig: plt.Figure, name: str) -> None:
        fig.savefig(OUT / name, bbox_inches="tight")
        plt.close(fig)
        written.append(name)

    # 8. forecasts against actuals, busiest store (FPP §5.8)
    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    plots.plot_forecast_folds(
        predictions, df, top_id, holdout_start=cutoff, origins=origins, ax=ax
    )
    fig.tight_layout()
    save(fig, "8_forecasts_busiest_store.png")

    # 9. error by horizon (FPP §5.10)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    plots.plot_rmsse_by_horizon(predictions, ax=ax)
    fig.tight_layout()
    save(fig, "9_rmsse_by_horizon.png")

    # 10. residual diagnostics for each model at the busiest store (FPP §5.4)
    for model in ("xgboost", "ets"):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
        plots.plot_residual_diagnostics(
            predictions, top_id, model, season=cfg.season, axes=axes
        )
        fig.suptitle(
            f"{model} at {top_store} — residual diagnostics", y=1.04, fontsize=11
        )
        fig.tight_layout()
        save(fig, f"10_residuals_{model}.png")

    # 11. RMSSE by store - the study's own view
    fig, ax = plt.subplots(figsize=(10, 3.8))
    plots.plot_rmsse_by_store(scores, stats, item_id=item, ax=ax)
    fig.tight_layout()
    save(fig, "11_rmsse_by_store.png")

    return written


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # matplotlib font chatter
        names = render_all() + render_evaluation()
    print(f"wrote {len(names)} figures to {OUT}")
    for n in names:
        print("  ", n)
