"""
Shared plotting.

Every function takes an optional `ax` and returns it, so plots compose into
grids and notebooks without this module owning any figure layout.

The design rules, so the figures stay consistent:

- The book's look. Every figure uses the style of Forecasting: Principles and
  Practice (the Pythonic edition's figures follow ggplot2's default theme). A
  light grey panel with white gridlines, no axis spines, black actuals, and
  the Okabe-Ito palette the book uses. The report's figures then read as
  continuations of the reference.
- Legends sit below the axes. Every legend is placed by `legend_below` and
  nothing is drawn inside the plotting area.
- One hue per single-series chart. Colour carries identity and never
  magnitude.
- Emphasis over enumeration. Where one series matters it is drawn in the
  series colour and everything else goes grey.
- Selective labels. Direct-label the points that carry the argument.

Layout is matplotlib's constrained layout, switched on in `use_style`, which
makes room for legends placed outside the axes. Do not call `tight_layout()`
on these figures. It replaces the layout engine and the legends get clipped.
"""

from __future__ import annotations

import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.step3_explore import ADI_CUT, CV2_CUT

# --- palette ---------------------------------------------------------------
# Sampled from the book's own figures (otexts.com/fpppy). The panel is
# rgb(229,229,229), the grid is white, text is black, ticks are mid grey.
SURFACE = "#ffffff"  # figure background
PANEL = "#e5e5e5"  # plotting area
INK = "#000000"
INK_SOFT = "#4d4d4d"
INK_MUTED = "#7f7f7f"
GRID = "#ffffff"
AXIS = "#7f7f7f"  # reference lines (zero, cut points, the holdout boundary)

# Okabe-Ito, as the book uses it. The first three carry the three models;
# the registry figures use 4 and 5 for the pooled and level-relative variants.
SERIES_1 = "#0072B2"  # blue
SERIES_2 = "#D55E00"  # vermillion
SERIES_3 = "#009E73"  # green
SERIES_4 = "#CC79A7"  # pink
SERIES_5 = "#E69F00"  # orange

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def use_style() -> None:
    """Apply the book's chart look. Call once at the top of a notebook."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "figure.constrained_layout.use": True,  # makes room for legends below
            "axes.facecolor": PANEL,
            "axes.edgecolor": PANEL,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",  # dejaVu Sans has no medium weight
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,  # chrome behind the data, never over it
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "grid.linestyle": "-",
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.dpi": 110,
        }
    )


# Figures already handed to `show`, so a later call does not display them
# again. Tracked by identity because figure numbers get reused once a figure
# is closed.
_shown: set[int] = set()


def show() -> None:
    """
    Show every figure created since the last call, inline and in a window.

    With a GUI backend (`%matplotlib tk`) the window opens as usual and a PNG
    of the figure also goes to the notebook output, so the Interactive Window
    keeps a scrollable record while the window is there for zooming and
    hovering. With the inline backend this is plain `plt.show()`. In a script
    run under Agg it does nothing visible.

    Every plotting cell in the notebooks ends with this call and nothing else,
    which is what keeps a figure from showing twice.
    """
    import matplotlib

    figs = [plt.figure(n) for n in plt.get_fignums() if id(plt.figure(n)) not in _shown]
    backend = matplotlib.get_backend().lower()
    if "inline" in backend:
        plt.show()
        return
    try:
        from IPython import get_ipython
        from IPython.display import Image, display
    except ImportError:
        plt.show()
        return
    if get_ipython() is not None:
        import io

        for fig in figs:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=fig.dpi, bbox_inches="tight")
            display(Image(buf.getvalue()))
            _shown.add(id(fig))
    with warnings.catch_warnings():
        # Under a non-GUI backend (a script run on Agg) there is no window to
        # open and matplotlib warns about it. Expected.
        warnings.simplefilter("ignore", UserWarning)
        plt.show(block=False)


def merge_legends(fig: plt.Figure, ncols: int | None = None):
    """
    Replace the per-axes legends of a multi-panel figure with one figure
    legend below all panels, one entry per distinct label. Panels that share
    their series would otherwise get two legends side by side.
    """
    for ax in fig.axes:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    return legend_below(fig, ncols=ncols)


def legend_below(target, ncols: int | None = None, handles=None, labels=None):
    """
    Place the one legend for an Axes or a Figure below the plotting area.

    This is the only legend call in the module. On an Axes it hangs under the
    x-axis, below the tick labels and any x-label. On a Figure it sits under
    every panel with one entry per distinct label. Constrained layout,
    switched on in `use_style`, makes room for it. Returns the legend, or None
    when there is nothing to label.
    """
    if handles is None:
        axes = (
            target.axes
            if hasattr(target, "axes") and not hasattr(target, "plot")
            else [target]
        )
        seen: dict[str, object] = {}
        for ax in axes:
            for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
                seen.setdefault(lab, h)
        handles, labels = list(seen.values()), list(seen.keys())
    if not handles:
        return None
    ncols = ncols or min(len(labels), 6)
    if hasattr(target, "plot"):  # an Axes
        # A fixed distance in points below the axes, enough to clear the tick
        # labels and the x-label when there is one. A fraction of the axes
        # height would put the legend far away on a tall panel.
        from matplotlib.transforms import offset_copy

        drop = 34 if target.get_xlabel() else 20
        anchor = offset_copy(
            target.transAxes, target.figure, x=0, y=-drop, units="points"
        )
        return target.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            bbox_transform=anchor,
            ncols=ncols,
        )
    return target.legend(handles, labels, loc="outside lower center", ncols=ncols)


def _series(df: pd.DataFrame, series_id: str) -> pd.DataFrame:
    return df[df["id"] == series_id].sort_values("date")


# --------------------------------------------------------------------------- #
# FPP §1.6 step 3: is there a pattern? a trend? outliers?
# --------------------------------------------------------------------------- #


def plot_series(
    df: pd.DataFrame,
    series_id: str,
    ax: plt.Axes | None = None,
    roll: int = 28,
    title: str | None = None,
    holdout_start: pd.Timestamp | None = None,
    flag_outliers: bool = False,
) -> plt.Axes:
    """
    Daily sales for one series, with a rolling mean over the top.

    The grey is the actual daily sales line. About 1,900 daily points in one
    panel zig-zag faster than the eye can follow, so it reads as a grey cloud,
    and the vertical thickness of the cloud at any date is how much day-to-day
    variation there was. The blue line is the rolling average, the thing to
    follow for level and trend.
    """
    ax = ax or plt.gca()
    g = _series(df, series_id)

    ax.plot(
        g["date"],
        g["sales"],
        lw=0.5,
        color=INK_MUTED,
        alpha=0.55,
        zorder=1,
        label="actual daily sales",
    )
    ax.plot(
        g["date"],
        g["sales"].rolling(roll, min_periods=roll // 2).mean(),
        lw=2.0,
        color=SERIES_1,
        zorder=2,
        label=f"{roll}-day average",
    )

    if flag_outliers:
        odd = _outliers(g["sales"], roll)
        ax.scatter(
            g["date"][odd],
            g["sales"][odd],
            s=26,
            facecolors="none",
            edgecolors=SERIES_2,
            linewidths=1.4,
            zorder=4,
            label="outlier",
        )

    if holdout_start is not None:
        # Everything right of this line is reserved for scoring, and took no
        # part in any statistic reported anywhere in this step.
        ax.axvline(holdout_start, color=SERIES_2, lw=1.2, zorder=3)

    ax.set_title(title or series_id)
    ax.set_ylabel("units/day")
    ax.margins(x=0.01)

    # Year ticks only. Monthly ticks collide once panels are this narrow.
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return ax


def plot_store_grid(
    df: pd.DataFrame,
    item_id: str,
    stats: pd.DataFrame,
    roll: int = 28,
    ncols: int = 5,
    holdout_start: pd.Timestamp | None = None,
    flag_outliers: bool = False,
) -> plt.Figure:
    """
    One product, every store, as small multiples ordered by volume.

    The central exhibit of the study. The product is held constant, so
    anything that differs between panels is a property of the store's demand.
    Each panel is titled with its measured demand class.

    The y-axes are not shared. Store volumes differ by an order of magnitude
    and a shared axis would flatten the small stores into a featureless line.
    Panel heights are then not comparable, which is why the mean is printed in
    each title.
    """
    s = stats[stats["item_id"] == item_id].sort_values("mean_sales", ascending=False)
    nrows = int(np.ceil(len(s) / ncols))

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.3 * nrows), sharex=True
    )
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, s.iterrows(), strict=False):
        plot_series(
            df,
            row["id"],
            ax=ax,
            roll=roll,
            title=f"{row['store_id']}  ·  {row['demand_class']}  ·  {row['mean_sales']:.1f}/day",
            holdout_start=holdout_start,
            flag_outliers=flag_outliers,
        )
        ax.set_ylabel("")
    for ax in axes[len(s) :]:
        ax.set_visible(False)

    axes[0].set_ylabel("units/day")
    fig.suptitle(
        f"{item_id} — daily sales by store, {roll}-day rolling mean", fontsize=11
    )
    legend_below(fig)
    return fig


# --------------------------------------------------------------------------- #
# FPP §1.6 step 3: is seasonality important?
# --------------------------------------------------------------------------- #


def plot_seasonality(
    df: pd.DataFrame, series_id: str, axes: np.ndarray | None = None
) -> np.ndarray:
    """
    Weekday and month profiles for one series, in two panels.

    Weekday seasonality is what a 7-day forecast lives or dies on. The month
    panel says whether an annual pattern exists that is worth a feature.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    g = _series(df, series_id)
    span = f"{g['date'].min().year}–{g['date'].max().year}"

    dow = g.groupby(g["date"].dt.dayofweek, observed=True)["sales"].mean()
    axes[0].bar(dow.index, dow.to_numpy(), color=SERIES_1, width=0.68)
    axes[0].set_xticks(range(7), DOW_NAMES)
    axes[0].set_title(f"by weekday  (all days {span})")
    axes[0].set_ylabel("mean units/day")

    mon = g.groupby(g["date"].dt.month, observed=True)["sales"].mean()
    axes[1].bar(mon.index, mon.to_numpy(), color=SERIES_1, width=0.68)
    axes[1].set_xticks(range(1, 13), list("JFMAMJJASOND"))
    axes[1].set_title(f"by month  (all days {span})")

    for ax in axes:
        ax.grid(axis="x", visible=False)
    return axes


def plot_seasonality_by_store(
    df: pd.DataFrame, item_id: str, axes: np.ndarray | None = None
) -> np.ndarray:
    """
    Weekday and month profiles for every store of one item, on one axis.

    Each store's profile is divided by that store's own mean, so a 100-unit
    store and a 15-unit store both read as "how far above or below my usual
    level is this weekday or month". Without that step the busiest store sets
    the shape and the small ones sit invisible along the bottom.

    Thin grey lines are the stores and the bold blue line is their average.
    Grey lines bunched around the blue mean the seasonal shape belongs to the
    product and one feature serves every store. Grey lines that fan out mean
    seasonality differs by store and a pooled model needs store identity.
    """
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9.5, 2.9))
    g = df[df["item_id"] == item_id].copy()
    g["dow"] = g["date"].dt.dayofweek
    g["month"] = g["date"].dt.month
    span = f"{g['date'].min().year}–{g['date'].max().year}"
    n_stores = g["store_id"].nunique()

    panels = (
        (axes[0], "dow", range(7), DOW_NAMES, "by weekday"),
        (axes[1], "month", range(1, 13), list("JFMAMJJASOND"), "by month"),
    )
    for ax, key, ticks, labels, title in panels:
        profile = g.groupby(["store_id", key], observed=True)["sales"].mean().unstack(key)
        profile = profile.div(profile.mean(axis=1), axis=0)  # index to own mean = 1.0

        store_lines = []
        for store, row in profile.iterrows():
            (line,) = ax.plot(
                row.index,
                row.to_numpy(),
                color=INK_MUTED,
                lw=1.0,
                alpha=0.45,
                zorder=2,
                label=str(store),  # read back by the hover tooltip
            )
            store_lines.append(line)
        (mean_line,) = ax.plot(
            profile.columns,
            profile.mean(axis=0).to_numpy(),
            color=SERIES_1,
            lw=2.2,
            zorder=3,
            label=f"mean of {n_stores} stores",
        )
        ax.axhline(1.0, color=AXIS, lw=1.0, zorder=1)
        ax.set_xticks(list(ticks), labels)
        ax.set_title(f"{title}  (every store, {span})")
        ax.grid(axis="x", visible=False)
        hover_labels(store_lines)

    axes[0].set_ylabel("× the store's own mean")
    # Only the mean line goes in the legend. The store lines are labelled for
    # the hover tooltip, a ten-entry legend box would be useless.
    legend_below(axes[0], handles=[mean_line], labels=[mean_line.get_label()])
    return axes


def hover_labels(artists) -> None:
    """
    Show an artist's label as a tooltip when the mouse hovers over it.

    Works in the Tk pop-out window the notebook can open. Does nothing for
    inline or PNG output, where there is no mouse, and nothing when
    `mplcursors` is not installed.
    """
    try:
        import mplcursors
    except ImportError:
        return

    cursor = mplcursors.cursor(artists, hover=True)

    @cursor.connect("add")
    def _show_label(selection):
        selection.annotation.set_text(selection.artist.get_label())


# --------------------------------------------------------------------------- #
# The measurement this project turns on
# --------------------------------------------------------------------------- #


def plot_demand_class_map(
    stats: pd.DataFrame,
    highlight_item: str | None = None,
    ax: plt.Axes | None = None,
    item_id: str | None = None,
) -> plt.Axes:
    """
    Every series placed by how often it sells (ADI) and how consistently
    (CV²), with the conventional cut points drawn in.

    Colour does not encode the class. The class is the quadrant, so position
    already carries it. Colour carries emphasis instead, the highlighted item
    is solid and labelled and everything else goes grey.

    The cut points at 1.32 and 0.49 are conventions, and a series just either
    side of a line is no different from its neighbour. The scatter shows that
    in a way a count-by-class table cannot.

    `item_id` restricts the plot to one item. `highlight_item` emphasises one
    item among several. With a single-item study set they are the same thing.
    """
    ax = ax or plt.gca()
    if item_id is not None:
        stats = stats[stats["item_id"] == item_id]
    finite = stats[np.isfinite(stats["adi"]) & np.isfinite(stats["cv2"])]

    if highlight_item is None:
        focus, rest = finite, finite.iloc[0:0]
    else:
        focus = finite[finite["item_id"] == highlight_item]
        rest = finite[finite["item_id"] != highlight_item]

    ax.scatter(
        rest["adi"],
        rest["cv2"],
        s=42,
        facecolors="none",
        edgecolors=INK_MUTED,
        linewidths=1.2,
        zorder=2,
    )
    ax.scatter(
        focus["adi"],
        focus["cv2"],
        s=58,
        color=SERIES_1,
        edgecolors=SURFACE,
        linewidths=2,
        zorder=3,
    )
    # Labels are stacked with a minimum vertical gap and joined to their point
    # by a hairline, so stores that sit on top of each other stay readable. A
    # smooth item's stores all do. Alternating sides halves the stacking.
    ordered = focus.sort_values("cv2")
    if len(ordered):
        # In CV² units. The axis always spans at least the 0.49 cut, so a fixed
        # gap reads the same on every item.
        gap = 0.022
        last = {True: -np.inf, False: -np.inf}
        for i, (_, r) in enumerate(ordered.iterrows()):
            right = i % 2 == 0
            y = max(float(r["cv2"]), last[right] + gap)
            last[right] = y
            ax.annotate(
                r["store_id"],
                (r["adi"], r["cv2"]),
                xytext=(r["adi"] + (0.012 if right else -0.012), y),
                textcoords="data",
                ha="left" if right else "right",
                va="center",
                fontsize=7.5,
                color=INK_SOFT,
                arrowprops=dict(
                    arrowstyle="-", color=INK_MUTED, lw=0.6, shrinkA=0, shrinkB=2
                ),
            )

    ax.axvline(ADI_CUT, color=AXIS, lw=1.0, zorder=1)
    ax.axhline(CV2_CUT, color=AXIS, lw=1.0, zorder=1)

    # Room below and left of the cloud so the corner labels have somewhere to
    # sit that is not on a data point. A smooth item's stores all crowd the
    # bottom-left corner, which is where "smooth" has to be written.
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(x0 - 0.04 * (x1 - x0), x1)
    ax.set_ylim(min(0.0, y0) - 0.06 * (y1 - y0), y1)

    # Quadrant names sit in the corners in muted ink. They label regions of the
    # plot.
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    px, py = 0.01 * (x1 - x0), 0.02 * (y1 - y0)
    for name, (x, y, ha, va) in {
        "smooth": (x0 + px, y0 + py, "left", "bottom"),
        "erratic": (x0 + px, y1 - py, "left", "top"),
        "intermittent": (x1 - px, y0 + py, "right", "bottom"),
        "lumpy": (x1 - px, y1 - py, "right", "top"),
    }.items():
        ax.text(x, y, name, ha=ha, va=va, fontsize=8, color=INK_MUTED, style="italic")

    ax.set_xlabel(f"ADI  —  days per sale  (cut {ADI_CUT})")
    ax.set_ylabel(f"CV²  —  variability of sale size  (cut {CV2_CUT})")
    ax.set_title(
        "Demand classification"
        + (f" — {highlight_item} highlighted" if highlight_item else "")
    )
    return ax


def plot_demand_class_cloud(
    stats: pd.DataFrame, keep: pd.Series | None = None, ax: plt.Axes | None = None
) -> plt.Axes:
    """
    The class map for a whole department: thousands of series as small dots
    on log axes, no store labels. `plot_demand_class_map` is for one item.

    Log axes because ADI runs from 1 to a few hundred and the cut at 1.32
    would otherwise sit against the left edge. `keep` marks the series that
    pass the availability screen in ink; the rest stay grey, so the eye sees
    what the screen removed.
    """
    ax = ax or plt.gca()
    finite = stats[np.isfinite(stats["adi"]) & np.isfinite(stats["cv2"])]
    if keep is None:
        keep = pd.Series(True, index=finite.index)
    keep = keep.reindex(finite.index, fill_value=False)
    for sel, colour, alpha, z in ((~keep, INK_MUTED, 0.3, 2), (keep, INK, 0.5, 3)):
        g = finite[sel]
        ax.scatter(
            g["adi"], g["cv2"], s=7, color=colour, alpha=alpha, linewidths=0, zorder=z
        )
    ax.axvline(ADI_CUT, color=AXIS, lw=1.0, zorder=1)
    ax.axhline(CV2_CUT, color=AXIS, lw=1.0, zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    for name, (x, y, ha, va) in {
        "smooth": (x0 * 1.05, y0 * 1.2, "left", "bottom"),
        "erratic": (x0 * 1.05, y1 / 1.2, "left", "top"),
        "intermittent": (x1 / 1.05, y0 * 1.2, "right", "bottom"),
        "lumpy": (x1 / 1.05, y1 / 1.2, "right", "top"),
    }.items():
        ax.text(x, y, name, ha=ha, va=va, fontsize=8, color=INK_MUTED, style="italic")
    ax.set_xlabel(f"ADI  —  days per sale  (cut {ADI_CUT}), log scale")
    ax.set_ylabel(f"CV²  —  variability of sale size  (cut {CV2_CUT}), log scale")
    ax.set_title(f"Demand classification, {len(finite):,} series")
    return ax


# --------------------------------------------------------------------------- #
# FPP §1.6 step 3: how strong are the relationships between variables?
# --------------------------------------------------------------------------- #


def plot_snap_effect(
    df: pd.DataFrame, item_id: str, stats: pd.DataFrame, ax: plt.Axes | None = None
) -> plt.Axes:
    """
    Mean daily sales on SNAP benefit days against other days, by store.

    SNAP dates are known years ahead, so any effect here is free forecasting
    signal. The model can use it without predicting anything.
    """
    ax = ax or plt.gca()
    order = stats[stats["item_id"] == item_id].sort_values("mean_sales", ascending=False)[
        "store_id"
    ]

    g = df[df["item_id"] == item_id]
    means = g.groupby(["store_id", "snap"], observed=True)["sales"].mean().unstack()
    means = means.reindex(order)

    x = np.arange(len(means))
    # A subset containing no SNAP days would otherwise raise KeyError.
    for col in (0, 1):
        if col not in means:
            means[col] = np.nan
    ax.bar(x - 0.19, means[0], width=0.36, color=SERIES_1, label="ordinary day")
    ax.bar(x + 0.19, means[1], width=0.36, color=SERIES_2, label="SNAP day")

    # The number to read is the lift. One callout per store, inside the top of
    # the SNAP bar on a light pill so it reads against the orange. Small
    # differences between neighbouring bars are hard to judge by eye and a
    # printed +6.5% is not.
    lift = (means[1] / means[0] - 1) * 100
    for xi, (top, pct) in enumerate(zip(means[1], lift, strict=True)):
        if np.isnan(top) or np.isnan(pct):
            continue
        ax.annotate(
            f"{pct:+.1f}%",
            (xi + 0.19, top),
            textcoords="offset points",
            xytext=(0, -3),
            ha="center",
            va="top",
            fontsize=7,
            color=INK,
            bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.85),
            zorder=4,
        )

    ax.set_xticks(x, means.index, rotation=0)
    ax.set_ylabel("mean units/day")
    ax.set_title(f"{item_id} — SNAP benefit days vs ordinary days")
    ax.grid(axis="x", visible=False)
    legend_below(ax)
    return ax


# --------------------------------------------------------------------------- #
# FPP §1.6 step 3: outliers, business cycles, and variable relationships
# --------------------------------------------------------------------------- #


def _outliers(sales: pd.Series, roll: int = 28, k: float = 4.0) -> pd.Series:
    """
    Points far from the local level, by a robust control-limit rule.

    Residual from a centred rolling median, scaled by the median absolute
    deviation (times 1.4826, which puts MAD on the same footing as a standard
    deviation for normal data). Anything beyond k robust sigma is flagged.

    Median and MAD instead of mean and standard deviation for the usual
    control-chart reason. A couple of large spikes drag the mean and inflate
    the standard deviation, and the outliers end up hiding themselves. k=4 is
    loose on purpose, this marks points worth asking about.
    """
    level = sales.rolling(roll, center=True, min_periods=roll // 2).median()
    resid = sales - level
    mad = resid.abs().rolling(roll, center=True, min_periods=roll // 2).median()
    return resid.abs() > k * (1.4826 * mad).replace(0, np.nan)


def sample_acf(y: np.ndarray, nlags: int) -> np.ndarray:
    """
    Sample autocorrelation r_k for k = 0..nlags, as defined in FPP §2.8.

    r_k measures how strongly a day resembles the day k days earlier. Written
    out so there is nothing here to take on trust:

        r_k = sum_t (y_t - ybar)(y_{t-k} - ybar) / sum_t (y_t - ybar)^2
    """
    y = np.asarray(y, dtype=float)
    y = y - y.mean()
    denom = (y**2).sum()
    return np.array([1.0] + [(y[k:] * y[:-k]).sum() / denom for k in range(1, nlags + 1)])


def plot_acf(
    df: pd.DataFrame,
    series_id: str,
    ax: plt.Axes | None = None,
    nlags: int = 35,
    title: str | None = None,
) -> plt.Axes:
    """
    Autocorrelation of daily sales, the evidence behind the lag features.

    This plot decides which lags are worth giving a model. Spikes at 7, 14,
    21, 28 mean the weekly cycle is the dominant structure. A slow decay from
    lag 1 means the recent level matters in its own right.

    The grey band is the +/- 1.96/sqrt(T) white-noise interval (FPP §2.9).
    Bars inside it are not distinguishable from random.
    """
    ax = ax or plt.gca()
    y = _series(df, series_id)["sales"].to_numpy(dtype=float)
    r = sample_acf(y, nlags)
    bound = 1.96 / np.sqrt(len(y))

    lags = np.arange(1, nlags + 1)
    ax.fill_between([0.5, nlags + 0.5], -bound, bound, color=AXIS, alpha=0.3, zorder=0)
    ax.bar(lags, r[1:], width=0.55, color=SERIES_1, zorder=2)
    ax.axhline(0, color=AXIS, lw=1.0, zorder=1)

    for k in range(7, nlags + 1, 7):  # label the weekly harmonics only
        ax.annotate(
            str(k),
            (k, r[k]),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=7,
            color=INK_SOFT,
        )

    ax.set_xlabel("lag (days)")
    ax.set_ylabel("autocorrelation")
    ax.set_title(title or "Autocorrelation")
    ax.grid(axis="x", visible=False)
    return ax


def plot_year_overlay(
    df: pd.DataFrame, series_id: str, ax: plt.Axes | None = None, roll: int = 28
) -> plt.Axes:
    """
    Each calendar year drawn on a shared day-of-year axis (FPP §2.4).

    Answers whether there is an annual pattern. Years that trace the same
    shape mean the shape is seasonal and worth a feature. Years that wander
    independently mean what looked like seasonality in a single time plot was
    drift, and a month feature would fit noise.

    More years than hues, so the most recent year is solid blue and earlier
    years go grey. The legend carries identity.
    """
    ax = ax or plt.gca()
    g = _series(df, series_id).copy()
    g["level"] = g["sales"].rolling(roll, min_periods=roll // 2).mean()
    years = sorted(g["date"].dt.year.unique())

    for i, yr in enumerate(years):
        sub = g[g["date"].dt.year == yr]
        newest = yr == years[-1]
        ax.plot(
            sub["date"].dt.dayofyear,
            sub["level"],
            color=SERIES_1 if newest else INK_MUTED,
            lw=2.0 if newest else 1.0,
            alpha=1.0 if newest else 0.3 + 0.12 * i,
            label=str(yr),
            zorder=3 if newest else 2,
        )

    ax.set_xlabel("day of year")
    ax.set_ylabel(f"{roll}-day average")
    ax.set_title("Year-on-year overlay")
    legend_below(ax, ncols=len(years))
    return ax


def plot_price_relationship(
    df: pd.DataFrame, item_id: str, ax: plt.Axes | None = None
) -> plt.Axes:
    """
    Sales against shelf price, one point per store-week (FPP §2.7 scatterplot).

    Price is known in advance in this dataset, so any relationship here is
    signal a forecast can use. Aggregated to weekly because price only changes
    weekly. Daily points would stack seven identical x-values on every price
    and make the cloud look far denser than the evidence behind it.
    """
    ax = ax or plt.gca()
    g = df[df["item_id"] == item_id]
    wk = (
        g.groupby(["store_id", "wm_yr_wk"], observed=True)
        .agg(price=("sell_price", "first"), sales=("sales", "mean"))
        .reset_index()
        .dropna()
    )
    ax.scatter(
        wk["price"], wk["sales"], s=14, color=SERIES_1, alpha=0.35, edgecolors="none"
    )
    # When every point stacks into a few columns, the columns are the finding.
    # Label each with the dates that price was in force, so the reader sees a
    # clock rather than a relationship.
    spans = (
        g.dropna(subset=["sell_price"])
        .groupby("sell_price", observed=True)["date"]
        .agg(["min", "max"])
    )
    top = wk["sales"].max()
    if len(spans) <= 6:
        for price, row in spans.iterrows():
            ax.annotate(
                f"{row['min']:%b %Y}\nto {row['max']:%b %Y}",
                (price, top),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                va="bottom",
                fontsize=7,
                color=INK_SOFT,
            )
        ax.set_ylim(0, top * 1.3)
        ax.margins(x=0.15)
    ax.set_xlabel("shelf price ($)")
    ax.set_ylabel("mean units/day that week")
    n_prices = wk["price"].nunique()
    ax.set_title(
        f"{item_id} — sales vs shelf price: {n_prices} prices in the whole history"
    )
    return ax


def plot_event_effects(
    table: pd.DataFrame,
    major: tuple[str, ...] | list[str] = (),
    ax: plt.Axes | None = None,
    threshold: float = 0.15,
) -> plt.Axes:
    """
    Every calendar event's effect on sales, from `step3_explore.event_effects`.
    The evidence behind the holiday feature set.

    Two markers per event, the event day itself and the larger of the two
    days before it (the run-up). Both are ratios to a same-weekday baseline,
    so 1.0 is an ordinary day and 1.7 is 70% above. Events in `major` are
    drawn in colour and the rest go grey. The dotted lines mark the threshold
    an event had to clear, in either direction, to be counted.

    A closure day (Christmas) has no day marker. The stores were shut, so
    there is no demand to measure and the loader imputed the value. Its
    run-up is still drawn, and it is the largest in the calendar, which is why
    an on/off flag alone is not enough.
    """
    ax = ax or plt.gca()
    t = table.reset_index().sort_values("max_deviation")
    y = np.arange(len(t))
    is_major = t["event"].isin(major).to_numpy()
    runup = t[["d-2", "d-1"]].max(axis=1).to_numpy()

    for sel, colour, alpha in ((~is_major, INK_MUTED, 0.7), (is_major, SERIES_1, 1.0)):
        ax.scatter(
            t["d0"].to_numpy()[sel], y[sel], s=34, color=colour, alpha=alpha, zorder=3
        )
        ax.scatter(
            runup[sel],
            y[sel],
            s=34,
            facecolors="none",
            edgecolors=colour,
            linewidths=1.4,
            alpha=alpha,
            zorder=3,
        )
    # A legend needs handles of its own. The coloured points above carry no
    # label because colour means "selected" here rather than a series.
    ax.scatter([], [], s=34, color=INK_SOFT, label="the day itself")
    ax.scatter(
        [],
        [],
        s=34,
        facecolors="none",
        edgecolors=INK_SOFT,
        linewidths=1.4,
        label="run-up (larger of the two days before)",
    )
    ax.axvline(1.0, color=AXIS, lw=1.0, zorder=1)
    for x in (1 - threshold, 1 + threshold):
        ax.axvline(x, color=AXIS, lw=0.8, ls=":", zorder=1)

    ax.set_yticks(y, t["event"], fontsize=7.5)
    for tick, m in zip(ax.get_yticklabels(), is_major, strict=True):
        tick.set_color(INK if m else INK_MUTED)
    ax.set_xlabel("sales ÷ same-weekday baseline")
    ax.set_title("Calendar events: how much each one moves sales")
    ax.grid(axis="y", visible=False)
    ax.margins(y=0.02)
    legend_below(ax)
    return ax


def plot_zero_rate(
    stats: pd.DataFrame, item_id: str | None = None, ax: plt.Axes | None = None
) -> plt.Axes:
    """
    Share of days with no sale, per series. The sparsity of the working set.

    One bar per store, ordered by volume. On a fast mover these all sit near
    zero. A store standing out is one to look at before it reaches a model.

    Pass `item_id` whenever `stats` holds more than one item, otherwise the
    bars of different products land on one axis with nothing to tell them
    apart.
    """
    ax = ax or plt.gca()
    if item_id is not None:
        stats = stats[stats["item_id"] == item_id]
    s = stats.sort_values("mean_sales", ascending=False)
    ax.bar(range(len(s)), s["zero_rate"] * 100, color=SERIES_1, width=0.68)
    ax.set_xticks(range(len(s)), s["store_id"])
    ax.set_ylabel("days with no sale")
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:.1f}%")
    ax.set_title("Zero-sales rate across the working set")
    ax.grid(axis="x", visible=False)
    return ax


# --------------------------------------------------------------------------- #
# Step 5: evaluating the forecasts (FPP §5.4, §5.8, §5.10)
# --------------------------------------------------------------------------- #

# Models carry a hue and benchmarks go grey. Cool colours are the statistical
# models, warm colours the machine learning ones, and gold is always the final
# model. Benchmarks are told apart by line style and marker instead, so the
# legend still identifies each one. Keys are method names, "@pooled" for a
# pooled run.
NAVY = "#1F3864"
MODEL_COLOURS = {
    "arima": NAVY,
    "ets": SERIES_3,
    "arma": "#7B52AB",
    "arima_plain": "#56B4E9",
    "xgboost_rel@pooled": SERIES_5,
    "xgboost": SERIES_2,
    "xgboost@pooled": "#8C2D04",
    "xgboost_rel": "#E8836B",
}
BENCH_LINES = {
    "seasonal_naive": "--",
    "moving_average_28": ":",
    "seasonal_naive_364": "-.",
    "mean": (0, (5, 2, 1, 2)),
    "naive": (0, (1, 1)),
    "drift": (0, (3, 1)),
}
BENCH_MARKERS = {
    "seasonal_naive": "s",
    "moving_average_28": "D",
    "seasonal_naive_364": "^",
    "mean": "v",
    "naive": "x",
    "drift": "+",
}

# The subset drawn by default where nine lines would be a tangle. The three
# models plus the two benchmarks that actually compete.
DEFAULT_METHODS = ("xgboost", "ets", "arima", "moving_average_28", "seasonal_naive")


def _method_style(name: str, kind: str) -> dict:
    if kind == "model":
        return dict(color=MODEL_COLOURS.get(name, SERIES_4), lw=1.8, alpha=1.0, zorder=3)
    return dict(
        color=INK_SOFT, lw=1.1, alpha=0.75, zorder=2, ls=BENCH_LINES.get(name, "-")
    )


def _drop_closures(predictions: pd.DataFrame) -> pd.DataFrame:
    """Closure days are imputed rather than observed, so they are never scored or drawn as residuals."""
    if "closure" in predictions:
        return predictions[~predictions["closure"].astype(bool)]
    return predictions


def _present(predictions: pd.DataFrame, methods) -> list[str]:
    """Requested methods that actually appear in the predictions, in order."""
    have = set(predictions["method"].unique())
    return [m for m in (methods or DEFAULT_METHODS) if m in have]


def plot_forecast_folds(
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    series_id: str,
    holdout_start: pd.Timestamp | None = None,
    origins=None,
    methods=None,
    lead_in: int = 28,
    window: tuple | None = None,
    horizon: int | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Each method's forecasts drawn over the actual sales, across every fold.
    FPP §5.8's figure, competing forecasts on the same axes as what happened.

    - `window` = (start, end) restricts the drawing to a span of dates. A
      full year of daily forecasts is too dense to read. Eight weeks around
      the holidays is where the methods separate.
    - `horizon` picks which forecast to draw for each day when folds overlap
      (a fold step shorter than the window gives every day one forecast per
      horizon). None means all of them when folds tile, and h = 1 when they
      overlap, the one-day-ahead line an order placed the day before would
      have used.
    - When the folds tile the held-out window end to end, each method's
      seven-day forecasts join into one line. The faint vertical lines are the
      fold origins. A kink there is the origin moving.
    - `lead_in` days of history are drawn before the held-out window so the
      eye has the recent level for context.
    """
    ax = ax or plt.gca()
    p = predictions[predictions["id"] == series_id]
    if window is not None:
        lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
        p = p[(p["target_date"] >= lo) & (p["target_date"] <= hi)]
        lead_in = 0
        if origins is not None:
            origins = [o for o in origins if lo <= o <= hi]
    overlapping = _folds_overlap(p)
    if horizon is None and overlapping:
        horizon = 1
    if horizon is not None:
        p = p[p["horizon"] == horizon]
        if overlapping:
            origins = None  # one line per day; daily origin lines would be noise
    methods = _present(p, methods)
    start, end = p["target_date"].min(), p["target_date"].max()

    actual = _series(df, series_id)
    actual = actual[
        (actual["date"] >= start - pd.Timedelta(days=lead_in)) & (actual["date"] <= end)
    ]
    ax.plot(actual["date"], actual["sales"], color=INK, lw=1.4, zorder=4, label="actual")

    for name in methods:
        q = p[p["method"] == name].sort_values("target_date")
        style = _method_style(name, q["kind"].iloc[0])
        ax.plot(q["target_date"], q["forecast"], label=name, **style)

    if origins is not None:
        for o in origins:
            ax.axvline(o, color="#d4d4d4", lw=0.7, zorder=1)
    if holdout_start is not None:
        ax.axvline(holdout_start, color=AXIS, lw=1.2, zorder=1)

    ax.set_ylabel("units/day")
    span = "every fold" if window is None else f"{start:%d %b %Y} to {end:%d %b %Y}"
    what = f"{horizon}-day-ahead forecasts" if horizon is not None else "forecasts"
    ax.set_title(f"{series_id.split('_evaluation')[0]} — {what} against actuals, {span}")
    ax.margins(x=0.01)
    if window is not None:
        ax.set_xlim(start - pd.Timedelta(days=1), end + pd.Timedelta(days=1))
    _date_axis(ax, start, end)
    legend_below(ax, ncols=len(methods) + 1)
    return ax


def _folds_overlap(predictions: pd.DataFrame) -> bool:
    """True when some day carries more than one forecast per method, i.e. the fold step is shorter than the window."""
    if predictions.empty:
        return False
    one = predictions[predictions["method"] == predictions["method"].iloc[0]]
    return bool(one["target_date"].duplicated().any())


def _date_axis(ax: plt.Axes, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """
    Tick a date axis at a density the panel can carry (weeks, months or years),
    thinned to roughly one tick per inch of panel width.
    """
    days = (end - start).days
    width_in = ax.get_position().width * ax.figure.get_figwidth()
    max_ticks = max(3, int(width_in * 1.1))
    if days <= 90:
        weeks = days // 7 + 1
        step = max(1, int(np.ceil(weeks / max_ticks)))
        ax.xaxis.set_major_locator(
            mdates.WeekdayLocator(byweekday=mdates.MO, interval=step)
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    elif days <= 800:
        months = days // 30 + 1
        step = max(1, int(np.ceil(months / max_ticks)))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=step))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def plot_rmsse_by_horizon(
    predictions: pd.DataFrame, methods=None, ax: plt.Axes | None = None
) -> plt.Axes:
    """
    Error as a function of how far ahead the forecast was made (FPP §5.10,
    figure 5.24).

    Each point is the RMSSE over every series-fold for forecasts made h days
    from the origin. Error should rise with h, a day-ahead forecast knows more
    than a week-ahead one, and how steeply it rises is the thing to read. A
    flat line means the method is not using the recent past at all (a mean
    does this). A steep one means its advantage is mostly at short horizons.

    On the weekly layout horizon is confounded with weekday. Origins step by
    exactly seven days, so h=1 is always the same weekday and h=7 always
    another, and the busy weekend day inflates h=7 for every method. The
    every-day layout (item 2) removes this and is the reported one.
    """
    ax = ax or plt.gca()
    predictions = _drop_closures(predictions)
    methods = _present(predictions, methods)
    p = predictions[predictions["method"].isin(methods)].copy()
    p["q2"] = ((p["forecast"] - p["actual"]) / p["scale"]) ** 2
    by_h = p.groupby(["method", "kind", "horizon"], observed=True)["q2"].mean() ** 0.5

    for name in methods:
        s = by_h.xs(name, level="method")
        kind = s.index.get_level_values("kind")[0]
        s = s.droplevel("kind")
        ax.plot(
            s.index,
            s.to_numpy(),
            marker=BENCH_MARKERS.get(name, "o"),
            ms=4,
            label=name,
            **_method_style(name, kind),
        )

    ax.set_xlabel("days ahead of the forecast origin (h)")
    ax.set_ylabel("RMSSE")
    ax.set_title("Error by forecast horizon")
    ax.set_xticks(sorted(p["horizon"].unique()))
    ax.grid(axis="x", visible=False)
    legend_below(ax)
    return ax


def plot_rmsse_by_store(
    scores: pd.DataFrame,
    stats: pd.DataFrame,
    item_id: str | None = None,
    methods=None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Mean RMSSE per store, one marker per method, busiest store first.

    Not one of FPP's figures. It is the view this study was designed around,
    whether a method's advantage depends on the store. Read left to right
    down the volume gradient and watch whether the coloured markers pull away
    from the grey pack or sink back into it.
    """
    ax = ax or plt.gca()
    if item_id is not None:
        stats = stats[stats["item_id"] == item_id]
        scores = scores[scores["item_id"] == item_id]
    order = stats.sort_values("mean_sales", ascending=False)["store_id"].tolist()
    methods = _present(scores, methods) if methods else sorted(scores["method"].unique())
    table = scores.pivot_table(
        index="store_id", columns="method", values="rmsse", aggfunc="mean"
    ).reindex(order)
    kinds = scores.drop_duplicates("method").set_index("method")["kind"]

    x = np.arange(len(order))
    for name in methods:
        if name not in table:
            continue
        style = _method_style(name, kinds[name])
        if kinds[name] == "model":
            ax.plot(x, table[name].to_numpy(), marker="o", ms=6, label=name, **style)
        else:
            ax.scatter(
                x,
                table[name].to_numpy(),
                s=26,
                marker=BENCH_MARKERS.get(name, "o"),
                label=name,
                color=style["color"],
                alpha=style["alpha"],
                zorder=style["zorder"],
            )

    ax.axhline(1.0, color=AXIS, lw=1.0, zorder=1)
    ax.set_xticks(x, order)
    ax.set_ylabel("mean RMSSE over folds")
    ax.set_title("RMSSE by store, busiest first  (1.0 = seasonal naive on training data)")
    ax.grid(axis="x", visible=False)
    legend_below(ax, ncols=min(len(methods), 5))
    return ax


def plot_bias_by_store(
    scores: pd.DataFrame,
    stats: pd.DataFrame,
    item_id: str | None = None,
    methods=("xgboost", "ets", "arima"),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Mean bias (forecast minus actual, units per day) per store, one marker per
    model, busiest store first. Zero is the line to sit on. A method whose
    markers sit on one side of it at most stores has a systematic over- or
    under-forecast that a symmetric error score does not show.
    """
    ax = ax or plt.gca()
    if item_id is not None:
        stats = stats[stats["item_id"] == item_id]
        scores = scores[scores["item_id"] == item_id]
    order = stats.sort_values("mean_sales", ascending=False)["store_id"].tolist()
    table = scores.pivot_table(
        index="store_id", columns="method", values="bias", aggfunc="mean"
    ).reindex(order)
    kinds = scores.drop_duplicates("method").set_index("method")["kind"]
    x = np.arange(len(order))
    ax.axhline(0, color=AXIS, lw=1.0, zorder=1)
    for name in methods:
        if name not in table:
            continue
        style = _method_style(name, kinds[name])
        ax.plot(
            x,
            table[name].to_numpy(),
            marker="o",
            ms=6,
            lw=0,
            label=name,
            **{k: v for k, v in style.items() if k != "lw"},
        )
    ax.set_xticks(x, order)
    ax.set_ylabel("mean forecast − actual, units/day")
    ax.set_title("Bias by store, busiest first  (above zero = over-forecast)")
    ax.grid(axis="x", visible=False)
    legend_below(ax, ncols=len(methods))
    return ax


def plot_residual_diagnostics(
    predictions: pd.DataFrame,
    series_id: str,
    method: str,
    season: int = 7,
    axes: np.ndarray | None = None,
) -> np.ndarray:
    """
    FPP §5.4's three residual panels for one method on one series.

    The book says a good method's residuals are uncorrelated and centred on
    zero, and ideally of constant variance and roughly normal. One panel per
    question:

      time plot  - is the mean zero, and is the spread constant over time?
      ACF        - is anything left that the method should have captured?
      histogram  - is the spread roughly symmetric and bell-shaped?

    Residuals here are `forecast - actual`, the same convention as the `bias`
    column in the tables (positive = over-forecast). FPP writes them the
    other way round. Every diagnostic is identical under a sign flip except
    the sign of the mean, which is stated in the title.

    The Ljung-Box p-value tests the ACF panel at the lag FPP prescribes for
    seasonal data (2m, capped at T/5). Above 0.05 means the residuals look
    like white noise and the method has taken everything predictable.

    FPP's residual tests are stated for one-step-ahead residuals. With
    overlapping folds (the every-day layout) the one-step residuals are used
    and the verdict is a real one. On the weekly layout these are 1- to
    7-step-ahead errors that share one origin, so a small autocorrelation at
    lags 1-6 is the design. A spike at lag 7 or beyond, or a mean far from
    zero, is what would count against a method there.
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    predictions = _drop_closures(predictions)
    p = predictions[(predictions["id"] == series_id) & (predictions["method"] == method)]
    # With overlapping folds every day has one residual per horizon. Use the
    # one-step-ahead residuals, which is what FPP's diagnostics are stated for.
    one_step = _folds_overlap(p)
    if one_step:
        p = p[p["horizon"] == 1]
    p = p.sort_values("target_date")
    r = (p["forecast"] - p["actual"]).to_numpy(dtype=float)
    n = len(r)
    label = "one-step residuals" if one_step else "residuals"

    # --- time plot ------------------------------------------------------
    axes[0].plot(p["target_date"], r, color=SERIES_1, lw=0.9)
    axes[0].axhline(0, color=AXIS, lw=1.0)
    axes[0].set_title(f"{label} over time  (mean {r.mean():+.2f})")
    axes[0].set_ylabel("forecast − actual")
    _date_axis(axes[0], p["target_date"].min(), p["target_date"].max())

    # --- ACF, with FPP's portmanteau test ---------------------------------
    nlags = max(1, min(3 * season, n // 2))
    acf = sample_acf(r, nlags)
    bound = 1.96 / np.sqrt(n)
    lb_lag = max(1, min(2 * season, n // 5))
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox

        pval = float(acorr_ljungbox(r, lags=[lb_lag])["lb_pvalue"].iloc[0])
        verdict = f"Ljung-Box p={pval:.2f} at lag {lb_lag}"
    except Exception:  # statsmodels absent or too few points
        verdict = "Ljung-Box unavailable"
    axes[1].fill_between(
        [0.5, nlags + 0.5], -bound, bound, color=AXIS, alpha=0.3, zorder=0
    )
    axes[1].bar(np.arange(1, nlags + 1), acf[1:], width=0.55, color=SERIES_1, zorder=2)
    axes[1].axhline(0, color=AXIS, lw=1.0)
    axes[1].set_title(f"residual ACF  ({verdict})")
    axes[1].set_xlabel("lag (days)")
    axes[1].grid(axis="x", visible=False)

    # --- histogram ----------------------------------------------------------
    axes[2].hist(r, bins=min(15, max(5, n // 4)), color=SERIES_1, edgecolor=SURFACE)
    axes[2].axvline(0, color=AXIS, lw=1.0)
    axes[2].set_title("residual distribution")
    axes[2].set_xlabel("forecast − actual")
    axes[2].grid(axis="x", visible=False)
    return axes


# --------------------------------------------------------------------------- #
# From forecast to order: quantile forecasts (FPP §5.5, §5.9)
# --------------------------------------------------------------------------- #


def plot_pinball_by_tau(
    summary: pd.DataFrame, methods=None, ax: plt.Axes | None = None
) -> plt.Axes:
    """
    Relative pinball loss against the service level τ, one line per method.

    Answers whether the ranking changes with the cost asymmetry. Lines that
    cross mean it does, the best method for a perishable (τ below 0.5) is a
    different one from the best for an ambient item (τ near 0.9). Lower is
    better everywhere. Values at different τ are on different scales and are
    never averaged along a line.
    """
    ax = ax or plt.gca()
    methods = _present(summary, methods)
    kinds = summary.drop_duplicates("method").set_index("method")["kind"]
    for name in methods:
        s = summary[summary["method"] == name].sort_values("tau")
        ax.plot(
            s["tau"],
            s["pinball_rel"],
            marker=BENCH_MARKERS.get(name, "o"),
            ms=5,
            label=name,
            **_method_style(name, kinds[name]),
        )
    ax.set_xticks(sorted(summary["tau"].unique()))
    ax.set_xlabel("service level τ  (share of weeks the order should cover)")
    ax.set_ylabel("pinball loss ÷ mean weekly sales")
    ax.set_title("Quantile score by service level  (FPP §5.9; lower is better)")
    ax.grid(axis="x", visible=False)
    legend_below(ax)
    return ax


def plot_coverage_by_tau(
    summary: pd.DataFrame, methods=None, ax: plt.Axes | None = None
) -> plt.Axes:
    """
    Achieved coverage against the target τ. A perfectly calibrated method
    sits on the diagonal, its 0.9-quantile order covers 90% of weeks. Above
    the line is over-ordering, below is stockouts more often than promised.
    The gap is how well one year's error distribution described the next.
    """
    ax = ax or plt.gca()
    methods = _present(summary, methods)
    kinds = summary.drop_duplicates("method").set_index("method")["kind"]
    taus = sorted(summary["tau"].unique())
    ax.plot(
        [taus[0] - 0.05, taus[-1] + 0.05],
        [taus[0] - 0.05, taus[-1] + 0.05],
        color=AXIS,
        lw=1.0,
        zorder=1,
        label="perfect calibration",
    )
    for name in methods:
        s = summary[summary["method"] == name].sort_values("tau")
        ax.plot(
            s["tau"],
            s["coverage"],
            marker=BENCH_MARKERS.get(name, "o"),
            ms=5,
            label=name,
            **_method_style(name, kinds[name]),
        )
    ax.set_xticks(taus)
    ax.set_xlabel("target service level τ")
    ax.set_ylabel("share of weeks covered")
    ax.set_title("Achieved coverage against target")
    ax.grid(axis="x", visible=False)
    legend_below(ax)
    return ax


def plot_weekly_order_band(
    scored: pd.DataFrame,
    series_id: str,
    method: str,
    taus=(0.3, 0.5, 0.7, 0.9),
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    One store, one method. Each scored week's actual total against the
    order-up-to levels at several service levels.

    The black line is what the store sold each week. The coloured bands are
    the quantile forecasts. The τ = 0.5 line is the median forecast, the
    upper edge is τ = 0.9 (what a 90% service level would order) and the
    lower edge τ = 0.3 (what a perishable's economics would order). Weeks
    where black rises above a band's top edge are the stockouts that service
    level would have accepted. Holiday weeks are shaded.
    """
    ax = ax or plt.gca()
    s = scored[(scored["id"] == series_id) & (scored["method"] == method)]
    wide = s.pivot_table(index="origin", columns="tau", values="q").sort_index()
    actual = s.drop_duplicates("origin").set_index("origin")["actual"].reindex(wide.index)
    x = wide.index + pd.Timedelta(days=1)  # the Monday the order covers from
    lo, mid, hi = min(taus), 0.5, max(taus)
    if lo in wide and hi in wide:
        ax.fill_between(
            x,
            wide[lo],
            wide[hi],
            color=SERIES_1,
            alpha=0.18,
            lw=0,
            zorder=1,
            label=f"order-up-to band, τ = {lo} to {hi}",
        )
    if mid in wide:
        ax.plot(
            x,
            wide[mid],
            color=SERIES_1,
            lw=1.4,
            zorder=2,
            label="median forecast (τ = 0.5)",
        )
    if hi in wide:
        ax.plot(x, wide[hi], color=SERIES_1, lw=0.9, ls="--", zorder=2, label=f"τ = {hi}")
    ax.plot(x, actual, color=INK, lw=1.4, zorder=3, label="actual weekly sales")

    holiday = (
        s.drop_duplicates("origin").set_index("origin")["week_kind"].reindex(wide.index)
    )
    for origin, kind in holiday.items():
        if kind == "holiday":
            ax.axvspan(
                origin + pd.Timedelta(days=1),
                origin + pd.Timedelta(days=8),
                color=SERIES_5,
                alpha=0.12,
                lw=0,
                zorder=0,
            )
    ax.plot([], [], color=SERIES_5, alpha=0.4, lw=8, label="holiday week")

    ax.set_ylabel("units/week")
    ax.set_title(
        f"{series_id.split('_evaluation')[0]} — {method}: weekly order-up-to levels"
    )
    ax.margins(x=0.01)
    _date_axis(ax, x.min(), x.max())
    legend_below(ax, ncols=5)
    return ax
