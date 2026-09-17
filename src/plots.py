"""
Shared plotting.

Every function takes an optional `ax` and returns it, so plots compose into
grids and notebooks without the module owning any figure layout.

Design rules followed here, so they are consistent and defensible:

- **One hue per single-series chart.** Colour carries identity, never magnitude.
  A darker-where-bigger bar chart double-encodes the bar length and wastes the
  only free channel.
- **Emphasis over enumeration.** Where one series matters, it is drawn in the
  series colour and everything else recedes to grey - rather than giving every
  series its own hue and asking the reader to decode a legend.
- **Recessive chrome.** Solid hairline grid one shade off the surface, no
  dashes; dashing reads as "threshold" when it is only a grid.
- **Selective labels.** Direct-label the points that carry the argument, not
  every point.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.step3_explore import ADI_CUT, CV2_CUT

# --- palette ---------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
SERIES_3 = "#1baf7a"  # aqua

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def use_style() -> None:
    """Apply the chart chrome. Call once at the top of a notebook."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.titlesize": 10,
            "axes.titleweight": "normal",  # DejaVu Sans has no medium weight
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,  # chrome behind the data, never over it
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "text.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.dpi": 110,
        }
    )


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

    The grey is **not** a shaded band or a confidence interval - it is the
    actual daily sales line. At ~1,900 daily points squeezed into one panel the
    line zig-zags faster than the eye can follow, so it reads as a grey cloud.
    That is useful rather than accidental: the *vertical thickness* of the grey
    at any date is how much day-to-day variation there was around then.

    The blue line on top is the rolling average, and it is what the eye should
    follow for level and trend - smoothed enough to be readable, not so smoothed
    that a real level shift disappears.
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

    This is the central exhibit of the study: the product is held constant, so
    anything that differs between panels is a property of the store's demand,
    not of the item. Each panel is titled with its measured demand class.

    Y-axes are deliberately *not* shared. Store volumes differ by an order of
    magnitude, and a shared axis would flatten the small stores into a
    featureless line - hiding exactly the series whose behaviour is in question.
    The cost is that panel heights are not comparable, which is why the mean is
    printed in each title.
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
        f"{item_id} — daily sales by store, {roll}-day rolling mean",
        y=1.0,
        fontsize=11,
        color=INK,
    )
    # One legend for the whole figure, in the top-right margin, rather than
    # inside the first panel where it sat on top of that store's data.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        ncols=len(labels),
        fontsize=8,
        bbox_to_anchor=(0.995, 1.005),
        frameon=False,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# --------------------------------------------------------------------------- #
# FPP §1.6 step 3: is seasonality important?
# --------------------------------------------------------------------------- #


def plot_seasonality(
    df: pd.DataFrame, series_id: str, axes: np.ndarray | None = None
) -> np.ndarray:
    """
    Weekday and month profiles for one series.

    Two separate panels rather than one chart with two scales. Weekday
    seasonality is what a 7-day forecast lives or dies on; the month panel says
    whether an annual pattern exists worth giving the model a feature for.
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
    level is this weekday / month". Without that step the busiest store would
    set the shape and the small ones would be invisible along the bottom.

    Thin grey lines are the individual stores; the bold blue line is their
    average. If the grey lines bunch tightly around the blue, the seasonal
    shape is a property of the *product* and one feature serves every store.
    If they fan out, seasonality differs by store and a pooled model needs
    store identity to capture it.
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
    # Only the mean line goes in the legend - the store lines are labelled for
    # the hover tooltip, not for a ten-entry legend box.
    axes[0].legend(handles=[mean_line], loc="upper left")
    return axes


def hover_labels(artists) -> None:
    """
    Show an artist's label as a tooltip when the mouse hovers over it.

    Works in an interactive window - the Tk pop-out the notebook can open - and
    is a silent no-op for inline or PNG output, where there is no mouse. Also a
    no-op if `mplcursors` is not installed, so nothing here can break a plot.
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
    Every series placed by how *often* it sells (ADI) and how *consistently*
    (CV²), with the conventional cut points drawn in.

    Colour deliberately does not encode the class: the class *is* the quadrant,
    so position already carries it and a second encoding would be redundant.
    Instead colour carries emphasis - the highlighted item is solid and
    labelled, everything else recedes.

    Showing the cloud rather than four buckets is the point. The cut points at
    1.32 and 0.49 are conventions, and a series sitting just either side of a
    line is not meaningfully different from its neighbour. The scatter makes
    that visible in a way a count-by-class table cannot.

    `item_id` restricts the plot to one item; `highlight_item` emphasises one
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
    for _, r in focus.iterrows():
        ax.annotate(
            r["store_id"],
            (r["adi"], r["cv2"]),
            textcoords="offset points",
            xytext=(7, 3),
            fontsize=7.5,
            color=INK_SOFT,
        )

    ax.axvline(ADI_CUT, color=AXIS, lw=1.0, zorder=1)
    ax.axhline(CV2_CUT, color=AXIS, lw=1.0, zorder=1)

    # Quadrant names sit in the corners, in muted ink - they label regions of
    # the plot, not data points.
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    pad = 0.02
    for name, (x, y, ha, va) in {
        "smooth": (x0 + pad, y0 + pad, "left", "bottom"),
        "erratic": (x0 + pad, y1 - pad, "left", "top"),
        "intermittent": (x1 - pad, y0 + pad, "right", "bottom"),
        "lumpy": (x1 - pad, y1 - pad, "right", "top"),
    }.items():
        ax.text(x, y, name, ha=ha, va=va, fontsize=8, color=INK_MUTED, style="italic")

    ax.set_xlabel(f"ADI  —  days per sale  (cut {ADI_CUT})")
    ax.set_ylabel(f"CV²  —  variability of sale size  (cut {CV2_CUT})")
    ax.set_title(
        "Demand classification"
        + (f" — {highlight_item} highlighted" if highlight_item else "")
    )
    return ax


# --------------------------------------------------------------------------- #
# FPP §1.6 step 3: how strong are the relationships between variables?
# --------------------------------------------------------------------------- #


def plot_snap_effect(
    df: pd.DataFrame, item_id: str, stats: pd.DataFrame, ax: plt.Axes | None = None
) -> plt.Axes:
    """
    Mean daily sales on SNAP benefit days versus other days, by store.

    Two series, so a legend is present. SNAP dates are known years ahead, which
    makes any effect here free forecasting signal - the model can use it without
    predicting anything.
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

    # The number that matters is the *lift*, not either bar's height - so one
    # callout per store, sitting inside the top of the SNAP bar on a light pill
    # so it reads against the orange. Small differences between neighbouring
    # bars are hard to judge by eye; a printed +6.5% is not.
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
    ax.legend(loc="upper right")
    return ax


# --------------------------------------------------------------------------- #
# FPP §1.6 step 3: outliers, business cycles, and variable relationships
# --------------------------------------------------------------------------- #


def _outliers(sales: pd.Series, roll: int = 28, k: float = 4.0) -> pd.Series:
    """
    Points far from the local level, by a robust control-limit rule.

    Residual from a centred rolling *median*, scaled by the median absolute
    deviation (x1.4826, which puts MAD on the same footing as a standard
    deviation for normal data). Anything beyond k robust sigma is flagged.

    Median and MAD are used rather than mean and standard deviation for the
    usual control-chart reason: a couple of large spikes drag the mean and
    inflate the standard deviation, so the outliers end up concealing
    themselves. k=4 is deliberately loose - this marks points worth *asking
    about*, not defects.
    """
    level = sales.rolling(roll, center=True, min_periods=roll // 2).median()
    resid = sales - level
    mad = resid.abs().rolling(roll, center=True, min_periods=roll // 2).median()
    return resid.abs() > k * (1.4826 * mad).replace(0, np.nan)


def sample_acf(y: np.ndarray, nlags: int) -> np.ndarray:
    """
    Sample autocorrelation r_k for k = 0..nlags, as defined in FPP §2.8.

    r_k measures how strongly a day resembles the day k days earlier. Written
    out rather than imported so there is nothing here to take on trust:

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
    Autocorrelation of daily sales - the evidence behind the lag features.

    This is the plot that decides which lags are worth giving a model. Spikes at
    7, 14, 21, 28 mean the weekly cycle is the dominant structure; a slow decay
    from lag 1 means the recent level matters in its own right. Picking lags
    without looking at this is picking by convention rather than by evidence.

    The grey band is the +/- 1.96/sqrt(T) white-noise interval (FPP §2.9): bars
    inside it are not distinguishable from random.
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

    Answers "are there business cycles, or an annual pattern?". If the years
    trace the same shape, that shape is seasonal and worth a feature. If they
    wander independently, what looked like seasonality in a single time plot was
    drift, and a month feature would fit noise.

    There are more years than the three hues a scatter can safely carry, so this
    uses emphasis rather than enumeration: the most recent year is solid blue,
    earlier years recede to grey. The legend carries identity.
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
    ax.legend(loc="best", ncols=2)
    return ax


def plot_price_relationship(
    df: pd.DataFrame, item_id: str, ax: plt.Axes | None = None
) -> plt.Axes:
    """
    Sales against shelf price, one point per store-week (FPP §2.7 scatterplot).

    Price is known in advance in this dataset, so any relationship visible here
    is signal a forecast can legitimately use.

    Aggregated to weekly because price only changes weekly - plotting daily
    points would stack seven identical x-values on every price and make the
    cloud look far denser than the evidence behind it.
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
    ax.set_xlabel("shelf price ($)")
    ax.set_ylabel("mean units/day that week")
    ax.set_title(f"{item_id} — sales vs price (store-weeks)")
    return ax


def plot_event_effect(
    df: pd.DataFrame, item_id: str, ax: plt.Axes | None = None
) -> plt.Axes:
    """
    Sales on calendar-event days by event type, against the ordinary-day level.

    Holidays are the other calendar feature known years in advance. Each bar is
    that event type's mean expressed as a percentage difference from the
    ordinary-day mean, so stores of very different size pool into one readable
    number.

    The `n` beside each bar counts **distinct event dates**, not rows. Ten
    stores on the same Mother's Day are one observation of Mother's Day, not
    ten - counting rows would overstate the evidence tenfold. Some types occur
    only a handful of times over five years, and a large bar resting on n=16
    dates is a hint rather than a finding.

    The four types are M5's own labels: Sporting (Super Bowl, NBA Finals),
    Cultural (Valentine's, Mother's/Father's Day, Halloween, ...), National (US
    federal holidays), Religious (Easter, Ramadan, Chanukah, ...).
    """
    ax = ax or plt.gca()
    g = df[df["item_id"] == item_id].copy()
    g["event"] = g["event_type_1"].fillna("(none)")

    base = g.loc[g["event"] == "(none)", "sales"].mean()
    effect = (g.groupby("event", observed=True)["sales"].mean() / base - 1) * 100
    effect = effect.drop("(none)", errors="ignore").sort_values()
    n_dates = g.groupby("event", observed=True)["date"].nunique()

    ax.barh(list(effect.index), effect.to_numpy(), color=SERIES_1, height=0.62)
    ax.axvline(0, color=AXIS, lw=1.0)
    for name, val in effect.items():
        # Sit the count just beyond the bar end, on whichever side that is -
        # a fixed offset would print it on top of every negative bar.
        ax.annotate(
            f"n={n_dates[name]} dates",
            (val, name),
            textcoords="offset points",
            xytext=(6 if val >= 0 else -6, 0),
            ha="left" if val >= 0 else "right",
            va="center",
            fontsize=7.5,
            color=INK_MUTED,
        )
    # Room for the "n=NN dates" label beyond the longest bar on either side -
    # without it the label on the most negative bar runs into the tick labels.
    ax.margins(x=0.32)
    ax.set_xlabel("% difference from an ordinary day")
    ax.set_title(f"{item_id} — calendar event effect")
    ax.grid(axis="y", visible=False)
    return ax


def plot_zero_rate(
    stats: pd.DataFrame, item_id: str | None = None, ax: plt.Axes | None = None
) -> plt.Axes:
    """
    Share of days with no sale, per series - the sparsity of the working set.

    One bar per store, ordered by volume. On a fast mover these should all sit
    near zero; a store standing out is one to look at before it reaches a model.

    Pass `item_id` whenever `stats` holds more than one item, or the bars of
    different products land on one axis with nothing to tell them apart.
    """
    ax = ax or plt.gca()
    if item_id is not None:
        stats = stats[stats["item_id"] == item_id]
    s = stats.sort_values("mean_sales", ascending=False)
    ax.bar(range(len(s)), s["zero_rate"] * 100, color=SERIES_1, width=0.68)
    ax.set_xticks(range(len(s)), s["store_id"])
    ax.set_ylabel("% of days with no sale")
    ax.set_title("Zero-sales rate across the working set")
    ax.grid(axis="x", visible=False)
    return ax


# --------------------------------------------------------------------------- #
# Step 5: evaluating the forecasts (FPP §5.4, §5.8, §5.10)
# --------------------------------------------------------------------------- #

# Models carry a hue; benchmarks recede to grey. Seven methods is past the
# three hues a chart can carry safely, and the story is "the two models
# against the pack", so that is what the colour says.
MODEL_COLOURS = {"xgboost": SERIES_1, "ets": SERIES_2}

# The subset drawn by default where seven lines would be a tangle: both
# models plus the two benchmarks that actually compete.
DEFAULT_METHODS = ("xgboost", "ets", "moving_average_28", "seasonal_naive")


def _method_style(name: str, kind: str) -> dict:
    if kind == "model":
        return dict(color=MODEL_COLOURS.get(name, SERIES_3), lw=2.0, alpha=1.0, zorder=3)
    return dict(color=INK_MUTED, lw=1.1, alpha=0.6, zorder=2)


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
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Each method's forecasts drawn over the actual sales, across every fold.

    This is FPP §5.8's figure - competing forecasts on the same axes as what
    happened, with the test period visible. The book's own verdict on its
    example is "it is obvious from the graph", and that is the standard: if a
    method is better, this plot should show it before any table does.

    Because the folds tile the held-out window end to end, each method's
    seven-day forecasts join into one continuous line. The faint vertical
    lines are the fold origins - every seventh day the forecaster was re-run
    from a fresh standing point, so a kink there is the origin moving, not a
    property of the method.

    `lead_in` days of history are drawn before the held-out window so the eye
    has the recent level for context.
    """
    ax = ax or plt.gca()
    p = predictions[predictions["id"] == series_id]
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
            ax.axvline(o, color=GRID, lw=0.8, zorder=1)
    if holdout_start is not None:
        ax.axvline(holdout_start, color=AXIS, lw=1.2, zorder=1)

    ax.set_ylabel("units/day")
    ax.set_title(
        f"{series_id.split('_evaluation')[0]} — forecasts against actuals, every fold"
    )
    ax.legend(loc="upper left", ncols=len(methods) + 1)
    ax.margins(x=0.01)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    return ax


def plot_rmsse_by_horizon(
    predictions: pd.DataFrame, methods=None, ax: plt.Axes | None = None
) -> plt.Axes:
    """
    Error as a function of how far ahead the forecast was made (FPP §5.10,
    figure 5.24).

    Each point is the RMSSE over every series-fold for forecasts made h days
    from the origin. Error should rise with h - a day-ahead forecast knows
    more than a week-ahead one - and *how steeply* it rises is the thing to
    read: a flat line means the method is not using the recent past at all
    (a mean does this), a steep one means its advantage is mostly at short
    horizons.
    """
    ax = ax or plt.gca()
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
            marker="o",
            ms=4,
            label=name,
            **_method_style(name, kind),
        )

    ax.set_xlabel("days ahead of the forecast origin (h)")
    ax.set_ylabel("RMSSE")
    ax.set_title("Error by forecast horizon")
    ax.set_xticks(sorted(p["horizon"].unique()))
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)
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

    Not one of FPP's figures - it is the view this study was designed around.
    The question is whether a method's advantage depends on the store, and
    the volume gradient across the ten stores is where that shows: read left
    to right and watch whether the coloured markers pull away from the grey
    pack, or sink back into it.
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
                s=22,
                label=name,
                color=style["color"],
                alpha=style["alpha"],
                zorder=style["zorder"],
            )

    ax.axhline(1.0, color=AXIS, lw=1.0, zorder=1)
    ax.set_xticks(x, order)
    ax.set_ylabel("mean RMSSE over folds")
    ax.set_title("RMSSE by store, busiest first  (1.0 = seasonal naive on training data)")
    ax.legend(loc="upper left", ncols=2)
    ax.grid(axis="x", visible=False)
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

    The book says a good method's residuals should be (essential) uncorrelated
    and centred on zero, and (useful) of constant variance and roughly normal.
    One panel per question:

      time plot  - is the mean zero, and is the spread constant over time?
      ACF        - is anything left that the method should have captured?
      histogram  - is the spread roughly symmetric and bell-shaped?

    Residuals here are `forecast - actual`, the same convention as the `bias`
    column in the tables (positive = over-forecast). FPP writes them the other
    way round; every diagnostic is identical under a sign flip except the sign
    of the mean, which is stated in the title.

    The Ljung-Box p-value tests the ACF panel formally, at the lag FPP
    prescribes for seasonal data (2m, capped at T/5). Above 0.05 means the
    residuals are indistinguishable from white noise - the method has taken
    everything predictable. These are held-out residuals over a short window,
    so treat the verdict as a check, not a proof.
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    p = predictions[(predictions["id"] == series_id) & (predictions["method"] == method)]
    p = p.sort_values("target_date")
    r = (p["forecast"] - p["actual"]).to_numpy(dtype=float)
    n = len(r)

    # --- time plot ------------------------------------------------------
    axes[0].plot(p["target_date"], r, color=SERIES_1, lw=1.2, marker="o", ms=3)
    axes[0].axhline(0, color=AXIS, lw=1.0)
    axes[0].set_title(f"residuals over time  (mean {r.mean():+.2f})")
    axes[0].set_ylabel("forecast − actual")
    axes[0].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))

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
