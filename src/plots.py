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
            "axes.titleweight": "medium",
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

    ax.set_title(title or series_id)
    ax.set_ylabel("units/day")
    ax.margins(x=0.01)

    # Year ticks only. Monthly ticks collide once panels are this narrow.
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    return ax


def plot_store_grid(
    df: pd.DataFrame, item_id: str, stats: pd.DataFrame, roll: int = 28, ncols: int = 5
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
        )
        ax.set_ylabel("")
    for ax in axes[len(s) :]:
        ax.set_visible(False)

    axes[0].set_ylabel("units/day")
    axes[0].legend(loc="upper left", fontsize=7)
    fig.suptitle(
        f"{item_id} — daily sales by store, {roll}-day rolling mean",
        y=1.0,
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
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

    dow = g.groupby(g["date"].dt.dayofweek, observed=True)["sales"].mean()
    axes[0].bar(dow.index, dow.to_numpy(), color=SERIES_1, width=0.68)
    axes[0].set_xticks(range(7), DOW_NAMES)
    axes[0].set_title("by weekday")
    axes[0].set_ylabel("mean units/day")

    mon = g.groupby(g["date"].dt.month, observed=True)["sales"].mean()
    axes[1].bar(mon.index, mon.to_numpy(), color=SERIES_1, width=0.68)
    axes[1].set_xticks(range(1, 13), list("JFMAMJJASOND"))
    axes[1].set_title("by month")

    for ax in axes:
        ax.grid(axis="x", visible=False)
    return axes


# --------------------------------------------------------------------------- #
# The measurement this project turns on
# --------------------------------------------------------------------------- #


def plot_demand_class_map(
    stats: pd.DataFrame, highlight_item: str | None = None, ax: plt.Axes | None = None
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
    """
    ax = ax or plt.gca()
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
    ax.bar(x - 0.19, means[0], width=0.36, color=SERIES_1, label="ordinary day")
    ax.bar(x + 0.19, means[1], width=0.36, color=SERIES_2, label="SNAP day")

    ax.set_xticks(x, means.index, rotation=0)
    ax.set_ylabel("mean units/day")
    ax.set_title(f"{item_id} — SNAP benefit days vs ordinary days")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    return ax
