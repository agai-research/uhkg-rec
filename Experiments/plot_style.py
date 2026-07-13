"""
plot_style.py
-------------
Shared, professional plotting style for every figure produced under
Experiments/. Uses a colorblind-safe qualitative palette (Okabe-Ito
derived) instead of matplotlib's default color cycle, consistent
marker/hatch conventions per method, and a helper to apply a clean
academic look (light grid, no top/right spines) to any matplotlib Axes.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette (method / series colors)
METHOD_COLORS = {
    "HKG-Rec": "#0072B2",               # blue
    "UHKG-Rec": "#D55E00",              # vermillion
    "UHKG-Rec (theta=0.6)": "#D55E00",
    "UHKG-Rec (theta=0.8)": "#009E73",  # bluish green
    "UHKG-Rec_NM": "#CC79A7",           # reddish purple
    "CAGE_Rec": "#E69F00",              # orange
    "GAT_Rec": "#56B4E9",               # sky blue
    "KGAT": "#56B4E9",
}

METHOD_HATCHES = {
    "HKG-Rec": "",
    "UHKG-Rec": "///",
    "UHKG-Rec (theta=0.6)": "///",
    "UHKG-Rec (theta=0.8)": "xxx",
    "UHKG-Rec_NM": "...",
}

METHOD_MARKERS = {
    "HKG-Rec": "D",
    "UHKG-Rec (theta=0.6)": "x",
    "UHKG-Rec (theta=0.8)": "o",
    "UHKG-Rec_NM": "^",
    "CAGE_Rec": "^",
    "GAT_Rec": "s",
    "KGAT": "s",
}

# 5-category palette for the embedding-visualization figure
CATEGORY_COLORS = {
    "Cancer": "#0072B2",       # blue
    "Diabetes": "#009E73",     # green
    "Dermatology": "#8E44AD",  # purple
    "Allergy": "#E69F00",      # orange
    "Respiratory": "#17BECF",  # cyan
}
CATEGORY_FALLBACK = ["#CC79A7", "#D55E00", "#999999", "#F0E442", "#56B4E9"]


def style_axis(ax):
    """Clean academic look: no top/right spines, light grid, readable ticks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def method_color(name, fallback_idx=0):
    if name in METHOD_COLORS:
        return METHOD_COLORS[name]
    palette = list(dict.fromkeys(METHOD_COLORS.values()))
    return palette[fallback_idx % len(palette)]


def category_color(name, idx=0):
    if name in CATEGORY_COLORS:
        return CATEGORY_COLORS[name]
    return CATEGORY_FALLBACK[idx % len(CATEGORY_FALLBACK)]
