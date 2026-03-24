import matplotlib

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
RC_PARAMS = {
    "font.family":        "serif",
    "font.serif":         ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset":   "stix",
    "font.size":          45,
    "axes.labelsize":     45,
    "axes.titlesize":     45,
    "xtick.labelsize":    35,
    "ytick.labelsize":    35,
    "legend.fontsize":    45,
    "axes.edgecolor":     "black",
    "axes.linewidth":     2.2,
    "lines.markersize":   10,
    "legend.markerscale": 1.8,
    "grid.color":         "0.1",   # dark grey (0 = black, 1 = white)
    "grid.linewidth":     1.5,
    "grid.linestyle":     "--",
    "xtick.color":        "black",
    "ytick.color":        "black",
    "xtick.major.size":   14,
    "ytick.major.size":   14,
    "xtick.major.width":  3.0,
    "ytick.major.width":  3.0,
    "xtick.minor.size":   10,
    "ytick.minor.size":   10,
    "xtick.minor.width":  2.0,
    "ytick.minor.width":  2.0,
    "xtick.direction":    "inout",
    "ytick.direction":    "inout",
}

# ---------------------------------------------------------------------------
# Figure geometry & output quality
# ---------------------------------------------------------------------------
FIGSIZE = (8, 8)   # (width, height) in inches
DPI     = 600     # dots per inch for saved figures


def apply():
    """Apply the shared rcParams.  Call once per script before any figure."""
    matplotlib.rcParams.update(RC_PARAMS)