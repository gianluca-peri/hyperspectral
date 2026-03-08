"""plotting/style.py – Canonical graphical style shared across all plotting scripts.

Import and call ``apply()`` once per script, after ``matplotlib.use(...)``
but before creating any figure.  Use ``FIGSIZE`` and ``DPI`` wherever
figure geometry or output quality is specified.
"""

import matplotlib

# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
RC_PARAMS = {
    "font.family":      "serif",
    "font.serif":       ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size":        30,
    "axes.labelsize":   30,
    "axes.titlesize":   30,
    "xtick.labelsize":  24,
    "ytick.labelsize":  24,
    "legend.fontsize":  24,
}

# ---------------------------------------------------------------------------
# Figure geometry & output quality
# ---------------------------------------------------------------------------
FIGSIZE = (8, 8)   # (width, height) in inches
DPI     = 150      # dots per inch for saved figures


def apply():
    """Apply the shared rcParams.  Call once per script before any figure."""
    matplotlib.rcParams.update(RC_PARAMS)
