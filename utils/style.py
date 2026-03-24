"""plotting/style.py – Canonical graphical style shared across all plotting scripts.

Import and call ``apply()`` once per script, after ``matplotlib.use(...)``
but before creating any figure.  Use ``FIGSIZE`` and ``DPI`` wherever
figure geometry or output quality is specified.
"""

import numpy as np
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
    "legend.fontsize":    35,
    # Border
    "axes.edgecolor":     "black",
    "axes.linewidth":     2.2,
    # Markers
    "lines.markersize":   10,
    "legend.markerscale": 1.8,
    # Grid – dark dotted lines
    "grid.color":         "0.1",   # dark grey (0 = black, 1 = white)
    "grid.linewidth":     1.5,
    "grid.linestyle":     "--",
    # Tick marks – thick black bars, extending both inside and outside
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
DPI     = 150      # dots per inch for saved figures


def apply():
    """Apply the shared rcParams.  Call once per script before any figure."""
    matplotlib.rcParams.update(RC_PARAMS)


# ---------------------------------------------------------------------------
# Smart legend helper
# ---------------------------------------------------------------------------

def fix_legend(ax, loc='lower right'):
    """Place legend at *loc* with collision detection.

    Strategy:
      1. Tentatively place the legend and call ``canvas.draw()`` to get its
         true rendered size in pixels.
      2. Convert the pixel bounding box to *axes-fraction* coordinates – this
         is stable with respect to ``tight_layout`` because tight_layout only
         moves the axes box in figure space; the content inside (axes fraction)
         is unaffected.
      3. Map axes-fraction → data coordinates using the current axis limits
         (also unaffected by tight_layout).
      4. Check whether any plotted line or fill_between path overlaps the
         legend box in data space.
      5. If there is overlap, expand the y-axis in the appropriate direction
         (down for lower-right, up for upper-right) by the exact legend height
         plus a small margin, then re-place the legend.
    """
    handles, _ = ax.get_legend_handles_labels()
    if not handles:
        return

    if loc not in ('lower right', 'upper right') or ax.get_yscale() == 'log':
        ax.legend(loc=loc)
        return

    # ── 1. Place legend, run tight_layout first, then render ─────────────────
    leg = ax.legend(loc=loc)
    fig = ax.figure
    fig.tight_layout()   # bring axes to its final pixel size before measuring
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # ── 2. Legend bbox in axes-fraction space (stable under tight_layout) ────
    bb_px = leg.get_window_extent(renderer=renderer)
    bb_ax = bb_px.transformed(ax.transAxes.inverted())

    # ── 3. Convert axes fraction → data coordinates ──────────────────────────
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    dx = xmax - xmin
    dy = ymax - ymin

    lx0 = xmin + bb_ax.x0 * dx
    lx1 = xmin + bb_ax.x1 * dx
    ly0 = ymin + bb_ax.y0 * dy
    ly1 = ymin + bb_ax.y1 * dy
    leg_h = ly1 - ly0      # legend height in data units

    # ── 4. Check overlap with every line and fill_between collection ─────────
    def _in_box(xd, yd):
        return np.any((xd >= lx0) & (xd <= lx1) & (yd >= ly0) & (yd <= ly1))

    overlap = False
    for line in ax.get_lines():
        xd = np.asarray(line.get_xdata(), dtype=float)
        yd = np.asarray(line.get_ydata(), dtype=float)
        if xd.size and _in_box(xd, yd):
            overlap = True
            break

    if not overlap:
        for coll in ax.collections:
            for path in coll.get_paths():
                v = path.vertices
                if v.size and _in_box(v[:, 0], v[:, 1]):
                    overlap = True
                    break
            if overlap:
                break

    # ── 5. Expand axis and re-place if needed ────────────────────────────────
    if overlap:
        margin = leg_h * 1.2
        if loc == 'upper right':
            ax.set_ylim(ymin, ymax + margin)
        else:
            ax.set_ylim(ymin - margin, ymax)
        leg.remove()
        ax.legend(loc=loc)
