"""Plot style and label constants (#53). Applied once at package import."""

import matplotlib

# Axis / colorbar label text. Importable as module-level constants so
# call sites can do ``from porosity_fe_analysis import LABEL_X_MM`` and
# the GUI / PNG / validation paths share a single source of truth.
LABEL_POROSITY_PCT = "Porosity (%)"
LABEL_POROSITY_VP = "Porosity Vp (%)"
LABEL_X_MM = "x (mm)"
LABEL_Y_MM = "y (mm)"
LABEL_Z_MM = "z (mm)"
LABEL_STIFFNESS_RETENTION = "Stiffness Retention (%)"
LABEL_STIFFNESS_RETENTION_FRAC = "Stiffness Retention (-)"
LABEL_KNOCKDOWN = "Knockdown Factor (-)"
LABEL_SCF = "Stress Concentration Factor (-)"
LABEL_STRESS_MPA = "Stress (MPa)"
LABEL_MAE_PCT = "MAE (%)"

# Legacy dict kept so anything outside this module that still does
# ``LABELS['knockdown_factor']`` keeps working. New code should use the
# ``LABEL_*`` constants above.
LABELS = {
    'porosity_pct': LABEL_POROSITY_PCT,
    'x_mm': LABEL_X_MM,
    'y_mm': LABEL_Y_MM,
    'z_mm': LABEL_Z_MM,
    'stiffness_retention_pct': LABEL_STIFFNESS_RETENTION,
    'knockdown_factor': LABEL_KNOCKDOWN,
    'scf': LABEL_SCF,
}


def _configure_matplotlib_style(style: str = 'default') -> None:
    """Set shared matplotlib rcParams for all plots in the project (#53).

    Parameters
    ----------
    style : {'default', 'publication'}
        ``'default'`` (the import-time setting) is the screen/README
        raster style: 11pt body, 14pt titles, ``savefig.dpi=300``,
        ``image.cmap='cividis'`` (perceptually-uniform + colorblind-
        safe; matches #51).

        ``'publication'`` bumps fonts (~+2pt) for use in figures
        embedded in papers. PNG is retained as the default
        ``savefig.format`` because some downstream consumers
        (Streamlit, GitHub README previews) cannot render PDF inline;
        callers that want vector output should pass an explicit
        ``.pdf`` extension to ``plt.savefig``.
    """
    base = {
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'legend.fontsize': 9,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.titlesize': 16,
        'figure.titleweight': 'bold',
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        # Colorblind-safe perceptually-uniform colormap; matches the
        # damage-contour fix in #51 so cividis is now the project-wide
        # default. Do NOT switch back to 'viridis'.
        'image.cmap': 'cividis',
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    }
    if style == 'publication':
        base.update({
            'font.size': 13,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'legend.fontsize': 11,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'figure.titlesize': 18,
        })
    matplotlib.rcParams.update(base)


# Backwards-compatible alias for callers that imported the old helper.
_apply_plot_style = _configure_matplotlib_style

# Apply at import so any module that imports ``porosity_fe_analysis``
# (the Streamlit app, validation runner, tests) inherits the same style.
_configure_matplotlib_style()

