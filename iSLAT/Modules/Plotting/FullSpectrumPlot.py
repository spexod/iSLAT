"""
FullSpectrumPlot — Multi-panel overview of an entire observed spectrum.

Generates a vertically stacked series of wavelength-range panels, each
showing the observed data, individual molecule models, summed model
spectrum, and optionally line-list annotations and atomic lines.

Can be used standalone (notebook / script) or embedded in a GUI layout.
"""

from typing import Optional, Tuple, List, Dict, Any, Union, TYPE_CHECKING
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .BasePlot import BasePlot

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict

class FullSpectrumPlot(BasePlot):
    """
    Multi-panel full-spectrum plot.

    Parameters
    ----------
    wave_data : np.ndarray
        Observed wavelength array (already RV-corrected if needed).
    flux_data : np.ndarray
        Observed flux array.
    molecules : MoleculeDict, optional
        Collection of molecules — visible ones are plotted.
    error_data : np.ndarray, optional
        Flux uncertainties.
    line_list : pd.DataFrame, optional
        Saved line annotations (columns: ``wave`` / ``lam``, ``species``,
        ``line``).
    atomic_lines : pd.DataFrame, optional
        Atomic line annotations (columns: ``wave``, ``species``, ``line``).
    n_panels : int, optional
        Target number of panels. Defaults to 10.
    step : float, optional
        Wavelength width of each panel. Overrides *n_panels*.
    xlim_range : tuple[float, float], optional
        ``(start, end)`` wavelength range. Defaults to full data range.
    ymax_factor : float, optional
        Fractional padding above the peak flux in each panel (0.2 = 20 %).
    figsize : tuple, optional
        Figure size.  Height is scaled automatically if *None*.
    fig : Figure, optional
        Existing figure for embedding.
    """

    def __init__(
        self,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        molecules: Optional["MoleculeDict"] = None,
        error_data: Optional[np.ndarray] = None,
        line_list: Optional[pd.DataFrame] = None,
        atomic_lines: Optional[pd.DataFrame] = None,
        n_panels: int = 10,
        step: Optional[float] = None,
        xlim_range: Optional[Tuple[float, float]] = None,
        ymax_factor: float = 0.2,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        # Defer figsize — calculated once we know the number of panels
        super().__init__(figsize=figsize, **kwargs)
        self.wave_data = np.asarray(wave_data)
        self.flux_data = np.asarray(flux_data)
        self.molecules = molecules
        self.error_data = np.asarray(error_data) if error_data is not None else None
        self.line_list = line_list
        self.atomic_lines = atomic_lines

        self.n_panels = n_panels
        self.ymax_factor = ymax_factor

        # Wavelength range
        if xlim_range is not None:
            self._xlim_start, self._xlim_end = xlim_range
        else:
            self._xlim_start = float(np.nanmin(self.wave_data))
            self._xlim_end = float(np.nanmax(self.wave_data))

        # Panel step
        if step is not None:
            self._step = step
        else:
            self._step = (self._xlim_end - self._xlim_start) / max(self.n_panels, 1)

        # Pre-compute panel edges
        self._panel_edges: np.ndarray = np.arange(
            self._xlim_start, self._xlim_end, self._step
        )
        # Auto figsize if not given
        if self._figsize is None:
            self._figsize = (12, 1.6 * len(self._panel_edges))

        # Storage for subplot Axes objects
        self.subplots: Dict[int, Axes] = {}

    # ------------------------------------------------------------------
    def generate_plot(self, **kwargs) -> None:
        """Build the multi-panel figure."""
        n = len(self._panel_edges)
        self._ensure_figure()
        # Clear previous axes so regeneration doesn't stack on top
        self.fig.clf()

        # Compute summed flux once (if molecules are available)
        summed_wave: Optional[np.ndarray] = None
        summed_flux: Optional[np.ndarray] = None
        if self.molecules is not None:
            try:
                summed_wave, summed_flux = self.molecules.get_summed_flux(
                    self.wave_data, visible_only=True
                )
            except Exception:
                pass

        # Prepare molecule legend info
        mol_labels: List[str] = []
        mol_colors: List[str] = []
        if self.molecules is not None:
            visible = self.molecules.get_visible_molecules(return_objects=True)
            mol_labels = [self.get_molecule_display_name(m) for m in visible]
            mol_colors = [self.get_molecule_color(m) for m in visible]

        for idx, xlim_start in enumerate(self._panel_edges):
            is_last = idx == n - 1
            panel_end = self._xlim_end if is_last else xlim_start + self._step
            xr = (xlim_start, panel_end)

            ax = self.fig.add_subplot(n, 1, idx + 1)
            self.subplots[idx] = ax

            # --- observed spectrum --------------------------------------
            mask = (self.wave_data >= xr[0]) & (self.wave_data <= xr[1])
            panel_wave = self.wave_data[mask]
            panel_flux = self.flux_data[mask]
            panel_err = self.error_data[mask] if self.error_data is not None else None

            self._plot_observed_spectrum(ax, panel_wave, panel_flux, panel_err)

            # y-limits
            if np.any(mask):
                ymax = float(np.nanmax(self.flux_data[mask]))
                ymax += ymax * self.ymax_factor
                ymin = -0.005
            else:
                ymin, ymax = -0.005, 0.1

            # --- molecule models ----------------------------------------
            if self.molecules is not None:
                self._plot_visible_molecules(
                    ax, self.molecules, wave_data=self.wave_data, linewidth=0.8, update_legend=False
                )

            # --- summed spectrum ----------------------------------------
            if summed_wave is not None and summed_flux is not None:
                s_mask = (summed_wave >= xr[0]) & (summed_wave <= xr[1])
                if np.any(s_mask):
                    self._plot_summed_spectrum(ax, summed_wave[s_mask], summed_flux[s_mask])

            # --- line annotations ---------------------------------------
            if self.line_list is not None and len(self.line_list) > 0:
                self._plot_line_annotations(ax, self.line_list, xr, ymin, ymax)

            # --- atomic lines -------------------------------------------
            if self.atomic_lines is not None and len(self.atomic_lines) > 0:
                self._plot_atomic_lines(ax, self.atomic_lines, xr=xr)

            # --- axes config --------------------------------------------
            ax.set_xlim(*xr)
            ax.set_ylim(ymin, ymax)
            ax.tick_params(axis="x", labelsize=7)

        # --- global labels & legend ------------------------------------
        if self.subplots:
            last_ax = self.subplots[n - 1]
            last_ax.set_xlabel("Wavelength (μm)")
        self.fig.supylabel("Flux Density (Jy)", fontsize=10)

        # Custom colour-legend on the first panel
        if mol_labels and 0 in self.subplots:
            self.subplots[0].legend(
                mol_labels,
                labelcolor=mol_colors,
                loc="upper center",
                ncols=min(12, len(mol_labels)),
                handletextpad=0.2,
                bbox_to_anchor=(0.5, 1.4),
                handlelength=0,
                fontsize=10,
                prop={"weight": "bold"},
            )

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def set_line_list(self, df: pd.DataFrame) -> None:
        """Attach or replace the line-list DataFrame."""
        self.line_list = df

    def set_atomic_lines(self, df: pd.DataFrame) -> None:
        """Attach or replace the atomic-lines DataFrame."""
        self.atomic_lines = df