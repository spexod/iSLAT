"""
LineInspectionPlot — Zoomed-in view of a narrow wavelength region.

Shows the observed spectrum, overlaid molecule model(s), and optionally
individual line markers with energy/A-coefficient labels.

Can be used standalone (notebook / script) or embedded in a GUI layout.
"""

from typing import Optional, Tuple, List, Dict, Any, Union, TYPE_CHECKING
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .BasePlot import BasePlot

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine

class LineInspectionPlot(BasePlot):
    """
    Plot a narrow wavelength region with observed data and molecule models.

    Parameters
    ----------
    wave_data : np.ndarray
        Full wavelength array (the region is selected with *xmin*/*xmax*).
    flux_data : np.ndarray
        Full flux array matching *wave_data*.
    xmin, xmax : float
        Wavelength bounds for the inspection window.
    error_data : np.ndarray, optional
        Error values matching *wave_data*.
    molecule : Molecule, optional
        Single active molecule whose model is overlaid.
    molecules : MoleculeDict, optional
        If provided *all visible* molecules are plotted instead of just one.
    line_data : list, optional
        List of ``(MoleculeLine, intensity, tau)`` tuples for line markers.
    line_threshold : float, optional
        Fraction (0-1) of the strongest line below which lines are hidden.
        Defaults to ``0.0`` (show all).
    figsize : tuple, optional
        Figure size. Defaults to ``(10, 4)``.
    ax : Axes, optional
        Pre-existing axes to draw into (for embedding in a larger figure).
    """

    def __init__(
        self,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        xmin: float,
        xmax: float,
        error_data: Optional[np.ndarray] = None,
        molecule: Optional["Molecule"] = None,
        molecules: Optional["MoleculeDict"] = None,
        line_data: Optional[List[Tuple["MoleculeLine", float, Optional[float]]]] = None,
        line_threshold: float = 0.0,
        figsize: Optional[Tuple[float, float]] = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ):
        super().__init__(figsize=figsize or (10, 4), **kwargs)
        self.wave_data = wave_data
        self.flux_data = flux_data
        self.xmin = xmin
        self.xmax = xmax
        self.error_data = error_data
        self.molecule = molecule
        self.molecules = molecules
        self.line_data = line_data
        self.line_threshold = line_threshold
        self._external_ax = ax

    # ------------------------------------------------------------------
    @property
    def ax(self) -> Axes:
        """The axes used for this plot."""
        return self._ax

    # ------------------------------------------------------------------
    def generate_plot(self, **kwargs) -> None:  # noqa: D401
        """Generate (or regenerate) the line inspection plot."""
        # Resolve axes
        if self._external_ax is not None:
            self._ax = self._external_ax
        else:
            self._ensure_figure()
            self._ax = self.fig.add_subplot(111)

        ax = self._ax
        ax.clear()

        fg = self._get_theme_value("foreground", "black")

        # Mask to the inspection range
        mask = (self.wave_data >= self.xmin) & (self.wave_data <= self.xmax)
        obs_wave = self.wave_data[mask]
        obs_flux = self.flux_data[mask]
        obs_err = self.error_data[mask] if self.error_data is not None else None

        # -- observed spectrum -------------------------------------------
        self._plot_observed_spectrum(ax, obs_wave, obs_flux, obs_err)

        max_y = float(np.nanmax(obs_flux)) if len(obs_flux) > 0 else 0.15

        # -- molecule model(s) ------------------------------------------
        if self.molecules is not None:
            visible = self.molecules.get_visible_molecules(return_objects=True)
            for mol in visible:
                mol_max = self._overlay_molecule(ax, mol)
                if mol_max is not None and len(obs_flux) == 0:
                    max_y = max(max_y, mol_max)
        elif self.molecule is not None:
            mol_max = self._overlay_molecule(ax, self.molecule)
            if mol_max is not None and len(obs_flux) == 0:
                max_y = max(max_y, mol_max)

        # -- individual line markers ------------------------------------
        if self.line_data:
            max_y = self._plot_line_markers(ax, self.line_data, max_y)

        # -- appearance -------------------------------------------------
        ax.set_xlim(self.xmin, self.xmax)
        if max_y > 0:
            ax.set_ylim(0, max_y * 1.1)
        ax.set_xlabel("Wavelength (μm)", color=fg)
        ax.set_ylabel("Flux density (Jy)", color=fg)
        ax.set_title("Line inspection plot", color=fg)
        self._update_legend(ax)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _overlay_molecule(self, ax: Axes, molecule: "Molecule") -> Optional[float]:
        """Plot one molecule model in the inspection range.

        Returns the max flux value in the range, or *None* if nothing was plotted.
        """
        plot_lam, model_flux = self.get_molecule_spectrum_data(molecule, self.wave_data)
        if plot_lam is None or model_flux is None:
            return None
        m = (plot_lam >= self.xmin) & (plot_lam <= self.xmax)
        if not np.any(m):
            return None
        model_wave_range = plot_lam[m]
        model_flux_range = model_flux[m]
        if len(model_wave_range) == 0 or len(model_flux_range) == 0:
            return None
        ax.plot(
            model_wave_range,
            model_flux_range,
            color=self.get_molecule_color(molecule),
            linestyle="--",
            linewidth=2,
            label=self.get_molecule_display_name(molecule),
        )
        return float(np.nanmax(model_flux_range))

    def _plot_line_markers(
        self,
        ax: Axes,
        line_data: List[Tuple["MoleculeLine", float, Optional[float]]],
        max_y: float,
    ) -> float:
        """Draw vertical dashed lines for individual molecular transitions."""
        if not line_data:
            return max_y

        intensities = [i for _, i, _ in line_data]
        max_intensity = max(intensities) if intensities else 1.0
        threshold_val = max_intensity * self.line_threshold

        color = self._get_theme_value("active_scatter_line_color", "green")

        for line_obj, intensity, _ in line_data:
            if intensity < threshold_val:
                continue
            lineheight = (intensity / max_intensity) * max_y if max_intensity > 0 else 0
            if lineheight <= 0:
                continue
            ax.vlines(
                line_obj.lam,
                0,
                lineheight,
                color=color,
                linestyle="dashed",
                linewidth=1,
            )
            ax.text(
                line_obj.lam,
                lineheight,
                f"{line_obj.e_up:.0f},{line_obj.a_stein:.3f}",
                fontsize="x-small",
                color=color,
                rotation=45,
            )
        return max_y

    # ------------------------------------------------------------------
    # Convenience: update the wavelength range without rebuilding
    # ------------------------------------------------------------------
    def set_range(self, xmin: float, xmax: float) -> None:
        """Change the inspection range and regenerate."""
        self.xmin = xmin
        self.xmax = xmax
        self.generate_plot()