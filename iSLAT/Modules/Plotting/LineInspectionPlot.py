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
    # Line information helpers
    # ------------------------------------------------------------------
    @staticmethod
    def get_line_info(
        line: "MoleculeLine",
        intensity: float,
        tau: Optional[float] = None,
        data_flux_in_range: Optional[float] = None,
        model_flux_in_range: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Build a structured information dict for a single molecular line.

        This is the canonical source of per-line metadata used by both
        the GUI data-field display and standalone notebooks/scripts.

        Parameters
        ----------
        line : MoleculeLine
            The molecular transition.
        intensity : float
            Computed model intensity for this line.
        tau : float, optional
            Line opacity.
        data_flux_in_range : float, optional
            Observed (data) flux integral in the selection range
            (erg s⁻¹ cm⁻²).
        model_flux_in_range : float, optional
            Model flux integral in the selection range (erg s⁻¹ cm⁻²).

        Returns
        -------
        dict
            Keys: ``lam``, ``e_up``, ``e_low``, ``a_stein``, ``g_up``,
            ``g_low``, ``up_lev``, ``low_lev``, ``intensity``, ``tau``,
            ``data_flux_in_range``, ``model_flux_in_range``,
            ``formatted_text``.
        """
        lam     = getattr(line, "lam", None)
        e_up    = getattr(line, "e_up", None)
        e_low   = getattr(line, "e_low", None)
        a_stein = getattr(line, "a_stein", None)
        g_up    = getattr(line, "g_up", None)
        g_low   = getattr(line, "g_low", None)
        up_lev  = getattr(line, "lev_up", None) or "N/A"
        low_lev = getattr(line, "lev_low", None) or "N/A"
        tau_val = tau if tau is not None else "N/A"

        # Build formatted text ------------------------------------------
        wav_s   = f"{lam:.6f}"       if lam     is not None else "N/A"
        a_s     = f"{a_stein:.3e}"   if a_stein is not None else "N/A"
        e_s     = f"{e_up:.0f}"      if e_up    is not None else "N/A"
        tau_s   = f"{tau_val:.3f}"   if isinstance(tau_val, (int, float)) else str(tau_val)
        dflux_s = f"{data_flux_in_range:.3e}"  if data_flux_in_range  is not None else "N/A"
        mflux_s = f"{model_flux_in_range:.3e}" if model_flux_in_range is not None else "N/A"

        text = (
            "\n--- Line Information ---\n"
            "Selected line:\n"
            f"Upper level = {up_lev}\n"
            f"Lower level = {low_lev}\n"
            f"Wavelength (μm) = {wav_s}\n"
            f"Einstein-A coeff. (1/s) = {a_s}\n"
            f"Upper level energy (K) = {e_s}\n"
            f"Opacity = {tau_s}\n"
            f"Data flux in range (erg/s/cm2) = {dflux_s}\n"
            f"Model flux in range (erg/s/cm2) = {mflux_s}\n"
        )

        return {
            "lam":                  lam,
            "e_up":                 e_up,
            "e_low":                e_low if e_low else "N/A",
            "a_stein":              a_stein,
            "g_up":                 g_up,
            "g_low":                g_low if g_low else "N/A",
            "up_lev":               up_lev,
            "low_lev":              low_lev,
            "intensity":            intensity,
            "tau":                  tau_val,
            "data_flux_in_range":   data_flux_in_range,
            "model_flux_in_range":  model_flux_in_range,
            "formatted_text":       text,
        }

    @staticmethod
    def format_line_info(info: Dict[str, Any]) -> str:
        """Return the pre-built formatted text from a :meth:`get_line_info` dict."""
        return info.get("formatted_text", "")

    # ------------------------------------------------------------------
    # Convenience: update the wavelength range without rebuilding
    # ------------------------------------------------------------------
    def set_range(self, xmin: float, xmax: float) -> None:
        """Change the inspection range and regenerate."""
        self.xmin = xmin
        self.xmax = xmax
        self.generate_plot()