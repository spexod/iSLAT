"""
PopulationDiagramPlot — Boltzmann / rotation diagram for a single molecule.

Plots ``ln(4πF / (hv A_u g_u))`` vs upper-state energy *E_u* using the
computed intensity data from a :class:`Molecule` object.

Can be used standalone (notebook / script) or embedded in a GUI layout.
"""

from typing import Optional, Tuple, List, Dict, Any, TYPE_CHECKING
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .BasePlot import BasePlot
import iSLAT.Constants as c

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine

class PopulationDiagramPlot(BasePlot):
    """
    Boltzmann / rotation diagram for a molecule.

    Parameters
    ----------
    molecule : Molecule
        Molecule whose intensity data is used.  Intensity will be
        automatically calculated if not already done.
    highlight_lines : list, optional
        List of ``(MoleculeLine, intensity, tau)`` tuples.  These are
        rendered as larger coloured scatter points on top of the base
        diagram.
    figsize : tuple, optional
        Defaults to ``(6, 5)``.
    ax : Axes, optional
        Pre-existing axes for embedding.
    """

    def __init__(
        self,
        molecule: "Molecule",
        highlight_lines: Optional[List[Tuple["MoleculeLine", float, Optional[float]]]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        ax: Optional[Axes] = None,
        **kwargs,
    ):
        super().__init__(figsize=figsize or (6, 5), **kwargs)
        self.molecule = molecule
        self.highlight_lines = highlight_lines
        self._external_ax = ax

    @property
    def ax(self) -> Axes:
        return self._ax

    # ------------------------------------------------------------------
    def generate_plot(self, **kwargs) -> None:
        """Generate the population diagram."""
        if self._external_ax is not None:
            self._ax = self._external_ax
        else:
            self._ensure_figure()
            self._ax = self.fig.add_subplot(111)

        ax = self._ax
        ax.clear()

        fg = self._get_theme_value("foreground", "black")

        mol = self.molecule
        if mol is None:
            ax.set_title("No molecule selected")
            return

        int_pars = self.get_intensity_data(mol)
        if int_pars is None:
            ax.set_title(
                f"{self.get_molecule_display_name(mol)} - No intensity data",
                color=fg,
            )
            return

        # Extract arrays from the intensity table
        wavelength = np.asarray(int_pars["lam"])
        intens_mod = np.asarray(int_pars["intens"])
        Astein_mod = np.asarray(int_pars["a_stein"])
        gu = np.asarray(int_pars["g_up"])
        eu = np.asarray(int_pars["e_up"])

        radius = getattr(mol, "radius", 1.0)
        distance = getattr(mol, "distance", 160.0)

        area = np.pi * (radius * c.ASTRONOMICAL_UNIT_M * 1e2) ** 2
        dist = distance * c.PARSEC_CM
        beam_s = area / dist ** 2

        F = intens_mod * beam_s
        frequency = c.SPEED_OF_LIGHT_MICRONS / wavelength

        # Suppress divide-by-zero warnings for log
        with np.errstate(divide="ignore", invalid="ignore"):
            rd_yax = np.log(4 * np.pi * F / (Astein_mod * c.PLANCK_CONSTANT * frequency * gu))

        threshold = np.nanmax(F) / 100

        valid_rd = rd_yax[F > threshold]
        valid_eu = eu[F > threshold]

        if len(valid_rd) == 0 or len(valid_eu) == 0:
            ax.set_title(
                f"{self.get_molecule_display_name(mol)} - No valid data for population diagram",
                color=fg,
            )
            return

        # Base scatter
        ax.scatter(
            eu,
            rd_yax,
            s=0.5,
            color=self._get_theme_value("scatter_main_color", "#838B8B"),
        )

        # Highlighted lines (e.g. from an inspection selection)
        if self.highlight_lines:
            self._render_highlights(ax, beam_s)

        ax.set_ylim(np.nanmin(valid_rd), np.nanmax(rd_yax) + 0.5)
        ax.set_xlim(np.nanmin(eu) - 50, np.nanmax(valid_eu))
        ax.set_ylabel(r"ln(4πF/(hν$A_{u}$$g_{u}$))", color=fg, labelpad=-1)
        ax.set_xlabel(r"$E_{u}$ (K)", color=fg)
        ax.set_title(
            f"{self.get_molecule_display_name(mol)} Population diagram",
            fontsize="medium",
            color=fg,
        )

    # ------------------------------------------------------------------
    def _render_highlights(self, ax: Axes, beam_s: float) -> None:
        """Overlay highlighted lines as larger scatter points."""
        if not self.highlight_lines:
            return
        color = self._get_theme_value("active_scatter_line_color", "green")
        e_ups, rd_vals = [], []
        for line_obj, intensity, _ in self.highlight_lines:
            if any(v is None for v in [intensity, line_obj.a_stein, line_obj.g_up, line_obj.lam]):
                continue
            F = intensity * beam_s
            freq = c.SPEED_OF_LIGHT_MICRONS / line_obj.lam
            rd = np.log(4 * np.pi * F / (line_obj.a_stein * c.PLANCK_CONSTANT * freq * line_obj.g_up))
            e_ups.append(line_obj.e_up)
            rd_vals.append(rd)
        if e_ups:
            ax.scatter(e_ups, rd_vals, s=30, color=color, edgecolors="black", zorder=5)

    # ------------------------------------------------------------------
    def set_molecule(self, molecule: "Molecule") -> None:
        """Switch to a different molecule and regenerate."""
        self.molecule = molecule
        self.generate_plot()