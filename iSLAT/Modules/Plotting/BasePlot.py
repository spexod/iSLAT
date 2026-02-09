"""
BasePlot — Abstract base class for all iSLAT plot types.

Provides common infrastructure for figure/axes management, theming,
molecule rendering helpers, and show/save functionality. All plot
classes inherit from this so they can work both inside the GUI and
as standalone matplotlib figures in scripts or Jupyter notebooks.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple, List, Union, TYPE_CHECKING
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import iSLAT.Constants as c

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine

# ---------------------------------------------------------------------------
# Default theme (used when no GUI theme is supplied)
# ---------------------------------------------------------------------------
DEFAULT_THEME: Dict[str, Any] = {
    "foreground": "black",
    "background": "white",
    "graph_fill_color": "white",
    "summed_spectra_color": "lightgray",
    "scatter_main_color": "#838B8B",
    "active_scatter_line_color": "green",
    "highlighted_line_color": "yellow",
    "saved_line_color": "red",
    "saved_line_color_one": "red",
    "saved_line_color_two": "orange",
    "default_molecule_color": "blue",
    "zorder_observed": 2,
    "zorder_summed": 1,
    "zorder_model": 3,
}

class BasePlot(ABC):
    """
    Abstract base class for all iSLAT plot types.

    Subclasses must implement :meth:`generate_plot`.  Everything else
    (figure lifecycle, molecule helpers, theming, show/save) is provided
    by the base class.

    Parameters
    ----------
    figsize : tuple, optional
        Figure size in inches ``(width, height)``.
    theme : dict, optional
        Theme dictionary.  Falls back to :data:`DEFAULT_THEME`.
    fig : Figure, optional
        Pre-existing matplotlib Figure to render into (e.g. from a GUI).
        When *None* a new figure will be created on demand.
    """

    def __init__(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        theme: Optional[Dict[str, Any]] = None,
        fig: Optional[Figure] = None,
        **kwargs,
    ):
        self._figsize = figsize
        self.theme: Dict[str, Any] = theme if theme is not None else DEFAULT_THEME.copy()
        self.fig: Optional[Figure] = fig
        self._owns_figure = fig is None  # True when we create the figure ourselves

    # ------------------------------------------------------------------
    # Theme helpers
    # ------------------------------------------------------------------
    def _get_theme_value(self, key: str, default: Any = None) -> Any:
        """Return a value from the theme dict, or *default*."""
        return self.theme.get(key, default)

    # ------------------------------------------------------------------
    # Molecule helpers (shared across all plot types)
    # ------------------------------------------------------------------
    @staticmethod
    def get_molecule_display_name(molecule: "Molecule") -> str:
        """Return the user-facing label for a molecule."""
        return getattr(molecule, "displaylabel", getattr(molecule, "name", "unknown"))

    @staticmethod
    def get_molecule_color(molecule: "Molecule") -> str:
        """Return the colour associated with a molecule."""
        color = getattr(molecule, "color", None)
        return color if color else "blue"

    @staticmethod
    def get_molecule_spectrum_data(
        molecule: "Molecule",
        wave_data: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retrieve ``(wavelength, flux)`` from a *Molecule* object.

        This delegates to ``molecule.get_flux()`` and works regardless of
        whether the molecule has already been computed or not.
        """
        if molecule is None:
            return None, None
        try:
            return molecule.get_flux(
                wavelength_array=wave_data,
                return_wavelengths=True,
                interpolate_to_input=False,
            )
        except Exception as exc:
            print(f"[BasePlot] Could not get flux for "
                  f"{BasePlot.get_molecule_display_name(molecule)}: {exc}")
            return None, None

    @staticmethod
    def get_intensity_data(molecule: "Molecule") -> Optional[pd.DataFrame]:
        """
        Return the Intensity table (DataFrame) from *molecule*.

        Triggers calculation if needed.
        """
        try:
            if molecule is None:
                return None
            if not hasattr(molecule, "intensity") or molecule.intensity is None:
                if hasattr(molecule, "calculate_intensity"):
                    molecule.calculate_intensity()
            intensity_obj = getattr(molecule, "intensity", None)
            if intensity_obj is None:
                return None
            table = getattr(intensity_obj, "get_table", None)
            if table is not None:
                df = table if isinstance(table, pd.DataFrame) else table
                if hasattr(df, "index"):
                    df.index = range(len(df.index))
                return df
            return None
        except Exception as exc:
            print(f"[BasePlot] Error getting intensity data: {exc}")
            return None

    # ------------------------------------------------------------------
    # Figure lifecycle helpers
    # ------------------------------------------------------------------
    def _ensure_figure(self, **subplot_kw) -> Figure:
        """Create the figure if it doesn't already exist."""
        if self.fig is None:
            kw: Dict[str, Any] = {"layout": "constrained"}
            if self._figsize is not None:
                kw["figsize"] = self._figsize
            self.fig = plt.figure(**kw)
            self._owns_figure = True
        return self.fig

    # ------------------------------------------------------------------
    # Rendering helpers (commonly used across sub-classes)
    # ------------------------------------------------------------------
    def _plot_observed_spectrum(
        self,
        ax: Axes,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        error_data: Optional[np.ndarray] = None,
        color: str = "black",
        label: str = "Data",
    ) -> None:
        """Plot observed spectrum data on *ax*."""
        if flux_data is None or len(flux_data) == 0:
            return
        if error_data is not None and len(error_data) == len(flux_data):
            ax.errorbar(
                wave_data,
                flux_data,
                yerr=error_data,
                fmt="-",
                color=color,
                linewidth=1,
                label=label,
                zorder=self._get_theme_value("zorder_observed", 2),
                elinewidth=0.5,
                capsize=0,
            )
        else:
            ax.plot(
                wave_data,
                flux_data,
                color=color,
                linewidth=1,
                label=label,
                zorder=self._get_theme_value("zorder_observed", 2),
            )

    def _plot_summed_spectrum(
        self,
        ax: Axes,
        wave_data: np.ndarray,
        summed_flux: np.ndarray,
        color: Optional[str] = None,
        label: str = "Sum",
    ) -> None:
        """Plot the summed model spectrum as a filled area on *ax*."""
        if summed_flux is None or len(summed_flux) == 0:
            return
        if not np.any(summed_flux > 0):
            return
        fill_color = color or self._get_theme_value("summed_spectra_color", "lightgray")
        ax.fill_between(
            wave_data,
            0,
            summed_flux,
            color=fill_color,
            alpha=1.0,
            label=label,
            zorder=self._get_theme_value("zorder_summed", 1),
        )

    def _plot_molecule_spectrum(
        self,
        ax: Axes,
        molecule: "Molecule",
        wave_data: Optional[np.ndarray] = None,
        linewidth: float = 1,
        alpha: float = 0.8,
        linestyle: str = "--",
    ) -> Optional[Line2D]:
        """Plot a single molecule's model spectrum on *ax*."""
        plot_lam, plot_flux = self.get_molecule_spectrum_data(molecule, wave_data)
        if plot_lam is None or plot_flux is None or len(plot_flux) == 0:
            return None
        (line,) = ax.plot(
            plot_lam,
            plot_flux,
            linestyle=linestyle,
            color=self.get_molecule_color(molecule),
            alpha=alpha,
            linewidth=linewidth,
            label=self.get_molecule_display_name(molecule),
            zorder=self._get_theme_value("zorder_model", 3),
        )
        return line

    def _plot_visible_molecules(
        self,
        ax: Axes,
        molecules: "MoleculeDict",
        wave_data: Optional[np.ndarray] = None,
        linewidth: float = 1,
        alpha: float = 0.8,
        update_legend: bool = True,
    ) -> None:
        """Plot all visible molecules from a *MoleculeDict* on *ax*."""
        visible = molecules.get_visible_molecules(return_objects=True)
        for mol in visible:
            self._plot_molecule_spectrum(
                ax, mol, wave_data=wave_data, linewidth=linewidth, alpha=alpha
            )
        if update_legend:
            self._update_legend(ax)

    def _update_legend(self, ax: Axes) -> None:
        """Add or update the legend on *ax* if there are labelled artists."""
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ncols = 2 if len(handles) > 8 else 1
            ax.legend(ncols=ncols)

    def _plot_line_annotations(
        self,
        ax: Axes,
        line_data: pd.DataFrame,
        xr: Tuple[float, float],
        ymin: float,
        ymax: float,
        offset_label: float = 0.003,
    ) -> None:
        """
        Draw vertical dotted lines + labels for a line list DataFrame.

        The DataFrame must contain at least ``wave`` (or ``lam``) and
        ``species`` columns.
        """
        if line_data is None or len(line_data) == 0:
            return
        col = "wave" if "wave" in line_data.columns else ("lam" if "lam" in line_data.columns else None)
        if col is None:
            return
        lam_arr = line_data[col].values
        species_arr = line_data["species"].values if "species" in line_data.columns else [""] * len(lam_arr)
        line_id_arr = line_data["line"].values if "line" in line_data.columns else [""] * len(lam_arr)

        for i, lam in enumerate(lam_arr):
            if xr[0] < lam < xr[1]:
                ax.vlines(lam, ymin, ymax, linestyles="dotted", color="grey", linewidth=0.7)
                ax.text(
                    lam + offset_label,
                    ymax,
                    f"{species_arr[i]} {line_id_arr[i]}",
                    fontsize=6,
                    rotation=90,
                    va="top",
                    ha="left",
                    color="grey",
                )

    def _plot_atomic_lines(
        self,
        ax: Axes,
        atomic_df: pd.DataFrame,
        xr: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Draw atomic line markers on *ax*."""
        if atomic_df is None or len(atomic_df) == 0:
            return
        if xr is not None:
            atomic_df = atomic_df[
                (atomic_df["wave"] >= xr[0]) & (atomic_df["wave"] <= xr[1])
            ]
        for _, row in atomic_df.iterrows():
            ax.axvline(row["wave"], linestyle="--", color="tomato", alpha=0.7)
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            ax.text(
                row["wave"] + 0.006 * (xlim[1] - xlim[0]),
                ylim[1],
                f"{row['species']} {row.get('line', '')}",
                fontsize=8,
                rotation=90,
                va="top",
                ha="left",
                color="tomato",
            )

    # ------------------------------------------------------------------
    # Abstract API — subclasses must implement
    # ------------------------------------------------------------------
    @abstractmethod
    def generate_plot(self, **kwargs) -> None:
        """Generate or refresh the plot. Subclasses must implement this."""
        ...

    # ------------------------------------------------------------------
    # Public convenience methods
    # ------------------------------------------------------------------
    def show(self, block: bool = False) -> None:
        """Display the plot interactively."""
        if self.fig is None:
            self.generate_plot()
        plt.show(block=block)

    def save(
        self,
        path: Union[str, Path],
        dpi: Optional[int] = None,
        bbox_inches: str = "tight",
        **kwargs,
    ) -> Path:
        """Save the figure to *path*."""
        if self.fig is None:
            self.generate_plot()
        path = Path(path)
        save_kw: Dict[str, Any] = {"bbox_inches": bbox_inches}
        if dpi is not None:
            save_kw["dpi"] = dpi
        save_kw.update(kwargs)
        self.fig.savefig(str(path), **save_kw)
        return path

    def close(self) -> None:
        """Close the figure and free memory."""
        if self.fig is not None and self._owns_figure:
            plt.close(self.fig)
        self.fig = None

    def get_figure(self) -> Optional[Figure]:
        """Return the underlying matplotlib *Figure*."""
        return self.fig