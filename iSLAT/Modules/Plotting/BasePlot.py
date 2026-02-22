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
from matplotlib.figure import Figure as MplFigure
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
        fig: Optional[MplFigure] = None,
        **kwargs,
    ):
        self._figsize = figsize
        self.theme: Dict[str, Any] = theme if theme is not None else DEFAULT_THEME.copy()
        self.fig: Optional[MplFigure] = fig
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
        interpolate_to_input: bool = False,
        target_wavelengths: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Retrieve ``(wavelength, flux)`` from a *Molecule* object.

        This delegates to ``molecule.get_flux()`` and works regardless of
        whether the molecule has already been computed or not.

        Parameters
        ----------
        molecule : Molecule
            The molecule to query.
        wave_data : np.ndarray, optional
            Wavelength array passed through to ``get_flux``.
        interpolate_to_input : bool, default False
            When True the model flux is resampled onto *target_wavelengths*
            (or *wave_data* when *target_wavelengths* is None).  This is
            used by the matched-spectral-sampling feature.
        target_wavelengths : np.ndarray, optional
            Rest-frame wavelength grid to interpolate onto.  When
            *interpolate_to_input* is True and this is provided, the model
            is resampled to these wavelengths (typically the data wavelengths
            corrected for the global stellar RV) while the returned
            wavelength array is *wave_data* (the observer-frame grid).
        """
        if molecule is None:
            return None, None
        try:
            interp_wave = target_wavelengths if target_wavelengths is not None else wave_data
            lam, flux = molecule.get_flux(
                wavelength_array=interp_wave,
                return_wavelengths=True,
                interpolate_to_input=interpolate_to_input,
            )
            # When we interpolated to rest-frame target wavelengths, return
            # the observer-frame (wave_data) wavelengths so the line is plotted
            # at the correct observed positions.
            if interpolate_to_input and target_wavelengths is not None and wave_data is not None:
                lam = wave_data
            return lam, flux
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
    def _ensure_figure(self, **subplot_kw) -> MplFigure:
        """Create the figure if it doesn't already exist.

        Uses :class:`matplotlib.figure.Figure` directly (not
        ``plt.figure()``) so the figure is **not** registered with the
        pyplot state machine.  This prevents the TkAgg backend from
        creating a hidden figure-manager window that would steal the
        application's taskbar icon.
        """
        if self.fig is None:
            kw: Dict[str, Any] = {"layout": "constrained"}
            if self._figsize is not None:
                kw["figsize"] = self._figsize
            self.fig = MplFigure(**kw)
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
        deduplicate: bool = False,
    ) -> None:
        """Plot observed spectrum data on *ax*.

        Parameters
        ----------
        ax : Axes
            Target axes.
        wave_data, flux_data : np.ndarray
            Observed wavelength / flux arrays.
        error_data : np.ndarray, optional
            Error bars.
        color : str
            Line / marker colour.
        label : str
            Legend label.
        deduplicate : bool
            When *True*, remove any existing artists tagged with
            ``_islat_observed`` before plotting new ones, and tag the
            newly created artists.  Useful in GUI contexts where the
            method may be called repeatedly on the same axes.
        """
        if flux_data is None or len(flux_data) == 0:
            return

        if deduplicate:
            BasePlot._clear_tagged_artists(ax, "_islat_observed")

        if error_data is not None and len(error_data) == len(flux_data):
            container = ax.errorbar(
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
            if deduplicate:
                # Tag every part of the ErrorbarContainer for later removal
                container[0]._islat_observed = True
                for cap in container[1]:
                    cap._islat_observed = True
                for bar_col in container[2]:
                    bar_col._islat_observed = True
        else:
            (line,) = ax.plot(
                wave_data,
                flux_data,
                color=color,
                linewidth=1,
                label=label,
                zorder=self._get_theme_value("zorder_observed", 2),
            )
            if deduplicate:
                line._islat_observed = True

    def _plot_summed_spectrum(
        self,
        ax: Axes,
        wave_data: np.ndarray,
        summed_flux: np.ndarray,
        color: Optional[str] = None,
        label: str = "Sum",
        deduplicate: bool = False,
    ) -> None:
        """Plot the summed model spectrum as a filled area on *ax*.

        Parameters
        ----------
        deduplicate : bool
            When *True*, remove any existing ``_islat_summed``-tagged
            collections before plotting a new fill.
        """
        if summed_flux is None or len(summed_flux) == 0:
            return
        if not np.any(summed_flux > 0):
            return

        if deduplicate:
            BasePlot._clear_tagged_artists(ax, "_islat_summed", lines=False)

        fill_color = color or self._get_theme_value("summed_spectra_color", "lightgray")
        fill = ax.fill_between(
            wave_data,
            0,
            summed_flux,
            color=fill_color,
            alpha=1.0,
            label=label,
            zorder=self._get_theme_value("zorder_summed", 1),
        )
        fill._islat_summed = True  # Tag for toggle-state-aware export

    def _plot_molecule_spectrum(
        self,
        ax: Axes,
        molecule: "Molecule",
        wave_data: Optional[np.ndarray] = None,
        linewidth: float = 1,
        alpha: float = 0.8,
        linestyle: str = "--",
        interpolate_to_input: bool = False,
        target_wavelengths: Optional[np.ndarray] = None,
    ) -> Optional[Line2D]:
        """Plot a single molecule's model spectrum on *ax*."""
        plot_lam, plot_flux = self.get_molecule_spectrum_data(
            molecule, wave_data,
            interpolate_to_input=interpolate_to_input,
            target_wavelengths=target_wavelengths,
        )
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
        # Tag with molecule identity so set_molecule_visibility() can find it
        line._molecule_name = getattr(molecule, "name", "unknown")
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
        """Plot all visible molecules from a *MoleculeDict* on *ax*.

        When *molecules.match_spectral_sampling* is enabled, each molecule's
        model flux is individually resampled to the data wavelength grid
        (corrected for the global stellar RV) before plotting.
        """
        # Determine if we should interpolate to the data pixel grid
        use_interp = False
        target_wave = None
        if wave_data is not None and hasattr(molecules, 'get_matched_sampling_wavelengths'):
            use_interp, target_wave = molecules.get_matched_sampling_wavelengths(wave_data)
            if not use_interp:
                target_wave = None  # don't pass target wavelengths when not interpolating

        visible = molecules.get_visible_molecules(return_objects=True)
        for mol in visible:
            self._plot_molecule_spectrum(
                ax, mol, wave_data=wave_data, linewidth=linewidth, alpha=alpha,
                interpolate_to_input=use_interp, target_wavelengths=target_wave,
            )
        if update_legend:
            self._update_legend(ax)

    @staticmethod
    def _update_legend(ax: Axes) -> None:
        """Add or update the legend on *ax*, excluding invisible artists."""
        handles, labels = ax.get_legend_handles_labels()
        # Filter to only visible artists
        visible_handles = []
        visible_labels = []
        for h, l in zip(handles, labels):
            # ErrorbarContainer and similar containers don't have get_visible()
            # directly — check the first child artist (the data line) instead.
            try:
                is_visible = h.get_visible()
            except AttributeError:
                is_visible = h[0].get_visible() if len(h) > 0 else True
            if is_visible:
                visible_handles.append(h)
                visible_labels.append(l)
        if visible_handles:
            ncols = 2 if len(visible_handles) > 8 else 1
            ax.legend(visible_handles, visible_labels, ncols=ncols)
        else:
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()

    @staticmethod
    def _clear_tagged_artists(
        ax: Axes,
        tag: str,
        *,
        lines: bool = True,
        collections: bool = True,
        texts: bool = False,
    ) -> None:
        """Remove every artist on *ax* that carries the attribute *tag*.

        Parameters
        ----------
        ax : Axes
            Target axes.
        tag : str
            Attribute name to look for (e.g. ``'_islat_observed'``).
        lines : bool
            Search ``ax.lines`` (includes ``Line2D`` artists).
        collections : bool
            Search ``ax.collections`` (includes ``LineCollection``,
            ``PolyCollection``, etc.).
        texts : bool
            Search ``ax.texts``.
        """
        if lines:
            for artist in ax.lines[:]:
                if hasattr(artist, tag):
                    artist.remove()
        if collections:
            for artist in ax.collections[:]:
                if hasattr(artist, tag):
                    artist.remove()
        if texts:
            for artist in ax.texts[:]:
                if hasattr(artist, tag):
                    artist.remove()

    @staticmethod
    def _plot_gaussian_fit(
        ax: Axes,
        gauss_fit: Any,
        fitted_wave: np.ndarray,
        fitted_flux: np.ndarray,
        color: str = "lime",
        linewidth: float = 2,
        zorder: int = 10,
        uncertainty_sigma: float = 3.0,
        fill_alpha: float = 0.3,
    ) -> None:
        """Plot a Gaussian fit result with uncertainty band on *ax*.

        Parameters
        ----------
        ax : Axes
            Target matplotlib Axes.
        gauss_fit : lmfit.model.ModelResult
            The fitted model result (must support ``eval_uncertainty``).
        fitted_wave, fitted_flux : np.ndarray
            Wavelength / flux arrays produced by the fit.
        color : str
            Line and fill colour.
        linewidth : float
            Width of the fit curve.
        zorder : int
            Drawing order for the fit curve.
        uncertainty_sigma : float
            Number of sigma for the uncertainty envelope.
        fill_alpha : float
            Transparency of the uncertainty band.
        """
        if gauss_fit is None or fitted_wave is None or fitted_flux is None:
            return
        ax.plot(
            fitted_wave,
            fitted_flux,
            color=color,
            linewidth=linewidth,
            zorder=zorder,
            linestyle="--",
        )
        try:
            dely = gauss_fit.eval_uncertainty(sigma=uncertainty_sigma)
            ax.fill_between(
                fitted_wave,
                fitted_flux - dely,
                fitted_flux + dely,
                color=color,
                alpha=fill_alpha,
            )
        except Exception:
            pass  # Uncertainty evaluation may fail for some fits

    @staticmethod
    def _plot_line_annotations(
        ax: Axes,
        line_data: pd.DataFrame,
        xr: Tuple[float, float],
        ymin: float,
        ymax: float,
        offset_label: float = 0.003,
        tag: Optional[str] = None,
    ) -> None:
        """
        Draw vertical dotted lines + labels for a line list DataFrame.

        The DataFrame must contain at least ``wave`` (or ``lam``) and
        ``species`` columns.

        Parameters
        ----------
        tag : str, optional
            When provided every created artist is stamped with
            ``setattr(artist, tag, True)`` so callers can later remove
            them with :meth:`_clear_tagged_artists`.
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
                vl = ax.vlines(lam, ymin, ymax, linestyles="dotted", color="grey", linewidth=0.7)
                txt = ax.text(
                    lam + offset_label,
                    ymax,
                    f"{species_arr[i]} {line_id_arr[i]}",
                    fontsize=6,
                    rotation=90,
                    va="top",
                    ha="left",
                    color="grey",
                )
                if tag is not None:
                    setattr(vl, tag, True)
                    setattr(txt, tag, True)

    @staticmethod
    def _plot_atomic_lines(
        ax: Axes,
        atomic_df: pd.DataFrame,
        xr: Optional[Tuple[float, float]] = None,
        tag: Optional[str] = None,
    ) -> None:
        """Draw atomic line markers on *ax*.

        Parameters
        ----------
        tag : str, optional
            When provided every created artist is stamped with
            ``setattr(artist, tag, True)`` so callers can later remove
            them with :meth:`_clear_tagged_artists`.
        """
        if atomic_df is None or len(atomic_df) == 0:
            return
        if xr is not None:
            atomic_df = atomic_df[
                (atomic_df["wave"] >= xr[0]) & (atomic_df["wave"] <= xr[1])
            ]
        for _, row in atomic_df.iterrows():
            line_artist = ax.axvline(row["wave"], linestyle="--", color="tomato", alpha=0.7)
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            text_artist = ax.text(
                row["wave"] + 0.006 * (xlim[1] - xlim[0]),
                ylim[1],
                f"{row['species']} {row.get('line', '')}",
                fontsize=8,
                rotation=90,
                va="top",
                ha="left",
                color="tomato",
            )
            if tag is not None:
                setattr(line_artist, tag, True)
                setattr(text_artist, tag, True)

    @staticmethod
    def _plot_saved_line_markers(
        ax: Axes,
        loaded_lines: pd.DataFrame,
        tag: str = "_islat_saved_line",
        lam_color: str = "red",
        range_color: str = "orange",
        alpha: float = 0.7,
    ) -> None:
        """Plot saved-line markers on *ax* with artist tagging.

        Handles both point markers (``lam`` column) and range markers
        (``xmin`` / ``xmax`` columns).  Every created artist is stamped
        with *tag* so it can be removed later via
        :meth:`_clear_tagged_artists`.

        Parameters
        ----------
        ax : Axes
            Target axes.
        loaded_lines : DataFrame
            Saved-line data.  Expected columns: ``lam`` and optionally
            ``xmin``, ``xmax``.
        tag : str
            Attribute name stamped onto every created artist.
        lam_color, range_color : str
            Colours for the centre-wavelength and range-boundary markers.
        alpha : float
            Transparency of the markers.
        """
        if loaded_lines is None or loaded_lines.empty:
            return
        for _, line in loaded_lines.iterrows():
            if "lam" in line:
                art = ax.axvline(
                    line["lam"], color=lam_color, alpha=alpha,
                    linestyle=":",
                )
                setattr(art, tag, True)
            if "xmin" in line and "xmax" in line:
                for val in (line["xmin"], line["xmax"]):
                    art = ax.axvline(val, color=range_color, alpha=alpha)
                    setattr(art, tag, True)

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
            try:
                # Try pyplot close first (works for pyplot-managed figs)
                plt.close(self.fig)
            except Exception:
                # Figure was created with MplFigure() — not in pyplot,
                # just clear it directly.
                try:
                    self.fig.clear()
                except Exception:
                    pass
        self.fig = None

    def get_figure(self) -> Optional[MplFigure]:
        """Return the underlying matplotlib *Figure*."""
        return self.fig