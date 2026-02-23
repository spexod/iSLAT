"""
MainPlotGrid — Three-panel composite plot for iSLAT spectral analysis.

Layout (GridSpec 2x2):
    Row 0, spanning both columns : full spectrum with molecule overlays
    Row 1, left column           : line inspection (zoomed region)
    Row 1, right column          : population (rotation) diagram

The grid can be used entirely standalone in a notebook / script, or
its axes can be shared with the GUI embedding layer.
"""

from typing import Optional, Tuple, List, Dict, Any, Union, TYPE_CHECKING
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .BasePlot import BasePlot
from .LineInspectionPlot import LineInspectionPlot
from .PopulationDiagramPlot import PopulationDiagramPlot

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine

class MainPlotGrid(BasePlot):
    """
    Three-panel composite plot mirroring the default iSLAT GUI layout.

    Parameters
    ----------
    wave_data : np.ndarray
        Full observed wavelength array.
    flux_data : np.ndarray
        Full observed flux array.
    molecules : MoleculeDict, optional
        Molecules to overlay.
    active_molecule : Molecule, optional
        The *active* molecule used for the population diagram and highlighted
        in the inspection panel.
    error_data : np.ndarray, optional
        Flux uncertainties.
    spectrum_range : tuple[float, float], optional
        ``(xmin, xmax)`` wavelength limits for the top spectrum panel.
        When *None* the full data range is used.  Can be updated later
        via :meth:`set_spectrum_range`.
    inspection_range : tuple[float, float], optional
        ``(xmin, xmax)`` for the line inspection sub-plot. If *None* the
        inspection and population panels are left empty until
        :meth:`set_inspection_range` is called.
    inspection_molecules : list[str] or bool, optional
        Controls which molecules appear in the line-inspection panel:

        - *None* or ``False`` (default) -- only the **active molecule** is
          shown.
        - ``True`` -- all visible molecules from the *MoleculeDict* are shown
          (same behaviour as standalone ``LineInspectionPlot``).
        - A list of molecule name strings (e.g. ``["H2O", "CO"]``) -- only
          those molecules are shown.
    line_data : list, optional
        List of ``(MoleculeLine, intensity, tau)`` tuples for line markers
        in the inspection panel.
    line_list : pd.DataFrame, optional
        Saved-line list for annotations on the main spectrum.
    atomic_lines : pd.DataFrame, optional
        Atomic-line annotations.
    figsize : tuple, optional
        Figure size.  Defaults to ``(15, 8.5)``.
    fig : Figure, optional
        Existing figure for GUI embedding.
    """

    def __init__(
        self,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        molecules: Optional["MoleculeDict"] = None,
        active_molecule: Optional["Molecule"] = None,
        error_data: Optional[np.ndarray] = None,
        spectrum_range: Optional[Tuple[float, float]] = None,
        inspection_range: Optional[Tuple[float, float]] = None,
        inspection_molecules: Optional[Union[bool, List[str]]] = None,
        line_data: Optional[List[Tuple["MoleculeLine", float, Any]]] = None,
        line_list: Optional[pd.DataFrame] = None,
        atomic_lines: Optional[pd.DataFrame] = None,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
        super().__init__(figsize=figsize or (15, 8.5), **kwargs)
        self.wave_data = np.asarray(wave_data)
        self.flux_data = np.asarray(flux_data)
        self.error_data = np.asarray(error_data) if error_data is not None else None
        self.molecules = molecules
        self.active_molecule = active_molecule
        self.spectrum_range = spectrum_range
        self.inspection_range = inspection_range
        self.inspection_molecules = inspection_molecules
        self.line_data = line_data
        self.line_list = line_list
        self.atomic_lines = atomic_lines

        # Panel axes (created in generate_plot)
        self.ax_spectrum: Optional[Axes] = None
        self.ax_inspection: Optional[Axes] = None
        self.ax_popdiagram: Optional[Axes] = None

    # ------------------------------------------------------------------
    def generate_plot(self, **kwargs) -> None:
        """Build the three-panel layout."""
        self._ensure_figure()
        # Clear previous axes so regeneration doesn't stack on top
        self.fig.clf()

        gs = GridSpec(
            2, 2,
            width_ratios=[1, 1],
            height_ratios=[1, 1.5],
            figure=self.fig,
        )
        self.ax_spectrum = self.fig.add_subplot(gs[0, :])
        self.ax_inspection = self.fig.add_subplot(gs[1, 0])
        self.ax_popdiagram = self.fig.add_subplot(gs[1, 1])

        # --- Top panel: full spectrum -----------------------------------
        self._render_spectrum_panel()

        # --- Bottom-left: line inspection --------------------------------
        self._render_inspection_panel()

        # --- Bottom-right: population diagram ----------------------------
        self._render_population_panel()

        # Apply full theme (backgrounds, spines, etc.) to the figure
        self.apply_theme_to_figure()

    # ------------------------------------------------------------------
    # Panel renderers
    # ------------------------------------------------------------------
    def _render_spectrum_panel(self) -> None:
        ax = self.ax_spectrum
        ax.clear()

        if self.wave_data is None or len(self.wave_data) == 0:
            ax.set_title("No spectrum data loaded")
            return

        # Observed spectrum
        self._plot_observed_spectrum(
            ax, self.wave_data, self.flux_data, self.error_data
        )

        # Molecule models + summed
        if self.molecules is not None:
            self._plot_visible_molecules(ax, self.molecules, wave_data=self.wave_data)
            try:
                s_wave, s_flux = self.molecules.get_summed_flux(
                    self.wave_data, visible_only=True
                )
                self._plot_summed_spectrum(ax, s_wave, s_flux)
            except Exception:
                pass

        # Apply spectrum_range if set, otherwise use full data range
        if self.spectrum_range is not None:
            ax.set_xlim(*self.spectrum_range)
            xr = self.spectrum_range
        else:
            xr = (float(np.nanmin(self.wave_data)), float(np.nanmax(self.wave_data)))

        # Line annotations
        ymin = float(ax.get_ylim()[0]) if ax.get_ylim()[0] != 0 else -0.005
        ymax = float(ax.get_ylim()[1])

        if self.line_list is not None:
            self._plot_line_annotations(ax, self.line_list, xr, ymin, ymax)
        if self.atomic_lines is not None:
            self._plot_atomic_lines(ax, self.atomic_lines, xr=xr)

        ax.set_xlabel("Wavelength (μm)")
        ax.set_ylabel("Flux density (Jy)")
        ax.set_title("Full Spectrum with Line Inspection")
        self._update_legend(ax)

    # ------------------------------------------------------------------
    def _render_inspection_panel(self) -> None:
        ax = self.ax_inspection
        ax.clear()

        if self.inspection_range is None:
            ax.set_title("Line Inspection -- select a range")
            return

        xmin, xmax = self.inspection_range

        # Resolve which molecule(s) to show in the inspection panel.
        # Default: only the active molecule.
        lip_molecule = None
        lip_molecules = None

        if self.inspection_molecules is True:
            # Show all visible molecules
            lip_molecules = self.molecules
        elif isinstance(self.inspection_molecules, (list, tuple)):
            # Show a specific subset by name — build a lightweight copy
            if self.molecules is not None:
                from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict as _MD
                lip_molecules = _MD.__new__(_MD)
                dict.__init__(lip_molecules)
                # Initialise the internal caches that MoleculeDict.__init__
                # normally creates — without these, get_visible_molecules()
                # and other helpers will raise AttributeError.
                lip_molecules._visible_molecules = set()
                lip_molecules._summed_flux_cache = {}
                lip_molecules._global_parms = getattr(
                    self.molecules, "_global_parms", {}
                )
                for name in self.inspection_molecules:
                    if name in self.molecules:
                        lip_molecules[name] = self.molecules[name]
            if not lip_molecules:
                lip_molecule = self.active_molecule
                lip_molecules = None
        else:
            # Default (None / False): only the active molecule
            lip_molecule = self.active_molecule

        # Use a temporary LineInspectionPlot (renders onto our axes)
        lip = LineInspectionPlot(
            wave_data=self.wave_data,
            flux_data=self.flux_data,
            xmin=xmin,
            xmax=xmax,
            error_data=self.error_data,
            molecule=lip_molecule,
            molecules=lip_molecules,
            line_data=self.line_data,
            ax=ax,
            theme=self.theme,
        )
        lip.generate_plot()

    # ------------------------------------------------------------------
    def _render_population_panel(self) -> None:
        ax = self.ax_popdiagram
        ax.clear()

        if self.active_molecule is None:
            ax.set_title("Population Diagram -- no molecule selected")
            return

        pdp = PopulationDiagramPlot(
            molecule=self.active_molecule,
            highlight_lines=self.line_data,
            ax=ax,
            theme=self.theme,
        )
        pdp.generate_plot()

    # ------------------------------------------------------------------
    # Public update helpers
    # ------------------------------------------------------------------
    def set_inspection_range(self, xmin: float, xmax: float) -> None:
        """Update the inspection range and refresh the bottom panels."""
        self.inspection_range = (xmin, xmax)
        if self.ax_inspection is not None:
            self._render_inspection_panel()
        if self.ax_popdiagram is not None:
            self._render_population_panel()

    def set_spectrum_range(
        self,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
    ) -> None:
        """Set the wavelength range for the top spectrum panel.

        Pass *None* for both to reset to the full data range.
        """
        if xmin is None and xmax is None:
            self.spectrum_range = None
        else:
            lo = xmin if xmin is not None else float(np.nanmin(self.wave_data))
            hi = xmax if xmax is not None else float(np.nanmax(self.wave_data))
            self.spectrum_range = (lo, hi)
        if self.ax_spectrum is not None:
            self._render_spectrum_panel()

    def set_inspection_molecules(
        self, molecules: Optional[Union[bool, List[str]]] = None,
    ) -> None:
        """Control which molecules appear in the inspection panel.

        Parameters
        ----------
        molecules : None | False | True | list[str]
            *None* / *False* -- only the active molecule (default).
            *True* -- all visible molecules.
            A list of name strings -- only those molecules.
        """
        self.inspection_molecules = molecules
        if self.ax_inspection is not None:
            self._render_inspection_panel()

    def set_active_molecule(self, molecule: "Molecule") -> None:
        """Switch the active molecule and refresh bottom panels."""
        self.active_molecule = molecule
        if self.ax_inspection is not None:
            self._render_inspection_panel()
        if self.ax_popdiagram is not None:
            self._render_population_panel()

    def set_line_data(
        self, line_data: List[Tuple["MoleculeLine", float, Any]]
    ) -> None:
        """Replace the line-data list and refresh the inspection panel."""
        self.line_data = line_data
        if self.ax_inspection is not None:
            self._render_inspection_panel()

    def refresh(self) -> None:
        """Full refresh of all three panels."""
        if self.ax_spectrum is not None:
            self._render_spectrum_panel()
        if self.ax_inspection is not None:
            self._render_inspection_panel()
        if self.ax_popdiagram is not None:
            self._render_population_panel()