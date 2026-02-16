"""
FullSpectrumView — :class:`PlotView` implementation for the multi-panel full
spectrum layout.

Wraps :class:`OutputFullSpectrum.FullSpectrumPlot` and provides **lightweight**
visibility / toggle methods that manipulate existing matplotlib artists instead
of reloading CSV data and regenerating every panel from scratch.

Performance contract:
    - ``on_molecule_visibility_changed``  →  O(panels x molecules) artist toggle
    - ``toggle_summed_spectrum``          →  O(panels) collection toggle
    - ``toggle_legend``                   →  O(1) legend visibility toggle
    - ``update_model_plot``               →  full ``reload_data()`` (only when really needed)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from .PlotView import PlotView
from .BasePlot import BasePlot

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.Molecule import Molecule

import iSLAT.Constants as c
from iSLAT.Modules.FileHandling.iSLATFileHandling import load_atomic_lines

# Import debug configuration
try:
    from iSLAT.Modules.Debug import debug_config
except ImportError:
    class _Fallback:
        def verbose(self, *a, **k): pass
        def info(self, *a, **k): pass
        def warning(self, *a, **k): print(f"WARNING: {a}")
        def error(self, *a, **k): print(f"ERROR: {a}")
        def trace(self, *a, **k): pass
    debug_config = _Fallback()

class FullSpectrumView(PlotView):
    """
    Multi-panel full spectrum view that wraps
    :class:`~iSLAT.Modules.FileHandling.OutputFullSpectrum.FullSpectrumPlot`.

    Key improvement over the old approach: molecule visibility changes are
    handled by toggling artists per-panel and updating the summed spectrum,
    rather than calling ``reload_data()`` which rereads the CSV and regenerates
    all ~10 panels.
    """

    def __init__(self, plot_manager: Any) -> None:
        """
        Parameters
        ----------
        plot_manager : iSLATPlot
            The main controller.
        """
        self._pm = plot_manager
        self._islat = plot_manager.islat
        self._renderer = plot_manager.plot_renderer
        self._parent_frame: Any = None

        # Lazily created
        self._fsp: Any = None          # OutputFullSpectrum.FullSpectrumPlot
        self._canvas: Optional[FigureCanvasTkAgg] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_full_spectrum_plot(self) -> None:
        """Create the FullSpectrumPlot instance (and canvas) if needed."""
        if self._fsp is not None:
            return

        from iSLAT.Modules.FileHandling.OutputFullSpectrum import FullSpectrumPlot
        self._fsp = FullSpectrumPlot(self._islat)
        self._fsp.generate_plot()

    def _ensure_canvas(self) -> None:
        """Build (or rebuild) the FigureCanvasTkAgg for the current figure."""
        if self._canvas is not None:
            # Canvas exists — check if the figure reference is still valid
            if self._fsp is not None and self._canvas.figure is self._fsp.fig:
                return
            # Figure was replaced (wavelength range changed) — destroy old canvas
            self._canvas.get_tk_widget().destroy()
            self._canvas = None

        if self._fsp is not None and self._fsp.fig is not None:
            self._canvas = FigureCanvasTkAgg(
                self._fsp.fig,
                master=self._parent_frame,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def activate(self, parent_frame: Any) -> None:
        self._parent_frame = parent_frame

        self._ensure_full_spectrum_plot()

        # If the plot already existed, refresh data (spectrum file may have changed)
        if self._fsp is not None and self._canvas is not None:
            old_fig = self._fsp.fig
            self._fsp.reload_data()
            if self._fsp.fig is not old_fig:
                # Figure was recreated — we need a new canvas
                self._canvas.get_tk_widget().destroy()
                self._canvas = None

        self._ensure_canvas()

        if self._canvas is not None:
            self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=0, pady=0)
            self._canvas.draw_idle()

    def deactivate(self) -> None:
        if self._canvas is not None:
            self._canvas.get_tk_widget().pack_forget()

    # ------------------------------------------------------------------
    # Core rendering
    # ------------------------------------------------------------------
    def update_model_plot(
        self,
        wave_data: Any = None,
        flux_data: Any = None,
        molecules_dict: "MoleculeDict" = None,
        error_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Full re-render — only called for heavy-weight changes like loading a
        new spectrum or global parameter change.
        """
        if self._fsp is None:
            return

        old_fig = self._fsp.fig
        self._fsp.reload_data()

        # If wavelength range changed the figure was recreated
        if self._fsp.fig is not old_fig:
            if self._canvas is not None:
                self._canvas.get_tk_widget().pack_forget()
                self._canvas.get_tk_widget().destroy()
                self._canvas = None
            self._ensure_canvas()
            if self._canvas is not None and self._parent_frame is not None:
                self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=0, pady=0)

        self.draw()

    # ------------------------------------------------------------------
    def on_molecule_visibility_changed(
        self,
        molecule_name: str,
        is_visible: bool,
        molecules_dict: "MoleculeDict",
        wave_data: Any,
        active_molecule: Optional["Molecule"] = None,
        current_selection: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Lightweight per-panel update — toggles artists instead of
        regenerating everything.

        Steps:
            1. Toggle molecule line visibility on every subplot.
            2. Recompute the summed spectrum and update per-panel fills.
            3. Refresh the legend.
        """
        if self._fsp is None or not self._fsp.subplots:
            return

        # --- 1. Toggle molecule line visibility on every subplot ----------
        for ax in self._fsp.subplots.values():
            self._renderer.set_molecule_visibility(
                molecule_name, is_visible, ax=ax, lines=ax.lines,
            )

        # --- 2. Recompute and update summed spectrum per panel -----------
        # Use the full spectrum plot's own wave data (already RV-corrected)
        fsp_wave = self._fsp.wave
        fsp_flux = self._fsp.flux

        # Compute new summed flux once
        try:
            summed_wavelengths, summed_flux = molecules_dict.get_summed_flux(
                self._islat.wave_data_original, visible_only=True,
            )
        except Exception as exc:
            debug_config.warning("full_spectrum_view", f"Could not compute summed flux: {exc}")
            summed_wavelengths = fsp_wave
            summed_flux = np.zeros_like(fsp_wave)

        # Determine if the summed spectrum should be visible
        summed_visible = self._pm.summed_toggle and bool(
            molecules_dict.get_visible_molecules(return_objects=True)
        )

        for ax in self._fsp.subplots.values():
            # Remove old summed fill
            for coll in ax.collections[:]:
                if hasattr(coll, '_islat_summed'):
                    coll.remove()
            # Re-add with current data
            xlim = ax.get_xlim()
            mask = (summed_wavelengths >= xlim[0]) & (summed_wavelengths <= xlim[1])
            if np.any(mask) and np.any(summed_flux[mask] > 0):
                fill = ax.fill_between(
                    summed_wavelengths[mask],
                    0,
                    summed_flux[mask],
                    color=self._renderer._get_theme_value("summed_spectra_color", "lightgray"),
                    alpha=1.0,
                    label='Sum',
                    zorder=self._renderer._get_theme_value("zorder_summed", 1),
                )
                fill._islat_summed = True
                fill.set_visible(summed_visible)

        # --- 3. Rebuild legend -------------------------------------------
        self._update_full_spectrum_legend(molecules_dict)
        self.draw()

    # ------------------------------------------------------------------
    # Toggle helpers
    # ------------------------------------------------------------------
    def toggle_summed_spectrum(self, visible: bool) -> None:
        if self._fsp is None:
            return
        for ax in self._fsp.subplots.values():
            for coll in ax.collections[:]:
                if hasattr(coll, '_islat_summed'):
                    coll.set_visible(visible)
        self.draw()

    def toggle_legend(self) -> None:
        if self._fsp is None:
            return
        legend = self._fsp.get_legend()
        if legend is not None:
            legend.set_visible(not legend.get_visible())
        self.draw()

    def toggle_saved_lines(self, show: bool, loaded_lines: Any = None) -> None:
        if self._fsp is None:
            return
        self._fsp.toggle_saved_lines(show)
        self.draw()

    def toggle_atomic_lines(self, show: bool) -> None:
        if self._fsp is None:
            return
        self._fsp.toggle_atomic_lines(show)
        self.draw()

    # ------------------------------------------------------------------
    # Canvas / drawing
    # ------------------------------------------------------------------
    def draw(self) -> None:
        if self._canvas is not None:
            self._canvas.draw_idle()

    def get_canvas(self) -> "FigureCanvasTkAgg":
        return self._canvas  # type: ignore[return-value]

    def get_figure(self) -> "Figure":
        return self._fsp.fig if self._fsp else None  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal legend helper
    # ------------------------------------------------------------------
    def _update_full_spectrum_legend(self, molecules_dict: "MoleculeDict") -> None:
        """Rebuild the colour-only legend on the first subplot."""
        if self._fsp is None:
            return

        # Re-derive visible molecule info
        visible_mols = molecules_dict.get_visible_molecules(return_objects=True)
        mol_labels = [mol.displaylabel for mol in visible_mols]
        mol_colors = [mol.color for mol in visible_mols]

        legend_ax = getattr(self._fsp, 'legend_subplot', None)
        if legend_ax is None and self._fsp.subplots:
            legend_ax = self._fsp.subplots.get(0)
        if legend_ax is None:
            return

        # Remove old legend
        old_legend = legend_ax.get_legend()
        if old_legend is not None:
            old_legend.remove()

        if mol_labels:
            legend_ax.legend(
                mol_labels,
                labelcolor=mol_colors,
                loc='upper center',
                ncols=12,
                handletextpad=0.2,
                bbox_to_anchor=(0.5, 1.4),
                handlelength=0,
                fontsize=10,
                prop={'weight': 'bold'},
            )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def destroy(self) -> None:
        """Permanently dispose of the full spectrum plot and canvas."""
        if self._canvas is not None:
            self._canvas.get_tk_widget().pack_forget()
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self._fsp is not None:
            self._fsp.close()
            self._fsp = None