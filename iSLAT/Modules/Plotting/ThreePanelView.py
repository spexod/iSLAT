"""
ThreePanelView — :class:`PlotView` implementation for the standard 3-panel layout.

Panels:
    1. Full spectrum overview  (ax1)
    2. Line inspection zoom    (ax2)
    3. Population diagram      (ax3)

This view simply delegates to the *existing* axes and :class:`PlotRenderer`
that already live on the :class:`iSLATPlot` controller.  It adds no new
figures or canvases — it packs / unpacks the original ``self.canvas``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple, List

import numpy as np

from .PlotView import PlotView
from .BasePlot import BasePlot

if TYPE_CHECKING:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from .PlotRenderer import PlotRenderer
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

class ThreePanelView(PlotView):
    """
    Standard 3-panel GUI view backed by the existing PlotRenderer.

    This is a *thin adapter*: it owns nothing new.  The axes, canvas, and
    PlotRenderer are still held by the iSLATPlot controller.  The view
    just provides the :class:`PlotView` API on top.
    """

    def __init__(self, plot_manager: Any) -> None:
        """
        Parameters
        ----------
        plot_manager : iSLATPlot
            The main controller that owns fig, canvas, ax1-3, and PlotRenderer.
        """
        self._pm = plot_manager  # short alias for the controller

    # ------------------------------------------------------------------
    # Helpers (private, short-hand access to controller state)
    # ------------------------------------------------------------------
    @property
    def _renderer(self) -> "PlotRenderer":
        return self._pm.plot_renderer

    @property
    def _islat(self):
        return self._pm.islat

    @property
    def _canvas(self) -> "FigureCanvasTkAgg":
        return self._pm.canvas

    @property
    def ax1(self) -> "Axes":
        return self._pm.ax1

    @property
    def ax2(self) -> "Axes":
        return self._pm.ax2

    @property
    def ax3(self) -> "Axes":
        return self._pm.ax3

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def activate(self, parent_frame: Any) -> None:
        """Show the original 3-panel canvas and refresh."""
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=0, pady=0)
        # Refresh the main plot so it reflects any changes made while the
        # view was inactive (molecules toggled in full-spectrum mode, etc.)
        self._do_update_model_plot()

    def deactivate(self) -> None:
        """Hide the original canvas."""
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
        self._do_update_model_plot()

    def _do_update_model_plot(self) -> None:
        """Internal full re-render mirroring the old ``iSLATPlot.update_model_plot`` logic."""
        islat = self._islat

        if not hasattr(islat, 'molecules_dict') or len(islat.molecules_dict) == 0:
            self._renderer.clear_model_lines()
            self._canvas.draw_idle()
            return

        wave_data = islat.wave_data_original

        # Compute summed flux
        summed_wavelengths = summed_flux = None
        try:
            if hasattr(islat.molecules_dict, 'get_summed_flux'):
                summed_wavelengths, summed_flux = islat.molecules_dict.get_summed_flux(
                    wave_data, visible_only=True
                )
        except Exception as e:
            debug_config.warning("three_panel", f"Could not get summed flux: {e}")

        # Apply RV correction
        wave_data = wave_data - (wave_data / c.SPEED_OF_LIGHT_KMS * islat.molecules_dict.global_stellar_rv)
        islat.wave_data = wave_data

        self._pm.atomic_lines.clear()
        self._pm.saved_lines.clear()

        self._renderer.render_main_spectrum_plot(
            wave_data=wave_data,
            flux_data=islat.flux_data,
            molecules=islat.molecules_dict,
            summed_wavelengths=summed_wavelengths,
            summed_flux=summed_flux,
            error_data=getattr(islat, 'err_data', None),
        )

        # Respect summed_toggle
        if not self._pm.summed_toggle:
            self._renderer.set_summed_spectrum_visibility(False)

        # Overlay saved / atomic lines if their toggles are on
        if islat.GUI.top_bar.atomic_toggle:
            self._pm.plot_atomic_lines()

        if islat.GUI.top_bar.line_toggle:
            self._pm.plot_saved_lines()

        # Recreate span selector and redraw
        self._pm.make_span_selector()
        self._canvas.draw_idle()

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
        Fast incremental update — toggle one molecule's artists on ax1.

        Delegates all heavy-lifting to ``PlotRenderer.handle_molecule_visibility_change``
        which already handles line removal, summed-spectrum update, and legend rebuild.
        """
        self._renderer.handle_molecule_visibility_change(
            molecule_name=molecule_name,
            is_visible=is_visible,
            molecules_dict=molecules_dict,
            wave_data=wave_data,
            active_molecule=active_molecule,
            current_selection=current_selection,
            is_full_spectrum=False,  # We are the three-panel view
        )
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Toggle helpers
    # ------------------------------------------------------------------
    def toggle_summed_spectrum(self, visible: bool) -> None:
        self._renderer.set_summed_spectrum_visibility(visible)
        self._canvas.draw_idle()

    def toggle_legend(self) -> None:
        ax1_leg = self.ax1.get_legend()
        ax2_leg = self.ax2.get_legend()
        if ax1_leg is not None:
            ax1_leg.set_visible(not ax1_leg.get_visible())
        if ax2_leg is not None:
            ax2_leg.set_visible(not ax2_leg.get_visible())
        self._canvas.draw_idle()

    def toggle_saved_lines(self, show: bool, loaded_lines: Any = None) -> None:
        if show:
            self._pm.plot_saved_lines(loaded_lines=loaded_lines)
        else:
            self._pm.remove_saved_lines()

    def toggle_atomic_lines(self, show: bool) -> None:
        if show:
            self._pm.plot_atomic_lines()
        else:
            self._pm.remove_atomic_lines()

    # ------------------------------------------------------------------
    # Canvas / drawing
    # ------------------------------------------------------------------
    def draw(self) -> None:
        self._canvas.draw_idle()

    def get_canvas(self) -> "FigureCanvasTkAgg":
        return self._canvas

    def get_figure(self) -> "Figure":
        return self._pm.fig