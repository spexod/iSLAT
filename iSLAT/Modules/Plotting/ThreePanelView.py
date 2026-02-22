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
        self._needs_refresh: bool = True  # Set True when data changes; cleared after re-render

    # ------------------------------------------------------------------
    # Helpers (private, short-hand access to controller state)
    # ------------------------------------------------------------------
    def _has_tagged_artists(self, tag: str) -> bool:
        """Return True if *ax1* contains any artist with the given tag."""
        for artist in list(self.ax1.lines) + list(self.ax1.texts):
            if hasattr(artist, tag):
                return True
        return False
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

        if self._needs_refresh:
            # Data changed while we were inactive — full re-render
            self._do_update_model_plot()
            self._needs_refresh = False
        else:
            # Simple view toggle — just sync overlay state
            self.sync_toggle_state(self._pm.toggle_state)

        # Restore the span selector and active line selection
        self._restore_line_selection()

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
        self._needs_refresh = False

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
        if self._pm.atomic_toggle:
            self._plot_atomic_lines()

        if self._pm.line_toggle:
            self._plot_saved_lines()

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
        force_rerender: bool = False,
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
            force_rerender=force_rerender,
        )
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Toggle helpers
    # ------------------------------------------------------------------
    def sync_toggle_state(self, toggle_state: dict) -> None:
        """
        Reconcile visual state with the controller's toggle_state dict.

        Ensures overlays match the canonical state without a full re-render.
        """
        # Atomic lines
        if toggle_state.get("atomic_lines", False):
            if not self._has_tagged_artists("_islat_atomic_line"):
                self._plot_atomic_lines()
        else:
            if self._has_tagged_artists("_islat_atomic_line"):
                self._remove_atomic_lines()

        # Saved lines
        if toggle_state.get("saved_lines", False):
            if not self._has_tagged_artists("_islat_saved_line"):
                self._plot_saved_lines()
        else:
            if self._has_tagged_artists("_islat_saved_line"):
                self._remove_saved_lines()

        # Summed spectrum
        self._renderer.set_summed_spectrum_visibility(
            toggle_state.get("summed", True)
        )

        self._canvas.draw_idle()

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
            self._plot_saved_lines(loaded_lines=loaded_lines)
        else:
            self._remove_saved_lines()
        self._canvas.draw_idle()

    def toggle_atomic_lines(self, show: bool) -> None:
        if show:
            self._plot_atomic_lines()
        else:
            self._remove_atomic_lines()
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Atomic / saved line helpers (self-contained via BasePlot)
    # ------------------------------------------------------------------
    def _plot_atomic_lines(self) -> None:
        """Render atomic lines on ax1 using BasePlot helpers.

        Note: does **not** call ``draw_idle()`` — the caller is responsible
        for batching a single draw after all artist mutations are done.
        """
        atomic_data = load_atomic_lines()
        if atomic_data.empty:
            return
        BasePlot._plot_atomic_lines(self.ax1, atomic_data, tag="_islat_atomic_line")

    def _remove_atomic_lines(self) -> None:
        """Remove previously plotted atomic line artists from ax1.

        Note: does **not** call ``draw_idle()`` — the caller is responsible
        for batching a single draw after all artist mutations are done.
        """
        BasePlot._clear_tagged_artists(
            self.ax1, "_islat_atomic_line", lines=True, collections=False, texts=True,
        )

    def _plot_saved_lines(self, loaded_lines: Any = None) -> None:
        """Render saved lines on ax1 using BasePlot helpers.

        Note: does **not** call ``draw_idle()`` — the caller is responsible
        for batching a single draw after all artist mutations are done.
        """
        import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
        if loaded_lines is None:
            loaded_lines = ifh.read_line_saves(file_name=self._islat.input_line_list)
            if loaded_lines.empty:
                return
        theme = self._pm.theme
        BasePlot._plot_saved_line_markers(
            self.ax1,
            loaded_lines,
            tag="_islat_saved_line",
            lam_color=theme.get("saved_line_color", theme.get("saved_line_color_one", "red")),
            range_color=theme.get("saved_line_color_two", "orange"),
        )

    def _remove_saved_lines(self) -> None:
        """Remove previously plotted saved line artists from ax1.

        Note: does **not** call ``draw_idle()`` — the caller is responsible
        for batching a single draw after all artist mutations are done.
        """
        BasePlot._clear_tagged_artists(
            self.ax1, "_islat_saved_line", lines=True, collections=False, texts=False,
        )

    # ------------------------------------------------------------------
    # Selection restoration
    # ------------------------------------------------------------------
    def _restore_line_selection(self) -> None:
        """Restore the span selector and line inspection from toggle_state."""
        sel = self._pm.toggle_state.get("current_selection")
        if sel is not None:
            xmin, xmax = sel
            # Rebuild span selector so it exists on the (possibly-cleared) axes
            self._pm.make_span_selector()
            # Restore the visual span extents
            if hasattr(self._pm, 'interaction_handler'):
                span = getattr(self._pm.interaction_handler, 'span_selector', None)
                if span is not None:
                    try:
                        span.set_visible(True)
                        span.extents = (xmin, xmax)
                        span.update()
                    except Exception:
                        pass
            # Re-run the line inspection / population diagram
            self._pm.onselect(xmin, xmax)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Canvas / drawing
    # ------------------------------------------------------------------
    def draw(self) -> None:
        self._canvas.draw_idle()

    def get_canvas(self) -> "FigureCanvasTkAgg":
        return self._canvas

    def get_figure(self) -> "Figure":
        return self._pm.fig