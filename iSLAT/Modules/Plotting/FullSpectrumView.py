"""
FullSpectrumView — :class:`PlotView` implementation for the multi-panel full
spectrum layout.

This view **composes** a :class:`FullSpectrumPlot` (a :class:`BasePlot`
subclass) for all rendering, then adds interactive features on top:

* Span selectors on every panel (click-to-inspect)
* Dynamic overlay toggles (atomic / saved lines, summed spectrum)
* Canvas lifecycle management for the Tk GUI
"""

from __future__ import annotations

import warnings
from os.path import dirname
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
from matplotlib.figure import Figure as MplFigure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import SpanSelector

from .PlotView import PlotView
from .FullSpectrumPlot import FullSpectrumPlot
from .BasePlot import BasePlot

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.legend import Legend
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.Molecule import Molecule

import iSLAT.Constants as c
from iSLAT.Constants import SPEED_OF_LIGHT_KMS
from iSLAT.Modules.FileHandling.iSLATFileHandling import load_atomic_lines
from iSLAT.Modules.FileHandling import absolute_data_files_path
import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh

# Suppress constrained_layout warnings triggered by adding/removing overlay
# artists after the layout engine has already run.
warnings.filterwarnings(
    "ignore",
    message=".*constrained_layout.*",
    category=UserWarning,
    module="matplotlib",
)

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
    Multi-panel full spectrum view backed by :class:`FullSpectrumPlot`.

    Rendering is delegated to the composed *FullSpectrumPlot* (a
    :class:`BasePlot` subclass) which owns the figure, axes, and all
    the standard rendering helpers.  This view adds:

    - Tk canvas management (pack / unpack)
    - Span selectors for click-to-inspect
    - Interactive overlay toggles (atomic lines, saved lines, summed)
    - PDF export (creates a fresh standalone FullSpectrumPlot)
    """

    def __init__(self, plot_manager: Any) -> None:
        self._pm = plot_manager
        self._islat = plot_manager.islat
        self._renderer = plot_manager.plot_renderer
        self._parent_frame: Any = None

        # Canvas — built lazily
        self._canvas: Optional[FigureCanvasTkAgg] = None

        # The composed plot — created on first activation
        self._plot: Optional[FullSpectrumPlot] = None

        # Span selectors for interactive inspection
        self.span_selectors: Dict[int, SpanSelector] = {}

        # Line data (for saved-line annotations in interactive mode)
        self.line_data: Optional[pd.DataFrame] = None

        self._initialised: bool = False  # True after first generate
        self._needs_refresh: bool = True  # Set True when data changes; cleared after re-render

    # ==================================================================
    # Convenience accessors (delegate to composed plot)
    # ==================================================================
    @property
    def fig(self) -> Optional["Figure"]:
        return self._plot.fig if self._plot is not None else None

    @property
    def subplots(self) -> Dict[int, "Axes"]:
        return self._plot.subplots if self._plot is not None else {}

    # ==================================================================
    # Data loading
    # ==================================================================
    def _load_spectrum_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load and RV-correct the observed spectrum. Returns (wave, flux)."""
        spectrum_data = pd.read_csv(self._islat.loaded_spectrum_file, sep=",")
        rv = self._islat.molecules_dict.global_stellar_rv
        wave = spectrum_data["wave"].values
        wave = wave - (wave / SPEED_OF_LIGHT_KMS * rv)
        flux = spectrum_data["flux"].values
        return wave, flux

    def _load_line_data(self) -> Optional[pd.DataFrame]:
        """Load the saved line list (for annotations)."""
        try:
            return pd.read_csv(self._islat.input_line_list, sep=",")
        except Exception:
            return None

    # ==================================================================
    # Plot creation / refresh (delegates to FullSpectrumPlot)
    # ==================================================================
    def _create_plot(self) -> FullSpectrumPlot:
        """Build a fresh :class:`FullSpectrumPlot` from the current iSLAT state."""
        wave, flux = self._load_spectrum_data()
        self.line_data = self._load_line_data()

        # The composed plot handles all rendering via BasePlot helpers.
        # We do NOT pass line_list / atomic_lines here — those are applied
        # dynamically by sync_toggle_state() so they can be toggled.
        plot = FullSpectrumPlot(
            wave_data=wave,
            flux_data=flux,
            molecules=self._islat.molecules_dict,
        )
        return plot

    def _rebuild_plot(self) -> None:
        """Refresh data and regenerate the composed plot.

        If the panel layout changed, the figure is rebuilt from scratch.
        """
        wave, flux = self._load_spectrum_data()
        self.line_data = self._load_line_data()

        if self._plot is None:
            self._plot = self._create_plot()
            self._plot.generate_plot()
            self._install_span_selectors()
            return

        layout_changed = self._plot.update_data(
            wave_data=wave,
            flux_data=flux,
            molecules=self._islat.molecules_dict,
        )

        if layout_changed:
            # Panel edges changed — full rebuild
            self.span_selectors.clear()
            self._plot.generate_plot()
            self._install_span_selectors()
        else:
            # Data changed but layout is the same — regenerate into existing fig
            self.span_selectors.clear()
            self._plot.generate_plot()
            self._install_span_selectors()

    # ==================================================================
    # Span selector (interactive-only feature)
    # ==================================================================
    def _install_span_selectors(self) -> None:
        """Add span selectors to every subplot for click-to-inspect."""
        self.span_selectors.clear()
        for idx, ax in self._plot.subplots.items():
            span = SpanSelector(
                ax,
                lambda xmin, xmax, i=idx: self._on_span_select(xmin, xmax, i),
                direction="horizontal",
                useblit=True,
                props=dict(alpha=0.3, facecolor="lime"),
                interactive=True,
                drag_from_anywhere=True,
            )
            self.span_selectors[idx] = span

    def _on_span_select(self, xmin: float, xmax: float, subplot_index: int) -> None:
        if abs(xmax - xmin) < 0.001:
            return
        main_plot = self._islat.GUI.plot
        if hasattr(main_plot, "is_full_spectrum") and main_plot.is_full_spectrum:
            main_plot.toggle_full_spectrum()
            if hasattr(self._islat, "root"):
                self._islat.root.after(100, lambda: self._apply_selection(xmin, xmax))
            else:
                self._apply_selection(xmin, xmax)
        else:
            self._apply_selection(xmin, xmax)

    def _apply_selection(self, xmin: float, xmax: float) -> None:
        main_plot = self._islat.GUI.plot

        if hasattr(main_plot, "plot_renderer"):
            main_plot.plot_renderer._pop_diagram_molecule = None
            main_plot.plot_renderer._pop_diagram_cache_key = None
            main_plot.plot_renderer._active_scatter_collection = None
            main_plot.plot_renderer._active_scatter_count = 0

        main_plot.current_selection = (xmin, xmax)

        selection_center = (xmin + xmax) / 2
        selection_width = xmax - xmin
        xlim_start = self._plot._xlim_start
        xlim_end = self._plot._xlim_end
        total_range = xlim_end - xlim_start
        min_padding = total_range * 0.025
        view_padding = max(selection_width * 2, min_padding)

        main_plot.ax1.set_xlim(
            selection_center - view_padding,
            selection_center + view_padding,
        )

        if hasattr(main_plot, "interaction_handler"):
            span = getattr(main_plot.interaction_handler, "span_selector", None)
            if span is not None:
                try:
                    span.set_visible(True)
                    span.extents = (xmin, xmax)
                    span.update()
                except Exception as exc:
                    debug_config.warning("full_spectrum_view", f"Could not set span extents: {exc}")

        main_plot.onselect(xmin, xmax)
        main_plot.canvas.draw_idle()

    # ==================================================================
    # Canvas helpers
    # ==================================================================
    def _ensure_canvas(self) -> None:
        """Build (or rebuild) the :class:`FigureCanvasTkAgg`."""
        fig = self.fig
        if self._canvas is not None:
            if fig is not None and self._canvas.figure is fig:
                return
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if fig is not None:
            self._canvas = FigureCanvasTkAgg(fig, master=self._parent_frame)

    # ==================================================================
    # PlotView lifecycle
    # ==================================================================
    def activate(self, parent_frame: Any) -> None:
        self._parent_frame = parent_frame

        if not self._initialised:
            # First activation — build the composed plot
            self._plot = self._create_plot()
            self._plot.generate_plot()
            self._install_span_selectors()
            self._initialised = True
            self._needs_refresh = False
        elif self._needs_refresh:
            # Data changed while we were inactive — full refresh
            old_fig = self.fig
            self._rebuild_plot()
            self._needs_refresh = False
            if self.fig is not old_fig:
                if self._canvas is not None:
                    self._canvas.get_tk_widget().destroy()
                    self._canvas = None
        # else: simple view toggle — just repack the existing canvas

        self._ensure_canvas()
        if self._canvas is not None:
            self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=0, pady=0)

        # Reconcile overlays with the controller's toggle dict
        self.sync_toggle_state(self._pm.toggle_state)

        if self._canvas is not None:
            self._canvas.draw_idle()

    def deactivate(self) -> None:
        if self._canvas is not None:
            self._canvas.get_tk_widget().pack_forget()

    # ==================================================================
    # Core rendering
    # ==================================================================
    def update_model_plot(
        self,
        wave_data: Any = None,
        flux_data: Any = None,
        molecules_dict: "MoleculeDict" = None,
        error_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        if not self._initialised:
            self._needs_refresh = True
            return

        old_fig = self.fig
        self._rebuild_plot()
        self._needs_refresh = False

        if self.fig is not old_fig:
            if self._canvas is not None:
                self._canvas.get_tk_widget().pack_forget()
                self._canvas.get_tk_widget().destroy()
                self._canvas = None
            self._ensure_canvas()
            if self._canvas is not None and self._parent_frame is not None:
                self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=0, pady=0)

        # Apply toggle state overlays after regeneration
        self.sync_toggle_state(self._pm.toggle_state)
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
        force_rerender: bool = False,
    ) -> None:
        if not self._initialised or not self.subplots:
            return

        # 1. Toggle molecule line visibility per subplot.
        #    When force_rerender is True (parameters changed while hidden),
        #    destroy the stale artists and re-create them from fresh data.
        if force_rerender and is_visible:
            molecule = molecules_dict.get(molecule_name)
            for ax in self.subplots.values():
                self._renderer.remove_molecule_lines(
                    molecule_name, ax=ax, lines=list(ax.lines), update_legend=False,
                )
                if molecule is not None:
                    self._renderer.render_individual_molecule_spectrum(
                        molecule, wave_data, subplot=ax, update_legend=False,
                    )
        else:
            for ax in self.subplots.values():
                self._renderer.set_molecule_visibility(
                    molecule_name, is_visible, ax=ax, lines=ax.lines,
                )

        # 2. Recompute summed spectrum
        try:
            summed_wavelengths, summed_flux = molecules_dict.get_summed_flux(
                self._islat.wave_data_original, visible_only=True,
            )
        except Exception as exc:
            debug_config.warning("full_spectrum_view", f"Could not compute summed flux: {exc}")
            summed_wavelengths = self._plot.wave_data if self._plot else np.array([])
            summed_flux = np.zeros_like(summed_wavelengths)

        summed_visible = self._pm.summed_toggle and bool(
            molecules_dict.get_visible_molecules(return_objects=True)
        )

        for ax in self.subplots.values():
            for coll in ax.collections[:]:
                if hasattr(coll, "_islat_summed"):
                    coll.remove()
            xlim = ax.get_xlim()
            mask = (summed_wavelengths >= xlim[0]) & (summed_wavelengths <= xlim[1])
            if np.any(mask) and np.any(summed_flux[mask] > 0):
                fill = ax.fill_between(
                    summed_wavelengths[mask], 0, summed_flux[mask],
                    color=self._renderer._get_theme_value("summed_spectra_color", "lightgray"),
                    alpha=1.0, label="Sum",
                    zorder=self._renderer._get_theme_value("zorder_summed", 1),
                )
                fill._islat_summed = True
                fill.set_visible(summed_visible)

        # 3. Rebuild legend
        self._update_full_spectrum_legend(molecules_dict)
        self.draw()

    # ==================================================================
    # Toggle helpers
    # ==================================================================
    def sync_toggle_state(self, toggle_state: dict) -> None:
        if not self._initialised:
            return

        # Atomic lines
        self._remove_atomic_line_artists()
        if toggle_state.get("atomic_lines", False):
            self._add_atomic_line_artists()

        # Saved lines
        self._remove_saved_line_artists()
        if toggle_state.get("saved_lines", False):
            # Refresh line data from disk so we always reflect the latest file
            self.line_data = self._load_line_data()
            self._add_saved_line_artists()

        # Summed spectrum
        summed_on = toggle_state.get("summed", True)
        for ax in self.subplots.values():
            for coll in ax.collections[:]:
                if hasattr(coll, "_islat_summed"):
                    coll.set_visible(summed_on)

        self.draw()

    def toggle_summed_spectrum(self, visible: bool) -> None:
        if not self._initialised:
            return
        for ax in self.subplots.values():
            for coll in ax.collections[:]:
                if hasattr(coll, "_islat_summed"):
                    coll.set_visible(visible)
        self.draw()

    def toggle_legend(self) -> None:
        if not self._initialised:
            return
        legend = self._get_legend()
        if legend is not None:
            legend.set_visible(not legend.get_visible())
        self.draw()

    def toggle_saved_lines(self, show: bool, loaded_lines: Any = None) -> None:
        if not self._initialised:
            return
        if show:
            # Accept caller-provided data or refresh from disk
            if loaded_lines is not None:
                self.line_data = loaded_lines
            else:
                self.line_data = self._load_line_data()
            self._add_saved_line_artists()
        else:
            self._remove_saved_line_artists()
        self.draw()

    def toggle_atomic_lines(self, show: bool) -> None:
        if not self._initialised:
            return
        if show:
            self._add_atomic_line_artists()
        else:
            self._remove_atomic_line_artists()
        self.draw()

    # ==================================================================
    # Overlay artist helpers (interactive-only)
    # ==================================================================
    def _add_saved_line_artists(self) -> None:
        """Add saved-line annotations to every subplot.

        Delegates to :meth:`BasePlot._plot_line_annotations` with
        ``tag='_islat_saved_line'`` so artists can be removed later.
        """
        if self._plot is None:
            return

        # Reload line data from disk so we always have the latest
        if self.line_data is None:
            self.line_data = self._load_line_data()
        if self.line_data is None:
            return

        # Ensure the 'line' column exists (BasePlot expects it)
        if "line" not in self.line_data.columns:
            self.line_data["line"] = [""] * len(self.line_data)

        for n, ax in self.subplots.items():
            is_last = n == len(self._plot._panel_edges) - 1
            panel_start = self._plot._panel_edges[n]
            panel_end = self._plot._xlim_end if is_last else panel_start + self._plot._step
            xr = (panel_start, panel_end)
            ymin, ymax = ax.get_ylim()

            BasePlot._plot_line_annotations(
                ax, self.line_data, xr, ymin, ymax,
                tag="_islat_saved_line",
            )

    def _remove_saved_line_artists(self) -> None:
        """Remove all ``_islat_saved_line`` artists."""
        for ax in self.subplots.values():
            BasePlot._clear_tagged_artists(
                ax, "_islat_saved_line", lines=True, collections=True, texts=True,
            )

    def _add_atomic_line_artists(self) -> None:
        """Add atomic-line annotations to every subplot.

        Delegates to :meth:`BasePlot._plot_atomic_lines` with
        ``tag='_islat_atomic_line'`` so artists can be removed later.
        No dependency on :class:`PlotRenderer` is needed.
        """
        atomic_data = load_atomic_lines()
        for n, ax in self.subplots.items():
            is_last = n == len(self._plot._panel_edges) - 1
            panel_start = self._plot._panel_edges[n]
            panel_end = self._plot._xlim_end if is_last else panel_start + self._plot._step
            xr = (panel_start, panel_end)
            BasePlot._plot_atomic_lines(
                ax, atomic_data, xr=xr, tag="_islat_atomic_line",
            )

    def _remove_atomic_line_artists(self) -> None:
        """Remove all ``_islat_atomic_line`` artists."""
        for ax in self.subplots.values():
            BasePlot._clear_tagged_artists(
                ax, "_islat_atomic_line", lines=True, collections=False, texts=True,
            )

    # ==================================================================
    # File output  (overrides PlotView.save_figure)
    # ==================================================================
    def save_figure(
        self,
        save_path: str | None = None,
        file_format: str = "pdf",
        dpi: int | None = None,
        rasterized: bool = False,
        **kwargs,
    ) -> str | None:
        """
        Export the full spectrum to a file using a **fresh standalone
        figure** that has the current toggle state baked in.

        This avoids side-effects on the interactive GUI figure.
        """
        from tkinter import filedialog

        if save_path is None:
            try:
                default_name = Path(self._islat.loaded_spectrum_file).stem + f"_full_output.{file_format}"
            except Exception:
                default_name = f"full_output.{file_format}"
            save_path = filedialog.asksaveasfilename(
                title="Save Spectrum Output",
                defaultextension=f".{file_format}",
                initialfile=default_name,
                initialdir=absolute_data_files_path,
                filetypes=[
                    (f"{file_format.upper()} files", f"*.{file_format}"),
                ],
            )
        if not save_path:
            return None

        # Build toggle-state-aware kwargs for the standalone plot
        ts = self._pm.toggle_state

        line_list_df: Optional[pd.DataFrame] = None
        if ts.get("saved_lines", False):
            line_list_df = self.line_data

        atomic_lines_df: Optional[pd.DataFrame] = None
        if ts.get("atomic_lines", False):
            atomic_lines_df = load_atomic_lines()

        # Create standalone figure — uses BasePlot._ensure_figure (non-pyplot)
        wave, flux = self._load_spectrum_data()
        standalone = FullSpectrumPlot(
            wave_data=wave,
            flux_data=flux,
            molecules=self._islat.molecules_dict,
            line_list=line_list_df,
            atomic_lines=atomic_lines_df,
            figsize=(12, 16),
        )
        standalone.generate_plot()

        # Respect summed toggle
        if not ts.get("summed", True):
            for ax in standalone.subplots.values():
                for coll in ax.collections[:]:
                    if hasattr(coll, "_islat_summed"):
                        coll.set_visible(False)

        if rasterized:
            for ax in standalone.fig.axes:
                ax.set_rasterized(True)

        save_kw: dict = {"bbox_inches": "tight", "format": file_format}
        if dpi is not None:
            save_kw["dpi"] = dpi
        elif rasterized:
            save_kw["dpi"] = 300
        save_kw.update(kwargs)

        standalone.fig.savefig(save_path, **save_kw)
        standalone.close()

        # Notify user via data field
        if hasattr(self._islat, "GUI") and hasattr(self._islat.GUI, "data_field"):
            self._islat.GUI.data_field.insert_text(f"Spectrum output saved to: {save_path}")

        return save_path

    # ==================================================================
    # Canvas / drawing
    # ==================================================================
    def draw(self) -> None:
        if self._canvas is not None:
            self._canvas.draw_idle()

    def get_canvas(self) -> "FigureCanvasTkAgg":
        return self._canvas  # type: ignore[return-value]

    def get_figure(self) -> "Figure":
        return self.fig  # type: ignore[return-value]

    # ==================================================================
    # Internal legend helper
    # ==================================================================
    def _get_legend(self) -> Optional["Legend"]:
        if self.subplots and 0 in self.subplots:
            return self.subplots[0].legend_
        return None

    def _update_full_spectrum_legend(self, molecules_dict: "MoleculeDict") -> None:
        visible_mols = molecules_dict.get_visible_molecules(return_objects=True)
        mol_labels = [mol.displaylabel for mol in visible_mols]
        mol_colors = [mol.color for mol in visible_mols]

        legend_ax = self.subplots.get(0)
        if legend_ax is None:
            return

        old_legend = legend_ax.get_legend()
        if old_legend is not None:
            old_legend.remove()

        if mol_labels:
            legend_ax.legend(
                mol_labels,
                labelcolor=mol_colors,
                loc="upper center",
                ncols=12,
                handletextpad=0.2,
                bbox_to_anchor=(0.5, 1.4),
                handlelength=0,
                fontsize=10,
                prop={"weight": "bold"},
            )

    # ==================================================================
    # Cleanup
    # ==================================================================
    def destroy(self) -> None:
        """Permanently dispose of the full spectrum plot and canvas."""
        if self._canvas is not None:
            self._canvas.get_tk_widget().pack_forget()
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self._plot is not None:
            self._plot.close()
            self._plot = None
        self._initialised = False
        self._needs_refresh = True

# ======================================================================
# Backward-compatible top-level function
# ======================================================================
def output_full_spectrum(islat_ref: Any, rasterized: bool = False) -> str | None:
    """
    Create a fresh full-spectrum PDF and a companion parameters CSV.

    This is the backward-compatible replacement for the function that
    previously lived in ``OutputFullSpectrum.py``.
    """
    # Load spectrum data
    spectrum_data = pd.read_csv(islat_ref.loaded_spectrum_file, sep=",")
    rv = islat_ref.molecules_dict.global_stellar_rv
    wave = spectrum_data["wave"].values
    wave = wave - (wave / SPEED_OF_LIGHT_KMS * rv)
    flux = spectrum_data["flux"].values

    # Read toggle state if available
    ts: dict = {}
    if hasattr(islat_ref, "GUI") and hasattr(islat_ref.GUI, "plot"):
        ts = getattr(islat_ref.GUI.plot, "toggle_state", {})

    line_list_df: Optional[pd.DataFrame] = None
    if ts.get("saved_lines", False):
        try:
            line_list_df = pd.read_csv(islat_ref.input_line_list, sep=",")
        except Exception:
            pass

    atomic_lines_df: Optional[pd.DataFrame] = None
    if ts.get("atomic_lines", False):
        atomic_lines_df = load_atomic_lines()

    # Create standalone plot — uses BasePlot._ensure_figure (non-pyplot MplFigure)
    standalone = FullSpectrumPlot(
        wave_data=wave,
        flux_data=flux,
        molecules=islat_ref.molecules_dict,
        line_list=line_list_df,
        atomic_lines=atomic_lines_df,
        figsize=(12, 16),
    )
    standalone.generate_plot()

    # Respect summed toggle
    if not ts.get("summed", True):
        for ax in standalone.subplots.values():
            for coll in ax.collections[:]:
                if hasattr(coll, "_islat_summed"):
                    coll.set_visible(False)

    from tkinter import filedialog
    default_name = Path(islat_ref.loaded_spectrum_file).stem + "_full_output.pdf"
    save_path = filedialog.asksaveasfilename(
        title="Save Spectrum Output",
        defaultextension=".pdf",
        initialfile=default_name,
        initialdir=absolute_data_files_path,
        filetypes=[("PDF files", "*.pdf")],
    )
    if not save_path:
        standalone.close()
        return None

    if rasterized:
        for ax in standalone.fig.axes:
            ax.set_rasterized(True)

    standalone.fig.savefig(
        save_path,
        bbox_inches="tight",
        format="pdf",
        dpi=300 if rasterized else None,
    )
    standalone.close()

    # Write companion parameters CSV
    file_name = str(Path(save_path).with_suffix("")) + "_parameters.csv"
    ifh.write_molecules_to_csv(
        islat_ref.molecules_dict,
        file_path=dirname(save_path),
        file_name=file_name,
    )

    if hasattr(islat_ref, "GUI") and hasattr(islat_ref.GUI, "data_field"):
        islat_ref.GUI.data_field.insert_text(f"Spectrum output saved to: {save_path}")

    return save_path