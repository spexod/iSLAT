"""
FullSpectrumView — :class:`PlotView` implementation for the multi-panel full
spectrum layout.

This view is entirely self-contained: it owns the multi-panel figure,
all subplot axes, overlay toggle logic (atomic / saved lines), span
selectors for click-to-inspect, and PDF export.

Performance contract:
    - ``on_molecule_visibility_changed``  →  O(panels × molecules) artist toggle
    - ``toggle_summed_spectrum``          →  O(panels) collection toggle
    - ``toggle_legend``                   →  O(1) legend visibility toggle
    - ``update_model_plot``               →  full regeneration (only when really needed)
"""

from __future__ import annotations

import warnings
from os.path import dirname
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.ticker import MaxNLocator
from matplotlib.widgets import SpanSelector

from .PlotView import PlotView
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
    Multi-panel full spectrum view — entirely self-contained.

    Replaces the old ``OutputFullSpectrum.FullSpectrumPlot`` class.  All
    interactive-panel logic (generate, reload, toggle overlays, span
    selectors) now lives here.  PDF export delegates to the *standalone*
    :class:`~iSLAT.Modules.Plotting.FullSpectrumPlot.FullSpectrumPlot`
    (a :class:`BasePlot` subclass) so toggle state is always baked in.
    """

    def __init__(self, plot_manager: Any) -> None:
        self._pm = plot_manager
        self._islat = plot_manager.islat
        self._renderer = plot_manager.plot_renderer
        self._parent_frame: Any = None

        # Canvas — built lazily
        self._canvas: Optional[FigureCanvasTkAgg] = None

        # ----- internal state (was in OutputFullSpectrum.FullSpectrumPlot) -----
        self.fig: Optional[Figure] = None
        self.subplots: Dict[int, Axes] = {}
        self.span_selectors: Dict[int, SpanSelector] = {}

        self.wave: Optional[np.ndarray] = None
        self.flux: Optional[np.ndarray] = None
        self.line_data: Optional[pd.DataFrame] = None

        self.xlim_start: float = 0.0
        self.xlim_end: float = 1.0
        self.step: float = 1.0
        self.xlim1: Optional[np.ndarray] = None
        self.offset_label: float = 0.003
        self.ymax_factor: float = 0.2

        self.mol_labels: List[str] = []
        self.mol_colors: List[str] = []
        self.visible_molecules: list = []
        self.legend_subplot: Optional[Axes] = None

        self._initialised: bool = False  # True after first generate

    # ==================================================================
    # Data helpers (migrated from OutputFullSpectrum)
    # ==================================================================
    def _load_data(self) -> None:
        """Load spectrum and line data from the iSLAT instance."""
        try:
            spectrum_path = Path(self._islat.loaded_spectrum_file)
            spectrum_data = pd.read_csv(spectrum_path, sep=",")

            rv = self._islat.molecules_dict.global_stellar_rv
            self.wave = spectrum_data["wave"].values
            self.wave = self.wave - (self.wave / SPEED_OF_LIGHT_KMS * rv)
            self.flux = spectrum_data["flux"].values
        except Exception as exc:
            debug_config.error("full_spectrum_view", f"Error loading spectrum data: {exc}")
            raise

        try:
            line_path = Path(self._islat.input_line_list)
            self.line_data = pd.read_csv(line_path, sep=",")
        except Exception:
            self.line_data = None

    def _update_wavelength_ranges(self) -> None:
        """Calculate wavelength panel ranges based on actual data."""
        wave_min = float(np.nanmin(self.wave))
        wave_max = float(np.nanmax(self.wave))
        self.xlim_start = wave_min
        self.xlim_end = wave_max

        wavelength_range = self.xlim_end - self.xlim_start
        target_panels = 10
        self.step = wavelength_range / target_panels
        self.xlim1 = np.arange(self.xlim_start, self.xlim_end, self.step)

    def _prepare_molecule_info(self) -> None:
        """Prepare molecule labels and colours for the legend."""
        mol_dict = self._islat.molecules_dict
        self.visible_molecules = mol_dict.get_visible_molecules(return_objects=True)
        self.mol_labels = [mol.displaylabel for mol in self.visible_molecules]
        self.mol_colors = [mol.color for mol in self.visible_molecules]

    # ==================================================================
    # Plot generation (migrated from OutputFullSpectrum)
    # ==================================================================
    def _generate_plot(self, force_clear: bool = False, figsize: Optional[Tuple[float, float]] = None) -> None:
        """Generate the multi-panel full spectrum figure.

        Overlay lines (atomic, saved) are **not** drawn here — they are
        applied exclusively by :meth:`sync_toggle_state` or the explicit
        ``toggle_*`` methods.
        """
        # Create figure if needed
        if self.fig is None:
            kw: dict = {"layout": "constrained"}
            if figsize is not None:
                kw["figsize"] = figsize
            self.fig = plt.figure(**kw)

        # Get summed flux
        summed_wavelengths, summed_flux = self._islat.molecules_dict.get_summed_flux(
            self._islat.wave_data_original, visible_only=True,
        )

        for n, xlim in enumerate(self.xlim1):
            is_last_panel = n == len(self.xlim1) - 1
            panel_end = self.xlim_end if is_last_panel else self.xlim1[n] + self.step
            xr = [self.xlim1[n], panel_end]

            # Create or reuse subplot
            if n not in self.subplots or self.subplots[n] not in self.fig.axes:
                self.subplots[n] = plt.subplot(len(self.xlim1), 1, n + 1)
                self._setup_span_selector(n)

            # Y-axis limits
            flux_mask = (self.wave > xr[0] - 0.02) & (self.wave < xr[1])
            if np.any(flux_mask):
                maxv = np.nanmax(self.flux[flux_mask])
                ymax = maxv + maxv * self.ymax_factor
                ymin = -0.005
            else:
                ymin, ymax = -0.005, 0.1

            plt.xlim(xr)
            self.subplots[n].xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
            plt.ylim([ymin, ymax])

            # Thinner lines for the multi-panel layout
            original_render_out = self._renderer.render_out
            self._renderer.render_out = True

            is_update = n in self.subplots and self.subplots[n] in self.fig.axes
            should_clear = (not is_update) or force_clear

            self._renderer.render_main_spectrum_plot(
                wave_data=self.wave,
                flux_data=self.flux,
                molecules=self._islat.molecules_dict,
                summed_wavelengths=summed_wavelengths,
                summed_flux=summed_flux,
                axes=self.subplots[n],
                update_legend=False,
                clear_axes=should_clear,
            )

            self._renderer.render_out = original_render_out

        # Hide summed spectrum if toggled off or no visible molecules
        _ts = self._pm.toggle_state
        if (not _ts.get("summed", True)) or not self.visible_molecules:
            for ax in self.subplots.values():
                for coll in ax.collections[:]:
                    if hasattr(coll, "_islat_summed"):
                        coll.set_visible(False)

        # Legend on first panel
        if self.legend_subplot is None:
            self.legend_subplot = self.subplots.get(0)

        if self.mol_labels and self.legend_subplot is not None:
            handles, labels = self.legend_subplot.get_legend_handles_labels()
            if handles:
                self.legend_subplot.legend(
                    self.mol_labels,
                    labelcolor=self.mol_colors,
                    loc="upper center",
                    ncols=12,
                    handletextpad=0.2,
                    bbox_to_anchor=(0.5, 1.4),
                    handlelength=0,
                    fontsize=10,
                    prop={"weight": "bold"},
                )
        elif self.legend_subplot is not None:
            legend = self.legend_subplot.get_legend()
            if legend is not None:
                legend.remove()

        # Axis labels
        self.subplots[len(self.xlim1) - 1].set_xlabel("Wavelength (μm)")
        self.fig.supylabel("Flux Density (Jy)", fontsize=10)
        self.fig.canvas.draw_idle()

    def _reload_data(self, force_clear: bool = False) -> None:
        """Refresh data and regenerate the plot."""
        self._load_data()

        old_xlim_start = self.xlim_start
        old_xlim_end = self.xlim_end
        old_step = self.step
        old_n_panels = len(self.xlim1) if self.xlim1 is not None else 0

        self._update_wavelength_ranges()

        new_n_panels = len(self.xlim1)
        range_changed = (
            old_n_panels != new_n_panels
            or not np.isclose(old_xlim_start, self.xlim_start, atol=1e-6)
            or not np.isclose(old_xlim_end, self.xlim_end, atol=1e-6)
            or not np.isclose(old_step, self.step, atol=1e-6)
        )

        if range_changed:
            self.subplots = {}
            self.span_selectors = {}
            self.legend_subplot = None
            if self.fig is not None:
                self.fig.clear()
                self.fig = None

        self._prepare_molecule_info()
        self._generate_plot(force_clear=force_clear or range_changed)

    # ==================================================================
    # Line list helpers (migrated from OutputFullSpectrum)
    # ==================================================================
    def _plot_line_list(self, ax: Axes, xr: List[float], ymin: float, ymax: float) -> None:
        """Draw saved line annotations on *ax*."""
        if self.line_data is None:
            return
        col = "wave" if "wave" in self.line_data.columns else "lam"
        if col not in self.line_data.columns:
            return
        svd_lamb = np.array(self.line_data[col])
        svd_species = self.line_data["species"]
        if "line" not in self.line_data.columns:
            self.line_data["line"] = [""] * len(self.line_data)
        svd_lineID = np.array(self.line_data["line"])

        for i in range(len(svd_lamb)):
            if xr[0] < svd_lamb[i] < xr[1]:
                line = ax.vlines(
                    svd_lamb[i], ymin, ymax,
                    linestyles="dotted", color="grey", linewidth=0.7,
                )
                line._islat_saved_line = True

                text = ax.text(
                    svd_lamb[i] + self.offset_label, ymax,
                    f"{svd_species[i]} {svd_lineID[i]}",
                    fontsize=6, rotation=90, va="top", ha="left", color="grey",
                )
                text._islat_saved_line = True

    # ==================================================================
    # Overlay toggle helpers (migrated from OutputFullSpectrum)
    # ==================================================================
    def _add_saved_line_artists(self) -> None:
        """Add saved-line annotations to every subplot."""
        for n, xlim in enumerate(self.xlim1):
            if n not in self.subplots:
                continue
            ax = self.subplots[n]
            is_last = n == len(self.xlim1) - 1
            panel_end = self.xlim_end if is_last else self.xlim1[n] + self.step
            xr = [self.xlim1[n], panel_end]
            flux_mask = (self.wave > xr[0] - 0.02) & (self.wave < xr[1])
            if np.any(flux_mask):
                ymax = float(np.nanmax(self.flux[flux_mask])) * (1 + self.ymax_factor)
                ymin = -0.01
            else:
                ymin, ymax = -0.01, 0.15
            self._plot_line_list(ax, xr, ymin, ymax)

    def _remove_saved_line_artists(self) -> None:
        """Remove all ``_islat_saved_line`` artists."""
        for ax in self.subplots.values():
            for coll in ax.collections[:]:
                if hasattr(coll, "_islat_saved_line"):
                    coll.remove()
            for txt in ax.texts[:]:
                if hasattr(txt, "_islat_saved_line"):
                    txt.remove()

    def _add_atomic_line_artists(self) -> None:
        """Add atomic-line annotations to every subplot."""
        atomic_data = load_atomic_lines()
        for n, xlim in enumerate(self.xlim1):
            if n not in self.subplots:
                continue
            ax = self.subplots[n]
            is_last = n == len(self.xlim1) - 1
            panel_end = self.xlim_end if is_last else self.xlim1[n] + self.step
            xr = [self.xlim1[n], panel_end]
            filtered = atomic_data[
                (atomic_data["wave"] >= xr[0]) & (atomic_data["wave"] <= xr[1])
            ]
            if len(filtered) > 0:
                self._renderer.render_atomic_lines(
                    filtered, ax,
                    filtered["wave"].values,
                    filtered["species"].values,
                    filtered["line"].values,
                    using_subplot=True,
                )

    def _remove_atomic_line_artists(self) -> None:
        """Remove all ``_islat_atomic_line`` artists."""
        for ax in self.subplots.values():
            for line in ax.lines[:]:
                if hasattr(line, "_islat_atomic_line"):
                    line.remove()
            for txt in ax.texts[:]:
                if hasattr(txt, "_islat_atomic_line"):
                    txt.remove()

    # ==================================================================
    # Span selector (migrated from OutputFullSpectrum)
    # ==================================================================
    def _setup_span_selector(self, subplot_index: int) -> None:
        if subplot_index not in self.subplots:
            return
        ax = self.subplots[subplot_index]
        span = SpanSelector(
            ax,
            lambda xmin, xmax, idx=subplot_index: self._on_span_select(xmin, xmax, idx),
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.3, facecolor="lime"),
            interactive=True,
            drag_from_anywhere=True,
        )
        self.span_selectors[subplot_index] = span

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
        total_range = self.xlim_end - self.xlim_start
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
        if self._canvas is not None:
            if self.fig is not None and self._canvas.figure is self.fig:
                return
            self._canvas.get_tk_widget().destroy()
            self._canvas = None
        if self.fig is not None:
            self._canvas = FigureCanvasTkAgg(self.fig, master=self._parent_frame)

    # ==================================================================
    # PlotView lifecycle
    # ==================================================================
    def activate(self, parent_frame: Any) -> None:
        self._parent_frame = parent_frame

        if not self._initialised:
            # First activation — load data and build everything
            self._load_data()
            self._update_wavelength_ranges()
            self._prepare_molecule_info()
            self._generate_plot()
            self._initialised = True
        elif self._canvas is not None:
            # Subsequent activation — refresh data
            old_fig = self.fig
            self._reload_data()
            if self.fig is not old_fig:
                self._canvas.get_tk_widget().destroy()
                self._canvas = None

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
            return
        old_fig = self.fig
        self._reload_data()

        if self.fig is not old_fig:
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
        if not self._initialised or not self.subplots:
            return

        # 1. Toggle molecule line visibility per subplot
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
            summed_wavelengths = self.wave
            summed_flux = np.zeros_like(self.wave)

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

        This avoids the problem of the interactive figure not reflecting
        toggle state in the output and avoids side-effects on the GUI
        figure.
        """
        from tkinter import filedialog
        from .FullSpectrumPlot import FullSpectrumPlot as StandaloneFullSpectrum

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

        # Create standalone figure
        standalone = StandaloneFullSpectrum(
            wave_data=self.wave,
            flux_data=self.flux,
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
        if self.legend_subplot is not None:
            return self.legend_subplot.legend_
        return None

    def _update_full_spectrum_legend(self, molecules_dict: "MoleculeDict") -> None:
        visible_mols = molecules_dict.get_visible_molecules(return_objects=True)
        mol_labels = [mol.displaylabel for mol in visible_mols]
        mol_colors = [mol.color for mol in visible_mols]

        legend_ax = self.legend_subplot
        if legend_ax is None and self.subplots:
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
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        self._initialised = False


# ======================================================================
# Backward-compatible top-level function
# ======================================================================
def output_full_spectrum(islat_ref: Any, rasterized: bool = False) -> str | None:
    """
    Create a fresh full-spectrum PDF and a companion parameters CSV.

    This is the backward-compatible replacement for the function that
    previously lived in ``OutputFullSpectrum.py``.
    """
    from .FullSpectrumPlot import FullSpectrumPlot as StandaloneFullSpectrum

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

    standalone = StandaloneFullSpectrum(
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
