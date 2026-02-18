import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import pandas as pd
from pathlib import Path

from iSLAT.Modules.Plotting.FullSpectrumPlot import FullSpectrumPlot
from iSLAT.Modules.FileHandling.iSLATFileHandling import load_atomic_lines
from iSLAT.Constants import SPEED_OF_LIGHT_KMS

from typing import Optional, Any, TYPE_CHECKING
if TYPE_CHECKING:
    from iSLAT.iSLATClass import iSLAT


class FullSpectrumWindow(tk.Toplevel):
    """
    GUI window for displaying and interacting with full spectrum plots.

    Uses the standalone :class:`FullSpectrumPlot` (a :class:`BasePlot`
    subclass) and respects the current GUI toggle state for overlays.
    """

    def __init__(self, parent, islat_ref: "iSLAT", **kwargs):
        super().__init__(parent)
        self.islat_ref = islat_ref
        self.kwargs = kwargs

        self.title("Full Spectrum Output")

        # --- Build the standalone plot with toggle state -----------------
        self.spectrum_plot = self._create_plot(**kwargs)

        # --- UI layout ---------------------------------------------------
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.toolbar_frame = ttk.Frame(self.main_frame)
        self.toolbar_frame.pack(fill=tk.X, pady=(0, 5))

        self.canvas_frame = ttk.Frame(self.main_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.spectrum_plot.generate_plot()
        self.canvas = FigureCanvasTkAgg(self.spectrum_plot.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.nav_toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        self.nav_toolbar.update()

        self._create_toolbar_buttons()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _create_plot(self, **kwargs) -> FullSpectrumPlot:
        """Build a :class:`FullSpectrumPlot` from the current iSLAT state."""
        islat = self.islat_ref

        # Spectrum data
        spectrum_data = pd.read_csv(islat.loaded_spectrum_file, sep=",")
        rv = islat.molecules_dict.global_stellar_rv
        wave = spectrum_data["wave"].values
        wave = wave - (wave / SPEED_OF_LIGHT_KMS * rv)
        flux = spectrum_data["flux"].values

        # Toggle state
        ts: dict = {}
        if hasattr(islat, "GUI") and hasattr(islat.GUI, "plot"):
            ts = getattr(islat.GUI.plot, "toggle_state", {})

        line_list_df: Optional[pd.DataFrame] = None
        if ts.get("saved_lines", False):
            try:
                line_list_df = pd.read_csv(islat.input_line_list, sep=",")
            except Exception:
                pass

        atomic_lines_df: Optional[pd.DataFrame] = None
        if ts.get("atomic_lines", False):
            atomic_lines_df = load_atomic_lines()

        return FullSpectrumPlot(
            wave_data=wave,
            flux_data=flux,
            molecules=islat.molecules_dict,
            line_list=line_list_df,
            atomic_lines=atomic_lines_df,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Toolbar
    # ------------------------------------------------------------------
    def _create_toolbar_buttons(self):
        self.save_btn = ttk.Button(self.toolbar_frame, text="Save as PDF", command=self.save_figure)
        self.save_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.save_png_btn = ttk.Button(self.toolbar_frame, text="Save as PNG", command=self.save_figure_png)
        self.save_png_btn.pack(side=tk.LEFT, padx=(0, 5))

        self.refresh_btn = ttk.Button(self.toolbar_frame, text="Refresh Plot", command=self.refresh_plot)
        self.refresh_btn.pack(side=tk.LEFT, padx=(0, 5))

        ttk.Separator(self.toolbar_frame, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=5)

        self.settings_frame = ttk.LabelFrame(self.toolbar_frame, text="Plot Settings")
        self.settings_frame.pack(side=tk.LEFT, padx=(5, 0))

        ttk.Label(self.settings_frame, text="Step:").grid(row=0, column=0, padx=2, sticky="w")
        self.step_var = tk.DoubleVar(value=self.spectrum_plot._step)
        self.step_spinbox = ttk.Spinbox(
            self.settings_frame, from_=0.5, to=5.0, increment=0.1,
            textvariable=self.step_var, width=8, command=self.update_step,
        )
        self.step_spinbox.grid(row=0, column=1, padx=2)

        ttk.Label(self.settings_frame, text="Y margin:").grid(row=0, column=2, padx=2, sticky="w")
        self.ymax_var = tk.DoubleVar(value=self.spectrum_plot.ymax_factor)
        self.ymax_spinbox = ttk.Spinbox(
            self.settings_frame, from_=0.0, to=1.0, increment=0.05,
            textvariable=self.ymax_var, width=8, command=self.update_ymax_factor,
        )
        self.ymax_spinbox.grid(row=0, column=3, padx=2)

        self.apply_btn = ttk.Button(self.settings_frame, text="Apply", command=self.apply_settings)
        self.apply_btn.grid(row=0, column=4, padx=(5, 2))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def save_figure(self):
        """Save via BasePlot.save (opens dialog if path is None)."""
        default_name = Path(self.islat_ref.loaded_spectrum_file).stem + "_full_output.pdf"
        save_path = filedialog.asksaveasfilename(
            title="Save Spectrum Output",
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF files", "*.pdf")],
        )
        if save_path:
            self.spectrum_plot.save(save_path)
            self.islat_ref.GUI.data_field.insert_text(f"Spectrum output saved to: {save_path}")

    def save_figure_png(self):
        default_name = Path(self.islat_ref.loaded_spectrum_file).stem + "_full_output.png"
        save_path = filedialog.asksaveasfilename(
            title="Save Spectrum Output as PNG",
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
        )
        if save_path:
            self.spectrum_plot.save(save_path, dpi=300)
            self.islat_ref.GUI.data_field.insert_text(f"Spectrum output saved to: {save_path}")

    def refresh_plot(self):
        try:
            figsize = self.spectrum_plot.fig.get_size_inches() if self.spectrum_plot.fig else None
            kw = dict(self.kwargs)
            if figsize is not None:
                kw["figsize"] = tuple(figsize)
            self.spectrum_plot.close()
            self.spectrum_plot = self._create_plot(**kw)
            self.spectrum_plot.generate_plot()
            self.canvas.figure = self.spectrum_plot.fig
            self.canvas.draw_idle()
            self.nav_toolbar.update()
            self.islat_ref.GUI.data_field.insert_text("Full spectrum plot refreshed")
        except Exception as e:
            self.islat_ref.GUI.data_field.insert_text(f"Error refreshing plot: {str(e)}")

    def update_step(self):
        self.spectrum_plot._step = self.step_var.get()
        self.spectrum_plot._panel_edges = np.arange(
            self.spectrum_plot._xlim_start,
            self.spectrum_plot._xlim_end,
            self.spectrum_plot._step,
        )

    def update_ymax_factor(self):
        self.spectrum_plot.ymax_factor = self.ymax_var.get()

    def apply_settings(self):
        try:
            self.update_step()
            self.update_ymax_factor()
            self.refresh_plot()
        except Exception as e:
            self.islat_ref.GUI.data_field.insert_text(f"Error applying settings: {str(e)}")

    def on_closing(self):
        try:
            self.spectrum_plot.close()
        except Exception:
            pass
        self.destroy()
