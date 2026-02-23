"""
PlotView — abstract interface for switchable plot views in the iSLAT GUI.

The iSLATPlot controller owns one *active_view* at a time.  Every user-facing
action (toggle molecule, toggle summed spectrum, toggle legend, …) is
forwarded to the active view's implementation, eliminating scattered
``if is_full_spectrum`` checks throughout the codebase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.Molecule import Molecule

class PlotView(ABC):
    """
    Abstract base for a swappable plot view inside the main iSLAT window.

    Each view owns a matplotlib *Figure* and a Tk *FigureCanvasTkAgg*.
    The controller (:class:`iSLATPlot`) calls these methods without knowing
    which concrete view is active.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    @abstractmethod
    def activate(self, parent_frame: Any) -> None:
        """
        Make this view the visible one.

        Pack / display the view's canvas inside *parent_frame* and ensure
        the rendered content is up-to-date.
        """
        ...

    @abstractmethod
    def deactivate(self) -> None:
        """
        Hide this view (pack_forget the canvas).

        The view should **not** destroy its resources — it may be
        reactivated later.
        """
        ...

    # ------------------------------------------------------------------
    # Core rendering
    # ------------------------------------------------------------------
    @abstractmethod
    def update_model_plot(
        self,
        wave_data: Any,
        flux_data: Any,
        molecules_dict: "MoleculeDict",
        error_data: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """
        Full re-render of the model spectrum (observed + molecules + sum).

        Called when a new spectrum is loaded, molecule parameters change
        globally, or the user switches into this view.
        """
        ...

    @abstractmethod
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
        Lightweight update after a single molecule's visibility is toggled.

        Implementations should **not** reload data from disk — only update
        the artists that changed.

        Parameters
        ----------
        force_rerender : bool
            When *True* the molecule's artists must be re-rendered from
            current parameters (e.g. because parameters changed while the
            molecule was hidden).  Implementations should remove the stale
            artists and create fresh ones instead of just toggling
            visibility.
        """
        ...

    # ------------------------------------------------------------------
    # Toggle helpers
    # ------------------------------------------------------------------
    @abstractmethod
    def sync_toggle_state(self, toggle_state: dict) -> None:
        """
        Reconcile the view's visual state with the canonical *toggle_state*
        dict from the controller.

        Called by the controller when this view is **activated** so that
        overlays (atomic lines, saved lines, summed spectrum, legend) match
        the state the user set while a different view was visible.
        """
        ...

    @abstractmethod
    def toggle_summed_spectrum(self, visible: bool) -> None:
        """Show or hide the summed model fill across all relevant axes."""
        ...

    @abstractmethod
    def toggle_legend(self, visible: Optional[bool] = None) -> None:
        """Toggle the legend visibility on all relevant axes."""
        ...

    @abstractmethod
    def toggle_saved_lines(self, show: bool, loaded_lines: Any = None) -> None:
        """Add or remove saved line annotations."""
        ...

    @abstractmethod
    def toggle_atomic_lines(self, show: bool) -> None:
        """Add or remove atomic line annotations."""
        ...

    # ------------------------------------------------------------------
    # File output
    # ------------------------------------------------------------------
    def save_figure(
        self,
        save_path: str | None = None,
        file_format: str = "pdf",
        dpi: int | None = None,
        rasterized: bool = False,
        **kwargs,
    ) -> str | None:
        """
        Save the current view's figure to a file.

        The default implementation saves the figure returned by
        :meth:`get_figure`.  Subclasses may override this to produce a
        *different* figure for export (e.g. with toggle state baked in).

        Parameters
        ----------
        save_path : str or None
            Destination path.  If *None* a file dialog is opened.
        file_format : str
            File format extension (``"pdf"``, ``"png"``, …).
        dpi : int or None
            Resolution.  *None* uses matplotlib's default.
        rasterized : bool
            If *True* axes are rasterized before saving (useful for PDFs
            with very dense data).
        **kwargs
            Extra keyword arguments forwarded to ``fig.savefig()``.

        Returns
        -------
        str or None
            The path that was saved to, or *None* if the user cancelled.
        """
        from pathlib import Path
        from tkinter import filedialog

        fig = self.get_figure()
        if fig is None:
            return None

        if save_path is None:
            save_path = filedialog.asksaveasfilename(
                title="Save Figure",
                defaultextension=f".{file_format}",
                filetypes=[(f"{file_format.upper()} files", f"*.{file_format}")],
            )
        if not save_path:
            return None

        if rasterized:
            for ax in fig.axes:
                ax.set_rasterized(True)

        save_kw = {"bbox_inches": "tight", "format": file_format}
        if dpi is not None:
            save_kw["dpi"] = dpi
        elif rasterized:
            save_kw["dpi"] = 300
        save_kw.update(kwargs)

        fig.savefig(save_path, **save_kw)
        return save_path

    # ------------------------------------------------------------------
    # Canvas / drawing
    # ------------------------------------------------------------------
    @abstractmethod
    def draw(self) -> None:
        """
        Flush all pending changes to the screen (``canvas.draw_idle()``).
        """
        ...

    @abstractmethod
    def get_canvas(self) -> "FigureCanvasTkAgg":
        """Return the Tk canvas widget for this view."""
        ...

    @abstractmethod
    def get_figure(self) -> "Figure":
        """Return the matplotlib Figure for this view."""
        ...