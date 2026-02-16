"""
PlotView — abstract interface for switchable plot views in the iSLAT GUI.

The iSLATPlot controller owns one *active_view* at a time.  Every user-facing
action (toggle molecule, toggle summed spectrum, toggle legend, …) is
forwarded to the active view's implementation, eliminating scattered
``if is_full_spectrum`` checks throughout the codebase.

Concrete implementations:
    - :class:`ThreePanelView`  — the standard spectrum + inspection + pop-diagram
    - :class:`FullSpectrumView` — multi-panel full spectrum overview
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Tuple

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
    ) -> None:
        """
        Lightweight update after a single molecule's visibility is toggled.

        Implementations should **not** reload data from disk — only update
        the artists that changed.
        """
        ...

    # ------------------------------------------------------------------
    # Toggle helpers
    # ------------------------------------------------------------------
    @abstractmethod
    def toggle_summed_spectrum(self, visible: bool) -> None:
        """Show or hide the summed model fill across all relevant axes."""
        ...

    @abstractmethod
    def toggle_legend(self) -> None:
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