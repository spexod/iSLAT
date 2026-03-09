"""
FullSpectrumPlot — Multi-panel overview of an entire observed spectrum.

Generates a vertically stacked series of wavelength-range panels, each
showing the observed data, individual molecule models, summed model
spectrum, and optionally line-list annotations and atomic lines.

Can be used standalone (notebook / script) or embedded in a GUI layout.
The interactive :class:`FullSpectrumView` composes an instance of this
class for rendering, adding span-selectors and toggle sync on top.
"""

from typing import Optional, Tuple, List, Dict, Any, Union, TYPE_CHECKING
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from .BasePlot import BasePlot

if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.Molecule import Molecule
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict

class FullSpectrumPlot(BasePlot):
    """
    Multi-panel full-spectrum plot.

    Parameters
    ----------
    wave_data : np.ndarray
        Observed wavelength array (already RV-corrected if needed).
    flux_data : np.ndarray
        Observed flux array.
    molecules : MoleculeDict, optional
        Collection of molecules — visible ones are plotted.
    error_data : np.ndarray, optional
        Flux uncertainties.
    line_list : pd.DataFrame, optional
        Saved line annotations (columns: ``wave`` / ``lam``, ``species``,
        ``line``).
    atomic_lines : pd.DataFrame, optional
        Atomic line annotations (columns: ``wave``, ``species``, ``line``).
    n_panels : int, optional
        Target number of panels. Defaults to 10.
    step : float, optional
        Wavelength width of each panel. Overrides *n_panels*.
    xlim_range : tuple[float, float], optional
        ``(start, end)`` wavelength range. Defaults to full data range.
    ymax_factor : float, optional
        Fractional padding above the peak flux in each panel (0.2 = 20 %).
    figsize : tuple, optional
        Figure size.  Height is scaled automatically if *None*.
    fig : Figure, optional
        Existing figure for embedding.
    """

    def __init__(
        self,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        molecules: Optional["MoleculeDict"] = None,
        error_data: Optional[np.ndarray] = None,
        line_list: Optional[pd.DataFrame] = None,
        atomic_lines: Optional[pd.DataFrame] = None,
        n_panels: int = 10,
        step: Optional[float] = None,
        xlim_range: Optional[Tuple[float, float]] = None,
        ymax_factor: float = 0.2,
        figsize: Optional[Tuple[float, float]] = None,
        wave_data_obs: Optional[np.ndarray] = None,
        **kwargs,
    ):
        # Defer figsize — calculated once we know the number of panels
        super().__init__(figsize=figsize, **kwargs)
        self.wave_data = np.asarray(wave_data)
        self.flux_data = np.asarray(flux_data)
        # Observer-frame wavelengths for model computation (get_summed_flux,
        # get_matched_sampling_wavelengths).  Falls back to wave_data when
        # no observer-frame array is provided (e.g. notebook usage).
        self.wave_data_obs: np.ndarray = (
            np.asarray(wave_data_obs) if wave_data_obs is not None
            else self.wave_data
        )
        self.molecules = molecules
        self.error_data = np.asarray(error_data) if error_data is not None else None
        self.line_list = line_list
        self.atomic_lines = atomic_lines

        self.n_panels = n_panels
        self.ymax_factor = ymax_factor

        # Wavelength range
        if xlim_range is not None:
            self._xlim_start, self._xlim_end = xlim_range
        else:
            self._xlim_start = float(np.nanmin(self.wave_data))
            self._xlim_end = float(np.nanmax(self.wave_data))

        # Panel step
        if step is not None:
            self._step = step
        else:
            self._step = (self._xlim_end - self._xlim_start) / max(self.n_panels, 1)

        # Pre-compute panel edges
        self._panel_edges: np.ndarray = np.arange(
            self._xlim_start, self._xlim_end, self._step
        )
        # Auto figsize if not given
        if self._figsize is None:
            self._figsize = (12, 1.6 * len(self._panel_edges))

        # Storage for subplot Axes objects
        self.subplots: Dict[int, Axes] = {}

    # ------------------------------------------------------------------
    # Public data mutators (for interactive reuse)
    # ------------------------------------------------------------------
    def update_data(
        self,
        wave_data: np.ndarray,
        flux_data: np.ndarray,
        molecules: Optional["MoleculeDict"] = None,
        error_data: Optional[np.ndarray] = None,
        line_list: Optional[pd.DataFrame] = None,
        atomic_lines: Optional[pd.DataFrame] = None,
        wave_data_obs: Optional[np.ndarray] = None,
    ) -> bool:
        """Replace the data arrays and recompute panel layout.

        Returns *True* if the panel edges changed (caller should rebuild
        subplots); *False* if only the data values changed.
        """
        self.wave_data = np.asarray(wave_data)
        self.flux_data = np.asarray(flux_data)
        self.wave_data_obs = (
            np.asarray(wave_data_obs) if wave_data_obs is not None
            else self.wave_data
        )
        if molecules is not None:
            self.molecules = molecules
        self.error_data = np.asarray(error_data) if error_data is not None else None
        if line_list is not None:
            self.line_list = line_list
        if atomic_lines is not None:
            self.atomic_lines = atomic_lines

        old_edges = self._panel_edges.copy()

        self._xlim_start = float(np.nanmin(self.wave_data))
        self._xlim_end = float(np.nanmax(self.wave_data))
        self._step = (self._xlim_end - self._xlim_start) / max(self.n_panels, 1)
        self._panel_edges = np.arange(self._xlim_start, self._xlim_end, self._step)

        return (
            len(old_edges) != len(self._panel_edges)
            or not np.allclose(old_edges, self._panel_edges, atol=1e-6)
        )

    # ------------------------------------------------------------------
    def generate_plot(self, **kwargs) -> None:
        """Build the multi-panel figure."""
        n = len(self._panel_edges)
        self._ensure_figure()
        # Clear previous axes so regeneration doesn't stack on top
        self.fig.clf()
        self.subplots.clear()

        # Compute summed flux once (if molecules are available)
        summed_wave: Optional[np.ndarray] = None
        summed_flux: Optional[np.ndarray] = None
        if self.molecules is not None:
            try:
                summed_wave, summed_flux = self.molecules.get_summed_flux(
                    self.wave_data_obs, visible_only=True
                )
            except Exception:
                pass

        # Prepare molecule legend info
        mol_labels: List[str] = []
        mol_colors: List[str] = []
        if self.molecules is not None:
            visible = self.molecules.get_visible_molecules(return_objects=True)
            mol_labels = [self.get_molecule_display_name(m) for m in visible]
            mol_colors = [self.get_molecule_color(m) for m in visible]

        # --- Pre-compute molecule flux arrays ONCE (Item 3) -------------
        # Each entry is (wavelength, flux, color, label, mol_name).
        # Slicing per panel is a cheap NumPy mask, avoiding N*M get_flux()
        # calls (N panels x M molecules).
        mol_cache: List[tuple] = []
        if self.molecules is not None:
            # Determine interpolation settings once using observer-frame
            # wavelengths — get_matched_sampling_wavelengths handles the
            # stellar RV correction internally.
            use_interp = False
            target_wave = None
            if self.wave_data_obs is not None and hasattr(self.molecules, 'get_matched_sampling_wavelengths'):
                use_interp, target_wave = self.molecules.get_matched_sampling_wavelengths(self.wave_data_obs)
                if not use_interp:
                    target_wave = None

            for mol in visible:
                lam, flux = self.get_molecule_spectrum_data(
                    mol, self.wave_data,
                    interpolate_to_input=use_interp,
                    target_wavelengths=target_wave,
                )
                if lam is not None and flux is not None and len(flux) > 0:
                    mol_cache.append((
                        lam, flux,
                        self.get_molecule_color(mol),
                        self.get_molecule_display_name(mol),
                        getattr(mol, "name", "unknown"),
                    ))

        for idx, xlim_start in enumerate(self._panel_edges):
            is_last = idx == n - 1
            panel_end = self._xlim_end if is_last else xlim_start + self._step
            xr = (xlim_start, panel_end)

            ax = self.fig.add_subplot(n, 1, idx + 1)
            self.subplots[idx] = ax

            # --- observed spectrum --------------------------------------
            mask = (self.wave_data >= xr[0]) & (self.wave_data <= xr[1])
            panel_wave = self.wave_data[mask]
            panel_flux = self.flux_data[mask]
            panel_err = self.error_data[mask] if self.error_data is not None else None

            self._plot_observed_spectrum(ax, panel_wave, panel_flux, panel_err,
                                        deduplicate=True)

            # y-limits
            if np.any(mask):
                ymax = float(np.nanmax(self.flux_data[mask]))
                ymax += ymax * self.ymax_factor
                ymin = -0.005
            else:
                ymin, ymax = -0.005, 0.1

            # --- molecule models (slice pre-computed data) ---------------
            for m_lam, m_flux, m_color, m_label, m_name in mol_cache:
                m_mask = (m_lam >= xr[0]) & (m_lam <= xr[1])
                if np.any(m_mask):
                    line, = ax.plot(
                        m_lam[m_mask], m_flux[m_mask],
                        linestyle="--", color=m_color,
                        alpha=self._get_theme_value("full_spectrum_model_alpha", 0.8),
                        linewidth=self._get_theme_value("full_spectrum_model_linewidth", 0.8),
                        label=m_label,
                        zorder=self._get_theme_value("zorder_model", 3),
                    )
                    line._molecule_name = m_name

            # --- summed spectrum ----------------------------------------
            if summed_wave is not None and summed_flux is not None:
                s_mask = (summed_wave >= xr[0]) & (summed_wave <= xr[1])
                if np.any(s_mask):
                    self._plot_summed_spectrum(ax, summed_wave[s_mask], summed_flux[s_mask])

            # --- axes config (set BEFORE annotations so they read
            #     correct ylim / xlim for label positioning) -------------
            ax.set_xlim(*xr)
            ax.set_ylim(ymin, ymax)
            ax.tick_params(axis="x", labelsize=7)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

            # --- line annotations ---------------------------------------
            if self.line_list is not None and len(self.line_list) > 0:
                self._plot_line_annotations(ax, self.line_list, xr, ymin, ymax)

            # --- atomic lines -------------------------------------------
            if self.atomic_lines is not None and len(self.atomic_lines) > 0:
                self._plot_atomic_lines(ax, self.atomic_lines, xr=xr)

        # --- global labels & legend ------------------------------------
        fg = self._get_theme_value("foreground", "black")
        if self.subplots:
            last_ax = self.subplots[n - 1]
            last_ax.set_xlabel("Wavelength (μm)", color=fg)
        self.fig.supylabel("Flux Density (Jy)", fontsize=10, color=fg)

        # Colour-legend on the first panel (handles removal when empty).
        if 0 in self.subplots:
            BasePlot.build_molecule_legend(self.subplots[0], mol_labels, mol_colors)

        # Apply full theme (backgrounds, spines, etc.) to the figure
        self.apply_theme_to_figure()

    # ------------------------------------------------------------------
    def update_panels_inplace(self) -> None:
        """Fast in-place update of existing subplot data without fig.clf().

        This is the fast-path used by :class:`FullSpectrumView` when the
        panel layout (edges/count) hasn't changed.  Instead of destroying
        and re-creating every axes object, we update the data on existing
        ``Line2D`` artists and ``PolyCollection`` fills.

        Falls back to a full :meth:`generate_plot` if the subplot dict is
        empty (first render) or structurally mismatched.
        """
        n = len(self._panel_edges)
        if not self.subplots or len(self.subplots) != n:
            # Structural mismatch — fall back to full rebuild
            self.generate_plot()
            return

        # --- Pre-compute molecule data once (same as generate_plot) ------
        summed_wave: Optional[np.ndarray] = None
        summed_flux: Optional[np.ndarray] = None
        if self.molecules is not None:
            try:
                summed_wave, summed_flux = self.molecules.get_summed_flux(
                    self.wave_data_obs, visible_only=True
                )
            except Exception:
                pass

        visible = []
        mol_cache: List[tuple] = []
        if self.molecules is not None:
            visible = self.molecules.get_visible_molecules(return_objects=True)
            use_interp = False
            target_wave = None
            if self.wave_data_obs is not None and hasattr(self.molecules, 'get_matched_sampling_wavelengths'):
                use_interp, target_wave = self.molecules.get_matched_sampling_wavelengths(self.wave_data_obs)
                if not use_interp:
                    target_wave = None
            for mol in visible:
                lam, flux = self.get_molecule_spectrum_data(
                    mol, self.wave_data,
                    interpolate_to_input=use_interp,
                    target_wavelengths=target_wave,
                )
                if lam is not None and flux is not None and len(flux) > 0:
                    mol_cache.append((
                        lam, flux,
                        self.get_molecule_color(mol),
                        self.get_molecule_display_name(mol),
                        getattr(mol, "name", "unknown"),
                    ))

        visible_names = {getattr(m, "name", None) for m in visible}

        # --- Update each panel in place ---------------------------------
        for idx, xlim_start in enumerate(self._panel_edges):
            is_last = idx == n - 1
            panel_end = self._xlim_end if is_last else xlim_start + self._step
            xr = (xlim_start, panel_end)
            ax = self.subplots[idx]

            # Update observed spectrum (tagged with _islat_observed)
            obs_mask = (self.wave_data >= xr[0]) & (self.wave_data <= xr[1])
            panel_wave = self.wave_data[obs_mask]
            panel_flux = self.flux_data[obs_mask]
            obs_updated = False
            for art in ax.lines[:]:
                if hasattr(art, "_islat_observed"):
                    art.set_data(panel_wave, panel_flux)
                    obs_updated = True
                    break
            if not obs_updated:
                # Fallback: create from scratch
                self._plot_observed_spectrum(ax, panel_wave, panel_flux, deduplicate=True)

            # y-limits
            if np.any(obs_mask):
                ymax = float(np.nanmax(self.flux_data[obs_mask]))
                ymax += ymax * self.ymax_factor
                ymin = -0.005
            else:
                ymin, ymax = -0.005, 0.1
            ax.set_ylim(ymin, ymax)

            # Update molecule lines in place
            existing_mol_lines = {}
            for art in ax.lines[:]:
                if hasattr(art, "_molecule_name"):
                    existing_mol_lines[art._molecule_name] = art

            rendered_names = set()
            for m_lam, m_flux, m_color, m_label, m_name in mol_cache:
                m_mask = (m_lam >= xr[0]) & (m_lam <= xr[1])
                rendered_names.add(m_name)
                if m_name in existing_mol_lines:
                    line = existing_mol_lines[m_name]
                    if np.any(m_mask):
                        line.set_data(m_lam[m_mask], m_flux[m_mask])
                        line.set_color(m_color)
                        line.set_visible(True)
                    else:
                        line.set_data([], [])
                else:
                    if np.any(m_mask):
                        new_line, = ax.plot(
                            m_lam[m_mask], m_flux[m_mask],
                            linestyle="--", color=m_color,
                            alpha=self._get_theme_value("full_spectrum_model_alpha", 0.8),
                            linewidth=self._get_theme_value("full_spectrum_model_linewidth", 0.8),
                            label=m_label,
                            zorder=self._get_theme_value("zorder_model", 3),
                        )
                        new_line._molecule_name = m_name

            # Remove stale molecule lines (molecule deleted or hidden)
            for m_name, art in existing_mol_lines.items():
                if m_name not in rendered_names:
                    art.remove()

            # Update summed spectrum fill
            for coll in ax.collections[:]:
                if hasattr(coll, "_islat_summed"):
                    coll.remove()
            if summed_wave is not None and summed_flux is not None:
                s_mask = (summed_wave >= xr[0]) & (summed_wave <= xr[1])
                if np.any(s_mask):
                    self._plot_summed_spectrum(ax, summed_wave[s_mask], summed_flux[s_mask])

        # --- Update legend on first panel -------------------------------
        if 0 in self.subplots:
            mol_labels = [self.get_molecule_display_name(m) for m in visible] if visible else []
            mol_colors = [self.get_molecule_color(m) for m in visible] if visible else []
            BasePlot.build_molecule_legend(self.subplots[0], mol_labels, mol_colors)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def set_line_list(self, df: pd.DataFrame) -> None:
        """Attach or replace the line-list DataFrame."""
        self.line_list = df

    def set_atomic_lines(self, df: pd.DataFrame) -> None:
        """Attach or replace the atomic-lines DataFrame."""
        self.atomic_lines = df