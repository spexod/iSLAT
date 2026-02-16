"""
LineListMaker — chainable masking and saving of molecule line-list objects.

Provides a fluent builder API for filtering :class:`MoleculeLineList` data and
exporting to CSV, ``.par``, or pandas DataFrame.  Every filter method
returns ``self`` so calls can be chained::

    (LineListMaker(h2o_lines)
        .filter_wavelength(5.0, 8.0)
        .filter_eup(max_val=4000)
        .filter_astein(min_val=1e-2)
        .to_csv("h2o_filtered.csv"))
"""

from __future__ import annotations

import copy
import warnings
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd

from ..DataTypes import MoleculeLineList

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Core CSV columns in canonical order (matches iSLAT line-list convention)
_CSV_COLUMNS: List[str] = [
    "species", "lev_up", "lev_low", "lam",
    "a_stein", "e_up", "e_low", "g_up", "g_low",
]

# Extended CSV columns (optional)
_CSV_EXTENDED_COLUMNS: List[str] = _CSV_COLUMNS + ["xmin", "xmax"]


def _ensure_dataframe(df_or_linelist: Union[pd.DataFrame, MoleculeLineList],
                       molecule_id: Optional[str] = None) -> pd.DataFrame:
    """Return a DataFrame from either a DataFrame or a MoleculeLineList."""
    if isinstance(df_or_linelist, pd.DataFrame):
        return df_or_linelist.copy()
    if isinstance(df_or_linelist, MoleculeLineList):
        df = df_or_linelist.get_pandas_table()
        if molecule_id is None:
            molecule_id = getattr(df_or_linelist, "molecule_id", None)
        if molecule_id and "species" not in df.columns:
            df.insert(0, "species", molecule_id)
        return df
    raise TypeError(
        f"Expected pd.DataFrame or MoleculeLineList, got {type(df_or_linelist).__name__}"
    )


# ╭──────────────────────────────────────────────────────────────────╮
# │  LineListMaker                                                   │
# ╰──────────────────────────────────────────────────────────────────╯

class LineListMaker:
    """Chainable builder for filtering and exporting spectral line lists.

    Parameters
    ----------
    source : MoleculeLineList or pd.DataFrame
        The line-list data to work with.  A *copy* is made so the original
        object is never modified.
    species : str, optional
        Override the species label written to exports.  If ``None``, the
        value is derived from ``source.molecule_id`` (for a
        ``MoleculeLineList``) or from an existing ``"species"`` column.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        source: Union[MoleculeLineList, pd.DataFrame],
        species: Optional[str] = None,
    ) -> None:
        # Keep a reference if user passed a MoleculeLineList (for .par export)
        self._linelist: Optional[MoleculeLineList] = (
            source if isinstance(source, MoleculeLineList) else None
        )

        # Resolve species
        if species is None and isinstance(source, MoleculeLineList):
            species = getattr(source, "molecule_id", None)
        self._species: Optional[str] = species

        # Build the working DataFrame
        self._df: pd.DataFrame = _ensure_dataframe(source, molecule_id=species)

        # Overwrite species column if explicitly provided
        if species is not None:
            self._df["species"] = species

        # Keep a snapshot of the original unfiltered data for reset()
        self._df_original: pd.DataFrame = self._df.copy()

        # Active filter registry: list of (name, kwargs) tuples
        self._filters: List[Tuple[str, Dict[str, Any]]] = []

    # ------------------------------------------------------------------
    # Repr / info
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # noqa: D105
        species = self._species or "unknown"
        return (
            f"<LineListMaker species={species!r} "
            f"lines={len(self._df)} filters={len(self._filters)}>"
        )

    def __len__(self) -> int:
        """Number of lines after filtering."""
        return len(self._df)

    def summary(self) -> str:
        """Human-readable summary of the current state."""
        lines = [repr(self)]
        if not self._df.empty:
            lam = self._df["lam"]
            lines.append(
                f"  λ range : {lam.min():.5f} - {lam.max():.5f} µm  "
                f"({len(self._df)} lines)"
            )
        if self._filters:
            lines.append("  Active filters:")
            for name, kw in self._filters:
                param_str = ", ".join(f"{k}={v!r}" for k, v in kw.items())
                lines.append(f"    • {name}({param_str})")
        else:
            lines.append("  No filters applied.")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Filter helpers (private)
    # ------------------------------------------------------------------

    def _record_filter(self, name: str, **kwargs: Any) -> None:
        """Store a filter entry for introspection / replay."""
        self._filters.append((name, kwargs))

    def _apply_mask(self, mask: pd.Series) -> "LineListMaker":
        """Apply a boolean mask and return *self* for chaining."""
        self._df = self._df.loc[mask].reset_index(drop=True)
        return self

    # ------------------------------------------------------------------
    # Column range filter (generic)
    # ------------------------------------------------------------------

    def filter_range(
        self,
        column: str,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "LineListMaker":
        """Keep rows where *column* falls within [min_val, max_val].

        Parameters
        ----------
        column : str
            Column name (e.g. ``"lam"``, ``"e_up"``, ``"a_stein"``).
        min_val, max_val : float, optional
            Inclusive bounds.  ``None`` means unbounded.
        """
        if column not in self._df.columns:
            warnings.warn(f"Column {column!r} not in DataFrame — filter skipped.")
            return self
        mask = pd.Series(True, index=self._df.index)
        if min_val is not None:
            mask &= self._df[column] >= min_val
        if max_val is not None:
            mask &= self._df[column] <= max_val
        self._record_filter("filter_range", column=column,
                            min_val=min_val, max_val=max_val)
        return self._apply_mask(mask)

    # ------------------------------------------------------------------
    # Convenience filters (all delegate to filter_range or _apply_mask)
    # ------------------------------------------------------------------

    def filter_wavelength(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "LineListMaker":
        """Filter by wavelength (µm)."""
        self._record_filter("filter_wavelength",
                            min_val=min_val, max_val=max_val)
        # Direct mask (don't double-record via filter_range)
        return self._range_mask("lam", min_val, max_val)

    def filter_eup(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "LineListMaker":
        """Filter by upper-state energy *E_up* (K)."""
        self._record_filter("filter_eup", min_val=min_val, max_val=max_val)
        return self._range_mask("e_up", min_val, max_val)

    def filter_elow(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "LineListMaker":
        """Filter by lower-state energy *E_low* (K)."""
        self._record_filter("filter_elow", min_val=min_val, max_val=max_val)
        return self._range_mask("e_low", min_val, max_val)

    def filter_astein(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "LineListMaker":
        """Filter by Einstein-A coefficient (s⁻¹)."""
        self._record_filter("filter_astein", min_val=min_val, max_val=max_val)
        return self._range_mask("a_stein", min_val, max_val)

    def filter_freq(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "LineListMaker":
        """Filter by frequency (Hz)."""
        self._record_filter("filter_freq", min_val=min_val, max_val=max_val)
        return self._range_mask("freq", min_val, max_val)

    def filter_gup(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "LineListMaker":
        """Filter by upper-state degeneracy."""
        self._record_filter("filter_gup", min_val=min_val, max_val=max_val)
        return self._range_mask("g_up", min_val, max_val)

    def filter_glow(
        self,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
    ) -> "LineListMaker":
        """Filter by lower-state degeneracy."""
        self._record_filter("filter_glow", min_val=min_val, max_val=max_val)
        return self._range_mask("g_low", min_val, max_val)

    # ------------------------------------------------------------------
    # Advanced filters
    # ------------------------------------------------------------------

    def filter_quantum(
        self,
        lev_up: Optional[str] = None,
        lev_low: Optional[str] = None,
        contains: bool = False,
    ) -> "LineListMaker":
        """Filter by quantum-state labels.

        Parameters
        ----------
        lev_up, lev_low : str, optional
            Exact match (or substring if *contains* is ``True``).
        contains : bool
            If ``True``, use substring matching instead of exact equality.
        """
        self._record_filter("filter_quantum",
                            lev_up=lev_up, lev_low=lev_low, contains=contains)
        mask = pd.Series(True, index=self._df.index)
        if lev_up is not None:
            if contains:
                mask &= self._df["lev_up"].str.contains(lev_up, na=False)
            else:
                mask &= self._df["lev_up"] == lev_up
        if lev_low is not None:
            if contains:
                mask &= self._df["lev_low"].str.contains(lev_low, na=False)
            else:
                mask &= self._df["lev_low"] == lev_low
        return self._apply_mask(mask)

    def filter_species(self, *species: str) -> "LineListMaker":
        """Keep only rows matching one of the given species names.

        Useful when working with a DataFrame that contains multiple species.
        """
        self._record_filter("filter_species", species=species)
        if "species" not in self._df.columns:
            warnings.warn("No 'species' column — filter_species skipped.")
            return self
        mask = self._df["species"].isin(species)
        return self._apply_mask(mask)

    def filter_custom(
        self,
        func: Callable[[pd.DataFrame], pd.Series],
        label: str = "custom",
    ) -> "LineListMaker":
        """Apply an arbitrary boolean mask function.

        Parameters
        ----------
        func : callable
            Receives the current DataFrame, must return a boolean Series.
        label : str
            A short description stored in the filter log.

        Example
        -------
        >>> maker.filter_custom(lambda df: df["a_stein"] > df["a_stein"].median(),
        ...                     label="above_median_astein")
        """
        self._record_filter("filter_custom", label=label)
        mask = func(self._df)
        return self._apply_mask(mask)

    # ------------------------------------------------------------------
    # Sort
    # ------------------------------------------------------------------

    def sort(
        self,
        by: Union[str, List[str]] = "lam",
        ascending: bool = True,
    ) -> "LineListMaker":
        """Sort lines by one or more columns.

        Parameters
        ----------
        by : str or list of str
            Column name(s) to sort on.  Default ``"lam"``.
        ascending : bool
            Sort direction.
        """
        self._df = self._df.sort_values(by, ascending=ascending).reset_index(drop=True)
        return self

    # ------------------------------------------------------------------
    # Filter inspection / editing
    # ------------------------------------------------------------------

    @property
    def filters(self) -> List[Tuple[str, Dict[str, Any]]]:
        """Return a *copy* of the active filter log."""
        return list(self._filters)

    @property
    def species(self) -> Optional[str]:
        """Currently set species label."""
        return self._species

    @species.setter
    def species(self, value: str) -> None:
        self._species = value
        if "species" in self._df.columns:
            self._df["species"] = value
        if "species" in self._df_original.columns:
            self._df_original["species"] = value

    def reset(self) -> "LineListMaker":
        """Remove all filters and restore the original data."""
        self._df = self._df_original.copy()
        self._filters.clear()
        return self

    def pop_filter(self) -> "LineListMaker":
        """Remove the last filter and replay the remaining ones.

        This rebuilds the DataFrame from the original snapshot and re-applies
        every filter except the last one.

        Returns
        -------
        LineListMaker
            ``self`` for chaining.

        Raises
        ------
        IndexError
            If no filters are applied.
        """
        if not self._filters:
            raise IndexError("No filters to pop.")
        self._filters.pop()
        return self._replay_filters()

    def remove_filter(self, index: int) -> "LineListMaker":
        """Remove the filter at *index* and replay the rest.

        Parameters
        ----------
        index : int
            Zero-based index into :attr:`filters`.
        """
        del self._filters[index]
        return self._replay_filters()

    # ------------------------------------------------------------------
    # Export — DataFrame
    # ------------------------------------------------------------------

    @property
    def df(self) -> pd.DataFrame:
        """Return a *copy* of the current (filtered) DataFrame."""
        return self._df.copy()

    def to_dataframe(self, include_species: bool = True) -> pd.DataFrame:
        """Return a copy of the current filtered data as a DataFrame.

        Parameters
        ----------
        include_species : bool
            If ``True`` (default), ensure a ``"species"`` column is present.
        """
        df = self._df.copy()
        if include_species and "species" not in df.columns and self._species:
            df.insert(0, "species", self._species)
        return df

    # ------------------------------------------------------------------
    # Export — CSV
    # ------------------------------------------------------------------

    def to_csv(
        self,
        path: Union[str, Path],
        extended: bool = False,
        extra_columns: Optional[Dict[str, Any]] = None,
        sort_by: str = "lam",
        **csv_kwargs: Any,
    ) -> Path:
        """Write the filtered line list to a CSV file.

        Parameters
        ----------
        path : str or Path
            Output file path.
        extended : bool
            If ``True``, include ``xmin`` and ``xmax`` columns (set to
            ``NaN`` if not already present).
        extra_columns : dict, optional
            Additional constant-value columns to append (e.g.
            ``{"note": ""}``).
        sort_by : str
            Column to sort on before writing.  Set to ``""`` to skip.
        **csv_kwargs
            Forwarded to :meth:`pandas.DataFrame.to_csv`.

        Returns
        -------
        Path
            The resolved output path.
        """
        path = Path(path)
        df = self._prepare_export_df(extended=extended,
                                     extra_columns=extra_columns,
                                     sort_by=sort_by)
        csv_kwargs.setdefault("index", False)
        df.to_csv(path, **csv_kwargs)
        return path

    # ------------------------------------------------------------------
    # Export — .par
    # ------------------------------------------------------------------

    def to_par(
        self,
        path: Union[str, Path],
        header: Optional[pd.DataFrame] = None,
    ) -> Path:
        """Write the filtered line list to a ``.par`` file.

        This delegates to :meth:`MoleculeLineList.write_par_file`.  A
        ``MoleculeLineList`` source is required (it carries the partition
        function needed for the ``.par`` header).

        Parameters
        ----------
        path : str or Path
            Output file path.
        header : pd.DataFrame, optional
            Single-row DataFrame overriding header fields (``molecule_id``,
            ``source``, ``molar_mass``, etc.).

        Returns
        -------
        Path
            The resolved output path.

        Raises
        ------
        RuntimeError
            If the maker was not initialised from a ``MoleculeLineList``.
        """
        if self._linelist is None:
            raise RuntimeError(
                "to_par() requires a MoleculeLineList source "
                "(pass one to the constructor)."
            )
        path = Path(path)
        # Build the lines DataFrame in the format write_par_file expects
        lines_df = self._df.drop(columns=["species"], errors="ignore").copy()
        self._linelist.write_par_file(
            file_path=path, header=header, lines_df=lines_df
        )
        return path

    # ------------------------------------------------------------------
    # Export — MoleculeLineList
    # ------------------------------------------------------------------

    def to_linelist(self) -> MoleculeLineList:
        """Create a new :class:`MoleculeLineList` from the filtered data.

        Returns
        -------
        MoleculeLineList
            A fresh instance containing only the filtered lines.
        """
        df = self._df.drop(columns=["species"], errors="ignore")
        lines_data = df.to_dict(orient="records")
        return MoleculeLineList(
            molecule_id=self._species,
            lines_data=lines_data,
        )

    # ------------------------------------------------------------------
    # Combination / merging
    # ------------------------------------------------------------------

    def append(
        self,
        other: Union["LineListMaker", MoleculeLineList, pd.DataFrame],
        species: Optional[str] = None,
    ) -> "LineListMaker":
        """Append lines from *other* to this maker (in-place).

        Parameters
        ----------
        other : LineListMaker, MoleculeLineList, or pd.DataFrame
            Additional lines to append.
        species : str, optional
            Species label for *other* (only used if *other* lacks one).

        Returns
        -------
        LineListMaker
            ``self`` for chaining.
        """
        if isinstance(other, LineListMaker):
            other_df = other._df
        else:
            other_df = _ensure_dataframe(other, molecule_id=species)

        self._df = pd.concat(
            [self._df, other_df], ignore_index=True
        )
        self._record_filter("append", species=species or "unknown")
        return self

    @classmethod
    def merge(
        cls,
        *makers: Union["LineListMaker", MoleculeLineList, pd.DataFrame],
        species_override: Optional[str] = None,
    ) -> "LineListMaker":
        """Merge multiple sources into a single ``LineListMaker``.

        Parameters
        ----------
        *makers : LineListMaker | MoleculeLineList | pd.DataFrame
            Two or more line-list sources.
        species_override : str, optional
            If given, every row receives this species label.

        Returns
        -------
        LineListMaker
            A new maker containing the concatenated data.
        """
        frames: List[pd.DataFrame] = []
        for src in makers:
            if isinstance(src, LineListMaker):
                frames.append(src._df)
            else:
                frames.append(_ensure_dataframe(src))

        combined = pd.concat(frames, ignore_index=True)
        return cls(combined, species=species_override)

    # ------------------------------------------------------------------
    # Copy
    # ------------------------------------------------------------------

    def copy(self) -> "LineListMaker":
        """Return a deep copy of this maker (filters, data, and all)."""
        new = LineListMaker.__new__(LineListMaker)
        new._linelist = self._linelist
        new._species = self._species
        new._df = self._df.copy()
        new._df_original = self._df_original.copy()
        new._filters = copy.deepcopy(self._filters)
        return new

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _range_mask(
        self,
        column: str,
        min_val: Optional[float],
        max_val: Optional[float],
    ) -> "LineListMaker":
        """Build and apply a range mask without double-recording the filter."""
        if column not in self._df.columns:
            warnings.warn(f"Column {column!r} not in DataFrame — filter skipped.")
            return self
        mask = pd.Series(True, index=self._df.index)
        if min_val is not None:
            mask &= self._df[column] >= min_val
        if max_val is not None:
            mask &= self._df[column] <= max_val
        return self._apply_mask(mask)

    def _replay_filters(self) -> "LineListMaker":
        """Reset to original data and replay all stored filters."""
        saved = list(self._filters)
        self._df = self._df_original.copy()
        self._filters.clear()

        # Map filter names to methods
        _method_map: Dict[str, Callable[..., "LineListMaker"]] = {
            "filter_range": self.filter_range,
            "filter_wavelength": self.filter_wavelength,
            "filter_eup": self.filter_eup,
            "filter_elow": self.filter_elow,
            "filter_astein": self.filter_astein,
            "filter_freq": self.filter_freq,
            "filter_gup": self.filter_gup,
            "filter_glow": self.filter_glow,
            "filter_quantum": self.filter_quantum,
            "filter_species": self.filter_species,
        }

        for name, kwargs in saved:
            method = _method_map.get(name)
            if method is not None:
                method(**kwargs)
            elif name == "filter_custom":
                # Custom lambdas are not replayable — warn and skip
                warnings.warn(
                    f"Cannot replay filter_custom(label={kwargs.get('label', '?')!r}); "
                    "it has been dropped."
                )
            # Other entries (e.g. "append") are structural, not replayable
        return self

    def _prepare_export_df(
        self,
        extended: bool = False,
        extra_columns: Optional[Dict[str, Any]] = None,
        sort_by: str = "lam",
    ) -> pd.DataFrame:
        """Build the DataFrame ready for file export."""
        df = self._df.copy()

        # Ensure species column
        if "species" not in df.columns and self._species:
            df.insert(0, "species", self._species)

        # Choose column set
        target_cols = list(_CSV_EXTENDED_COLUMNS if extended else _CSV_COLUMNS)

        # Add xmin/xmax as NaN if they are missing and extended is requested
        if extended:
            for col in ("xmin", "xmax"):
                if col not in df.columns:
                    df[col] = np.nan

        # Extra user columns
        if extra_columns:
            for col_name, col_val in extra_columns.items():
                df[col_name] = col_val
                if col_name not in target_cols:
                    target_cols.append(col_name)

        # Keep any existing columns not in target_cols at the end
        remaining = [c for c in df.columns if c not in target_cols]
        ordered = [c for c in target_cols if c in df.columns] + remaining
        df = df[ordered]

        # Sort
        if sort_by and sort_by in df.columns:
            df = df.sort_values(sort_by).reset_index(drop=True)

        return df

