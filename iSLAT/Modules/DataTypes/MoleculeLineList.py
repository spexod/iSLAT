import numpy as np
import time
import os
import hashlib
from collections import namedtuple
from typing import Optional, List, Any, NamedTuple, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas

# Performance logging
from iSLAT.Modules.Debug.PerformanceLogger import perf_log, log_timing, PerformanceSection

#from .MoleculeLine import MoleculeLine

# Lazy imports for performance
_pandas_imported = False

def _get_pandas():
    """Lazy import of pandas"""
    global _pandas_imported, pd
    if not _pandas_imported:
        try:
            import pandas as pd
            _pandas_imported = True
        except ImportError:
            pd = None
            _pandas_imported = True
    return pd

# Cache version - increment when cache format changes
# v2: Switched from compressed npz to separate uncompressed npy files for faster loading
_CACHE_VERSION = 1

class LineTuple(NamedTuple):
    """Named tuple for line data"""
    nr: np.ndarray[int]
    '''Line number'''
    lev_up: np.ndarray[str]
    '''Upper energy level (quantum state label)'''
    lev_low: np.ndarray[str]
    '''Lower energy level (quantum state label)'''
    lam: np.ndarray[float]
    '''Wavelength in microns'''
    freq: np.ndarray[float]
    '''Frequency in Hz'''
    a_stein: np.ndarray[float]
    '''Einstein A coefficient'''
    e_up: np.ndarray[float]
    '''Upper state energy'''
    e_low: np.ndarray[float]
    '''Lower state energy'''
    g_up: np.ndarray[int]
    '''Upper state degeneracy'''
    g_low: np.ndarray[int]
    '''Lower state degeneracy'''

# Structured array dtype for efficient line data storage
# Note: lev_up and lev_low are quantum state labels (strings like '0_0_0|10_2_9')
_LINE_DTYPE = np.dtype([
    ('nr', np.int32),
    ('lev_up', 'U64'),  # Unicode string up to 64 chars for quantum state labels
    ('lev_low', 'U64'), # Unicode string up to 64 chars for quantum state labels
    ('lam', np.float64),
    ('freq', np.float64),
    ('a_stein', np.float64),
    ('e_up', np.float64),
    ('e_low', np.float64),
    ('g_up', np.int32),
    ('g_low', np.int32)
])

class MoleculeLineList:
    """
    Efficient molecular line list with lazy loading and caching.
    """
    __slots__ = ('molecule_id', 'lines', 'partition_function', '_partition_type', 
                 '_lines_type', '_lines_cache', '_lines_cache_valid', '_wavelengths_cache',
                 '_frequencies_cache', '_a_stein_cache', '_e_up_cache', '_e_low_cache',
                 '_g_up_cache', '_g_low_cache', '_data_loaded', '_filename', '_raw_lines_data',
                 '_pandas_df_cache', '_molar_mass')
    
    def __init__(self, molecule_id: Optional[str] = None, filename: Optional[str] = None, 
                 lines_data: Optional[List[dict]] = None):
        """
        Initialize a MoleculeLineList object.

        Parameters
        ----------
        molecule_id : str
            Identifier for the molecule.
        lines_data : list of dict, optional
            List of dictionaries containing line data with keys like 'frequency', 'wavelength', 'intensity', etc.
        filename : str, optional
            Path to a .par file to read molecular data from.
        """
        self.molecule_id = molecule_id
        self.lines: List[Any] = []
        self.partition_function = None
        self._data_loaded = False
        self._filename = filename
        self._raw_lines_data = None
        self._pandas_df_cache = None
        self._molar_mass = None
        
        # Define namedtuple types for data structure
        self._partition_type = namedtuple('partition', ['t', 'q'])
        self._lines_type = LineTuple
        #namedtuple('lines', ['nr', 'lev_up', 'lev_low', 'lam', 'freq', 'a_stein',
                           #                    'e_up', 'e_low', 'g_up', 'g_low'])
        
        # Cache for performance optimization
        self._lines_cache = None
        self._lines_cache_valid = False
        self._wavelengths_cache = None
        self._frequencies_cache = None
        self._a_stein_cache = None
        self._e_up_cache = None
        self._e_low_cache = None
        self._g_up_cache = None
        self._g_low_cache = None
        
        # Lazy loading - only load data when needed
        if filename:
            self._filename = filename
        elif lines_data:
            self._load_from_data(lines_data)

    def _ensure_data_loaded(self):
        """Ensure molecular data is loaded (lazy loading)."""
        if not self._data_loaded and self._filename is not None:
            self._load_from_file(self._filename)
            self._data_loaded = True

    def _load_from_data(self, lines_data: List[dict]):
        """Load from provided line data with optimized batch processing."""
        # Convert list of dicts to structured numpy array for fast column access
        if lines_data:
            n_lines = len(lines_data)
            self._raw_lines_data = np.empty(n_lines, dtype=_LINE_DTYPE)
            
            # Fill structured array from list of dicts
            for i, d in enumerate(lines_data):
                self._raw_lines_data[i] = (
                    d.get('nr', 0),
                    str(d.get('lev_up', '')),
                    str(d.get('lev_low', '')),
                    d.get('lam', 0.0),
                    d.get('freq', 0.0),
                    d.get('a_stein', 0.0),
                    d.get('e_up', 0.0),
                    d.get('e_low', 0.0),
                    d.get('g_up', 0),
                    d.get('g_low', 0)
                )
        else:
            self._raw_lines_data = np.empty(0, dtype=_LINE_DTYPE)
        
        self.lines = None  # Will be created on demand
        self._data_loaded = True
        self._invalidate_caches()

    # ================================
    # Binary Cache System for HITRAN Files
    # ================================
    def _get_cache_path(self, source_filepath: str) -> str:
        """
        Get the cache file path for a given source file.
        
        Parameters
        ----------
        source_filepath : str
            Path to the original .par file
            
        Returns
        -------
        str
            Path to the cache directory for this molecule/file
        """
        from iSLAT.Modules.FileHandling import hitran_cache_folder_path
        
        # Create cache folder if it doesn't exist
        os.makedirs(hitran_cache_folder_path, exist_ok=True)
        
        # Generate cache directory name based on source file name and molecule_id
        source_basename = os.path.basename(source_filepath)
        # Use a directory for v2 cache (multiple .npy files)
        cache_dir = os.path.join(hitran_cache_folder_path, f"{self.molecule_id}_{source_basename}")
        return cache_dir
    
    def _is_cache_valid(self, source_filepath: str, cache_filepath: str) -> bool:
        """
        Check if the cache file is valid (exists and is newer than source).
        
        Parameters
        ----------
        source_filepath : str
            Path to the original .par file
        cache_filepath : str
            Path to the cache directory
            
        Returns
        -------
        bool
            True if cache is valid and can be used
        """
        # For v2 cache, check if the directory and metadata file exist
        metadata_file = os.path.join(cache_filepath, "metadata.npy")
        lines_file = os.path.join(cache_filepath, "lines.npy")
        
        if not os.path.exists(metadata_file) or not os.path.exists(lines_file):
            return False
        
        if not os.path.exists(source_filepath):
            return False
        
        # Check if cache is newer than source file
        cache_mtime = os.path.getmtime(metadata_file)
        source_mtime = os.path.getmtime(source_filepath)
        
        return cache_mtime > source_mtime
    
    def _load_from_cache(self, cache_filepath: str) -> bool:
        """
        Load molecular data from fast binary cache files.
        
        Uses separate .npy files without compression.
        
        Parameters
        ----------
        cache_filepath : str
            Path to the cache directory
            
        Returns
        -------
        bool
            True if cache was loaded successfully, False otherwise
        """
        try:
            metadata_file = os.path.join(cache_filepath, "metadata.npy")
            lines_file = os.path.join(cache_filepath, "lines.npy")
            partition_t_file = os.path.join(cache_filepath, "partition_t.npy")
            partition_q_file = os.path.join(cache_filepath, "partition_q.npy")
            
            # Load metadata (small, fast)
            metadata = np.load(metadata_file, allow_pickle=False)
            cache_version = int(metadata[0])
            
            if cache_version != _CACHE_VERSION:
                print(f"Cache version mismatch (got {cache_version}, expected {_CACHE_VERSION}), rebuilding...")
                return False
            
            self._molar_mass = float(metadata[1])
            
            # Load partition function (small arrays, fast)
            partition_t = np.load(partition_t_file, allow_pickle=False)
            partition_q = np.load(partition_q_file, allow_pickle=False)
            self.partition_function = self._partition_type(t=partition_t, q=partition_q)
            
            # Use memory mapping for very large files, direct load for smaller ones
            file_size = os.path.getsize(lines_file)
            if file_size > 25_000_000:  # > 25MB: use memory mapping
                self._raw_lines_data = np.load(lines_file, mmap_mode='r', allow_pickle=False)
            else:
                self._raw_lines_data = np.load(lines_file, allow_pickle=False)
            
            self.lines = None  # Will be created on demand
            self._data_loaded = True
            self._invalidate_caches()
            
            return True
            
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return False
    
    def _save_to_cache(self, cache_filepath: str) -> bool:
        """
        Save molecular data to fast binary cache files.
        
        Uses separate uncompressed .npy files for maximum load speed.
        
        Parameters
        ----------
        cache_filepath : str
            Path to the cache directory
            
        Returns
        -------
        bool
            True if cache was saved successfully, False otherwise
        """
        try:
            # Create cache directory
            os.makedirs(cache_filepath, exist_ok=True)
            
            # Prepare partition function arrays
            partition_t = np.array(self.partition_function.t, dtype=np.float64) if self.partition_function else np.array([], dtype=np.float64)
            partition_q = np.array(self.partition_function.q, dtype=np.float64) if self.partition_function else np.array([], dtype=np.float64)
            
            # Save metadata (version + molar mass)
            metadata = np.array([_CACHE_VERSION, self._molar_mass if self._molar_mass else 0.0], dtype=np.float64)
            np.save(os.path.join(cache_filepath, "metadata.npy"), metadata)
            
            # Save partition function arrays
            np.save(os.path.join(cache_filepath, "partition_t.npy"), partition_t)
            np.save(os.path.join(cache_filepath, "partition_q.npy"), partition_q)
            
            # Save lines data - the main payload
            np.save(os.path.join(cache_filepath, "lines.npy"), self._raw_lines_data)
            
            return True
            
        except Exception as e:
            print(f"Failed to save cache: {e}")
            return False

    def _load_from_file(self, filename: str):
        """
        Load molecular data from a .par file, using binary cache if available.
        
        Parameters
        ----------
        filename : str
            Path to the .par file
        """
        section = PerformanceSection(f"MoleculeLineList._load_from_file({self.molecule_id})")
        section.start()
        
        # Resolve full path for cache lookup
        from iSLAT.Modules.FileHandling import hitran_data_folder_path, absolute_data_files_path
        
        if os.path.isabs(filename):
            source_filepath = filename
        else:
            # Try relative to data files path
            source_filepath = os.path.join(absolute_data_files_path, filename)
            if not os.path.exists(source_filepath):
                source_filepath = filename
        
        cache_filepath = self._get_cache_path(source_filepath)
        
        section.mark("check_cache")
        # Try to load from cache first
        if self._is_cache_valid(source_filepath, cache_filepath):
            section.mark("load_from_cache")
            if self._load_from_cache(cache_filepath):
                print(f"[CACHE HIT] Loaded {self.molecule_id} from binary cache")
                section.mark("cache_load_complete")
                section.end()
                section.get_breakdown(print_output=True)
                return
        
        # Cache miss or invalid - parse the original file
        section.mark("read_molecular_data")
        print(f"[CACHE MISS] Parsing {self.molecule_id} from source file...")
        
        from iSLAT.Modules.FileHandling.molecular_data_reader import read_molecular_data
        
        partition_function, lines_data, other_fields = read_molecular_data(self.molecule_id, filename)
        section.mark("parse_complete")
        
        self.partition_function = partition_function
        
        self._molar_mass = other_fields[0][1]
        print(f'Molar_mass: {self._molar_mass}')

        # Convert list of dicts to structured numpy array for fast column access
        section.mark("convert_to_structured_array")
        if lines_data:
            n_lines = len(lines_data)
            self._raw_lines_data = np.empty(n_lines, dtype=_LINE_DTYPE)
            
            # Fill structured array from list of dicts
            for i, d in enumerate(lines_data):
                self._raw_lines_data[i] = (
                    d.get('nr', 0),
                    str(d.get('lev_up', '')),
                    str(d.get('lev_low', '')),
                    d.get('lam', 0.0),
                    d.get('freq', 0.0),
                    d.get('a_stein', 0.0),
                    d.get('e_up', 0.0),
                    d.get('e_low', 0.0),
                    d.get('g_up', 0),
                    d.get('g_low', 0)
                )
        else:
            self._raw_lines_data = np.empty(0, dtype=_LINE_DTYPE)
        
        self.lines = None  # Will be created on demand
        self._data_loaded = True
        self._invalidate_caches()
        
        # Save to cache for next time
        section.mark("save_to_cache")
        if self._save_to_cache(cache_filepath):
            print(f"[CACHE SAVED] {self.molecule_id} cached for faster loading")
        
        section.end()
        section.get_breakdown(print_output=True)
        
    def _ensure_lines_created(self):
        """Ensure MoleculeLine objects are created from raw data."""
        if self.lines is None and self._raw_lines_data is not None and len(self._raw_lines_data) > 0:
            from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine
            # Create MoleculeLine objects from structured array
            self.lines = [
                MoleculeLine(self.molecule_id, {
                    'nr': int(row['nr']),
                    'lev_up': str(row['lev_up']),
                    'lev_low': str(row['lev_low']),
                    'lam': float(row['lam']),
                    'freq': float(row['freq']),
                    'a_stein': float(row['a_stein']),
                    'e_up': float(row['e_up']),
                    'e_low': float(row['e_low']),
                    'g_up': int(row['g_up']),
                    'g_low': int(row['g_low'])
                })
                for row in self._raw_lines_data
            ]

    def _invalidate_caches(self):
        """Invalidate all cached data."""
        self._lines_cache_valid = False
        self._lines_cache = None
        self._wavelengths_cache = None
        self._frequencies_cache = None
        self._a_stein_cache = None
        self._e_up_cache = None
        self._e_low_cache = None
        self._g_up_cache = None
        self._g_low_cache = None
        self._pandas_df_cache = None  # Invalidate DataFrame cache too

    def get_ndarray(self) -> np.ndarray:
        """
        Convert the line data to a numpy ndarray.

        Returns
        -------
        np.ndarray
            Numpy array containing the line data.
        """
        self._ensure_data_loaded()
        self._ensure_lines_created()
        if not self.lines:
            return np.array([])
        return np.array([line.get_ndarray() for line in self.lines])
    
    def get_pandas_table(self) -> "pandas.DataFrame":
        """
        Get all lines as a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing all line data
        """
        pd = _get_pandas()
        if pd is None:
            raise ImportError("Pandas required to create table")
            
        self._ensure_data_loaded()
        
        # Create DataFrame directly from structured numpy array if available
        if self._raw_lines_data is not None and len(self._raw_lines_data) > 0:
            return pd.DataFrame(self._raw_lines_data)
        
        self._ensure_lines_created()
        if not self.lines:
            return pd.DataFrame()
        
        # Combine all individual line DataFrames efficiently
        line_dfs = [line.get_pandas_table() for line in self.lines]
        return pd.concat(line_dfs, ignore_index=True)
    
    def get_partition_table(self):
        """
        Get partition function as a pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            DataFrame containing partition function data
        """
        pd = _get_pandas()
        if pd is None:
            raise ImportError("Pandas required to create table")
            
        self._ensure_data_loaded()
        if self.partition_function is None:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'Temperature': self.partition_function.t,
            'Partition_Function': self.partition_function.q
        })
    
    @property
    def partition(self):
        """Access to partition function"""
        self._ensure_data_loaded()
        return self.partition_function
    
    @property
    def num_lines(self):
        """Number of lines in the list"""
        self._ensure_data_loaded()
        # Use structured array if available for faster count
        if self._raw_lines_data is not None:
            return len(self._raw_lines_data)
        elif self.lines is not None:
            return len(self.lines)
        return 0
    
    # Compatibility properties for legacy code expecting MolData format
    @property 
    def lines_as_namedtuple(self):
        """Get all line data as a single namedtuple structure for compatibility"""
        self._ensure_data_loaded()
        
        # Use cache if valid
        if self._lines_cache_valid and self._lines_cache is not None:
            return self._lines_cache
        
        # Use structured array for fast column access (O(1) slicing)
        if self._raw_lines_data is not None and len(self._raw_lines_data) > 0:
            # Direct column access from structured array - no list comprehension needed
            self._lines_cache = self._lines_type(
                self._raw_lines_data['nr'],
                self._raw_lines_data['lev_up'],
                self._raw_lines_data['lev_low'],
                self._raw_lines_data['lam'],
                self._raw_lines_data['freq'],
                self._raw_lines_data['a_stein'],
                self._raw_lines_data['e_up'],
                self._raw_lines_data['e_low'],
                self._raw_lines_data['g_up'],
                self._raw_lines_data['g_low']
            )
        else:
            # Fallback to using MoleculeLine objects
            self._ensure_lines_created()
            if not self.lines:
                self._lines_cache = self._lines_type([], [], [], [], [], [], [], [], [], [])
            else:
                line_data = list(zip(*[(line.nr, line.lev_up, line.lev_low, line.lam, line.freq,
                                       line.a_stein, line.e_up, line.e_low, line.g_up, line.g_low) 
                                      for line in self.lines]))
                
                self._lines_cache = self._lines_type(*[np.array(data) for data in line_data])
        
        self._lines_cache_valid = True
        return self._lines_cache
    
    @property
    def name(self):
        """Molecule name for compatibility"""
        return self.molecule_id
    
    def get_wavelengths(self):
        """
        Get wavelengths of all lines as a numpy array.
        
        Returns
        -------
        np.ndarray
            Array of wavelengths in microns
        """
        self._ensure_data_loaded()
        
        # Use cached wavelengths if available
        if self._wavelengths_cache is not None:
            return self._wavelengths_cache
        
        # Use direct structured array column access (O(1) slice)
        if self._raw_lines_data is not None and len(self._raw_lines_data) > 0:
            self._wavelengths_cache = self._raw_lines_data['lam'].copy()
        else:
            self._ensure_lines_created()
            if not self.lines:
                self._wavelengths_cache = np.array([])
            else:
                self._wavelengths_cache = np.array([line.lam for line in self.lines])
                
        return self._wavelengths_cache
    
    def get_frequencies(self):
        """
        Get frequencies of all lines as a numpy array.
        
        Returns
        -------
        np.ndarray
            Array of frequencies in Hz
        """
        self._ensure_data_loaded()
        
        # Use cached frequencies if available
        if self._frequencies_cache is not None:
            return self._frequencies_cache
        
        # Use direct structured array column access (O(1) slice)
        if self._raw_lines_data is not None and len(self._raw_lines_data) > 0:
            self._frequencies_cache = self._raw_lines_data['freq'].copy()
        else:
            self._ensure_lines_created()
            if not self.lines:
                self._frequencies_cache = np.array([])
            else:
                self._frequencies_cache = np.array([line.freq for line in self.lines])
                
        return self._frequencies_cache
    
    def get_einstein_coefficients(self):
        """
        Get all Einstein A coefficients from the lines.
        
        Returns
        -------
        np.ndarray
            Array of Einstein A coefficients
        """
        self._ensure_data_loaded()
        
        # Use cached values if available
        if self._a_stein_cache is not None:
            return self._a_stein_cache
        
        # Use direct structured array column access (O(1) slice)
        if self._raw_lines_data is not None and len(self._raw_lines_data) > 0:
            self._a_stein_cache = self._raw_lines_data['a_stein'].copy()
        else:
            self._ensure_lines_created()
            if not self.lines:
                self._a_stein_cache = np.array([])
            else:
                self._a_stein_cache = np.array([line.a_stein for line in self.lines])
                
        return self._a_stein_cache
    
    def get_upper_energies(self):
        """
        Get all upper level energies from the lines.
        
        Returns
        -------
        np.ndarray
            Array of upper level energies in K
        """
        self._ensure_data_loaded()
        
        # Use cached values if available
        if self._e_up_cache is not None:
            return self._e_up_cache
        
        # Use direct structured array column access (O(1) slice)
        if self._raw_lines_data is not None and len(self._raw_lines_data) > 0:
            self._e_up_cache = self._raw_lines_data['e_up'].copy()
        else:
            self._ensure_lines_created()
            if not self.lines:
                self._e_up_cache = np.array([])
            else:
                self._e_up_cache = np.array([line.e_up for line in self.lines])
                
        return self._e_up_cache
    
    def get_lower_energies(self):
        """
        Get all lower level energies from the lines.
        
        Returns
        -------
        np.ndarray
            Array of lower level energies in K
        """
        self._ensure_data_loaded()
        
        # Use cached values if available
        if self._e_low_cache is not None:
            return self._e_low_cache
        
        # Use direct structured array column access (O(1) slice)
        if self._raw_lines_data is not None and len(self._raw_lines_data) > 0:
            self._e_low_cache = self._raw_lines_data['e_low'].copy()
        else:
            self._ensure_lines_created()
            if not self.lines:
                self._e_low_cache = np.array([])
            else:
                self._e_low_cache = np.array([line.e_low for line in self.lines])
                
        return self._e_low_cache
    
    def get_upper_weights(self):
        """
        Get all upper level statistical weights from the lines.
        
        Returns
        -------
        np.ndarray
            Array of upper level statistical weights
        """
        self._ensure_data_loaded()
        
        # Use cached values if available
        if self._g_up_cache is not None:
            return self._g_up_cache
        
        # Use direct structured array column access (O(1) slice)
        if self._raw_lines_data is not None and len(self._raw_lines_data) > 0:
            self._g_up_cache = self._raw_lines_data['g_up'].copy()
        else:
            self._ensure_lines_created()
            if not self.lines:
                self._g_up_cache = np.array([])
            else:
                self._g_up_cache = np.array([line.g_up for line in self.lines])
                
        return self._g_up_cache
    
    def get_lower_weights(self):
        """
        Get all lower level statistical weights from the lines.
        
        Returns
        -------
        np.ndarray
            Array of lower level statistical weights
        """
        self._ensure_data_loaded()
        
        # Use cached values if available
        if self._g_low_cache is not None:
            return self._g_low_cache
        
        # Use direct structured array column access (O(1) slice)
        if self._raw_lines_data is not None and len(self._raw_lines_data) > 0:
            self._g_low_cache = self._raw_lines_data['g_low'].copy()
        else:
            self._ensure_lines_created()
            if not self.lines:
                self._g_low_cache = np.array([])
            else:
                self._g_low_cache = np.array([line.g_low for line in self.lines])
                
        return self._g_low_cache
    
    def get_lines_in_range(self, lam_min: float, lam_max: float):
        """
        Get lines within a wavelength range as MoleculeLine objects.
        
        Parameters
        ----------
        lam_min : float
            Minimum wavelength in microns
        lam_max : float
            Maximum wavelength in microns
            
        Returns
        -------
        list
            List of MoleculeLine objects within the range
        """
        self._ensure_data_loaded()
        
        # Use structured array boolean mask for efficient filtering
        if self._raw_lines_data is not None and len(self._raw_lines_data) > 0:
            # Boolean mask indexing is O(n) but very fast with numpy
            lam_values = self._raw_lines_data['lam']
            mask = (lam_values >= lam_min) & (lam_values <= lam_max)
            filtered_raw = self._raw_lines_data[mask]
            
            # Create MoleculeLine objects from filtered structured array
            from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine
            return [
                MoleculeLine(self.molecule_id, {
                    'nr': int(row['nr']),
                    'lev_up': str(row['lev_up']),
                    'lev_low': str(row['lev_low']),
                    'lam': float(row['lam']),
                    'freq': float(row['freq']),
                    'a_stein': float(row['a_stein']),
                    'e_up': float(row['e_up']),
                    'e_low': float(row['e_low']),
                    'g_up': int(row['g_up']),
                    'g_low': int(row['g_low'])
                })
                for row in filtered_raw
            ]
        else:
            # Fallback to using existing lines
            self._ensure_lines_created()
            lines_in_range = []
            for line in self.lines:
                if lam_min <= line.lam <= lam_max:
                    lines_in_range.append(line)
            return lines_in_range
    
    def get_ndarray_of_attribute(self, attribute_name):
        """
        Get a numpy array of a specific attribute for all lines.
        
        Parameters
        ----------
        attribute_name : str
            Name of the attribute to extract from each line
            
        Returns
        -------
        np.ndarray
            Array of attribute values
        """
        self._ensure_data_loaded()
        
        # Use structured array column access if the attribute is a field
        if self._raw_lines_data is not None and len(self._raw_lines_data) > 0:
            if attribute_name in self._raw_lines_data.dtype.names:
                return self._raw_lines_data[attribute_name].copy()
        
        # Fallback to line objects
        self._ensure_lines_created()
        return np.array([getattr(line, attribute_name) for line in self.lines])

    @property
    def fname(self):
        """File name for compatibility with old MolData interface"""
        return getattr(self, '_filename', None)
    
    @fname.setter
    def fname(self, value):
        """Set file name"""
        self._filename = value

    def enable_parallel_line_loading(self, use_parallel=True):
        """
        Enable or disable parallel line loading for very large molecules.
        This can significantly speed up loading of molecules with >100k lines.
        """
        self._use_parallel_loading = use_parallel
        
    def _load_from_file_parallel(self, filename: str, chunk_size: int = 50000):
        """
        Load molecular data from a .par file using parallel processing for large files.
        
        Parameters
        ----------
        filename : str
            Path to the .par file
        chunk_size : int
            Number of lines to process per chunk
        """
        from iSLAT.Modules.FileHandling.molecular_data_reader import read_molecular_data
        import multiprocessing as mp
        import numpy as np
        
        # First, read the file to determine size
        with open(filename, "r") as f:
            data_raw = f.readlines()
        
        # Count clean lines to estimate size
        data_clean = [line for line in data_raw if line.strip() and not line.strip().startswith("#")]
        
        if len(data_clean) < 3:
            return self._load_from_file(filename)  # Fallback to regular loading
            
        n_partition = int(data_clean[0])
        num_lines = len(data_clean) - n_partition - 2  # Subtract partition function and headers
        
        # Use parallel loading only for large files
        if num_lines < 100000 or not hasattr(self, '_use_parallel_loading') or not self._use_parallel_loading:
            return self._load_from_file(filename)  # Use regular loading
            
        print(f"Using parallel loading for {num_lines} lines...")
        
        # Use regular loading for now - parallel loading can be complex with file I/O
        # This is a placeholder for future implementation
        return self._load_from_file(filename)
    
    def _get_pandas_dataframe(self):
        """Get or create cached pandas DataFrame from structured array."""
        # Fast path: return cached DataFrame if available
        if self._pandas_df_cache is not None:
            return self._pandas_df_cache
        
        # Check if we have raw data to convert
        if self._raw_lines_data is None or len(self._raw_lines_data) == 0:
            return None
        
        pd = _get_pandas()
        if pd is None:
            return None
        
        # Create DataFrame directly from structured numpy array - very efficient
        self._pandas_df_cache = pd.DataFrame(self._raw_lines_data)
        
        return self._pandas_df_cache
    
    @property
    def molar_mass(self):
        """Get the molar mass of the molecule."""
        self._ensure_data_loaded()
        return self._molar_mass