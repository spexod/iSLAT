import numpy as np
from collections import namedtuple
from typing import Optional, List, Union, Any, NamedTuple

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

class LineTuple(NamedTuple):
    """Named tuple for line data"""
    nr: np.ndarray[int]
    '''Line number'''
    lev_up: np.ndarray[int]
    '''Upper energy level'''
    lev_low: np.ndarray[int]
    '''Lower energy level'''
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

class MoleculeLineList:
    """
    Efficient molecular line list with lazy loading and caching.
    """
    __slots__ = ('molecule_id', 'lines', 'partition_function', '_partition_type', 
                 '_lines_type', '_lines_cache', '_lines_cache_valid', '_wavelengths_cache',
                 '_frequencies_cache', '_data_loaded', '_filename', '_raw_lines_data',
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
        # Store raw data for lazy MoleculeLine creation
        self._raw_lines_data = lines_data
        self.lines = None  # Will be created on demand
        self._data_loaded = True
        self._invalidate_caches()

    def _load_from_file(self, filename: str):
        """
        Load molecular data from a .par file using the FileHandling module.
        
        Parameters
        ----------
        filename : str
            Path to the .par file
        """
        from iSLAT.Modules.FileHandling.molecular_data_reader import read_molecular_data
        
        partition_function, lines_data, other_fields = read_molecular_data(self.molecule_id, filename)
        self.partition_function = partition_function
        
        self._molar_mass = other_fields[0][1]
        print(f'Molar_mass: {self._molar_mass}')

        # Store raw data for lazy MoleculeLine creation
        self._raw_lines_data = lines_data
        self.lines = None  # Will be created on demand
        self._data_loaded = True
        self._invalidate_caches()
        
    def _ensure_lines_created(self):
        """Ensure MoleculeLine objects are created from raw data."""
        if self.lines is None and hasattr(self, '_raw_lines_data'):
            from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine
            # Create MoleculeLine objects only when needed
            self.lines = [MoleculeLine(self.molecule_id, line_data) for line_data in self._raw_lines_data]

    def _invalidate_caches(self):
        """Invalidate all cached data."""
        self._lines_cache_valid = False
        self._lines_cache = None
        self._wavelengths_cache = None
        self._frequencies_cache = None
        self._pandas_df_cache = None  # Invalidate DataFrame cache too

    def get_ndarray(self):
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
    
    def get_pandas_table(self):
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
        
        # Optimize for direct pandas creation from raw data if available
        if hasattr(self, '_raw_lines_data') and self._raw_lines_data:
            # Create DataFrame directly from raw data - much faster
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
        # Use raw data if available for faster count
        if hasattr(self, '_raw_lines_data') and self._raw_lines_data is not None:
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
        
        # Try to create from raw data first - use cached DataFrame
        if hasattr(self, '_raw_lines_data') and self._raw_lines_data:
            # Use cached pandas DataFrame for extraction if available
            df = self._get_pandas_dataframe()
            if df is not None:
                self._lines_cache = self._lines_type(
                    df['nr'].values,
                    df['lev_up'].values,
                    df['lev_low'].values, 
                    df['lam'].values,
                    df['freq'].values,
                    df['a_stein'].values,
                    df['e_up'].values,
                    df['e_low'].values,
                    df['g_up'].values,
                    df['g_low'].values
                )
            else:
                # Fallback to list comprehensions but with numpy arrays
                self._lines_cache = self._lines_type(
                    np.array([d['nr'] for d in self._raw_lines_data]),
                    np.array([d['lev_up'] for d in self._raw_lines_data]),
                    np.array([d['lev_low'] for d in self._raw_lines_data]),
                    np.array([d['lam'] for d in self._raw_lines_data]),
                    np.array([d['freq'] for d in self._raw_lines_data]),
                    np.array([d['a_stein'] for d in self._raw_lines_data]),
                    np.array([d['e_up'] for d in self._raw_lines_data]),
                    np.array([d['e_low'] for d in self._raw_lines_data]),
                    np.array([d['g_up'] for d in self._raw_lines_data]),
                    np.array([d['g_low'] for d in self._raw_lines_data])
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
        
        # get from cached DataFrame
        if hasattr(self, '_raw_lines_data') and self._raw_lines_data:
            df = self._get_pandas_dataframe()
            if df is not None:
                # Use pandas for vectorized extraction - much faster
                self._wavelengths_cache = df['lam'].values
            else:
                # Fallback to numpy array creation
                self._wavelengths_cache = np.array([d['lam'] for d in self._raw_lines_data])
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
        
        # get from cached DataFrame
        if hasattr(self, '_raw_lines_data') and self._raw_lines_data:
            df = self._get_pandas_dataframe()
            if df is not None:
                # Use pandas for vectorized extraction - much faster
                self._frequencies_cache = df['freq'].values
            else:
                # Fallback to numpy array creation
                self._frequencies_cache = np.array([d['freq'] for d in self._raw_lines_data])
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
        # Try to get from raw data first (faster)
        if hasattr(self, '_raw_lines_data') and self._raw_lines_data:
            return np.array([line_data['a_stein'] for line_data in self._raw_lines_data])
        else:
            self._ensure_lines_created()
            return np.array([line.a_stein for line in self.lines])
    
    def get_upper_energies(self):
        """
        Get all upper level energies from the lines.
        
        Returns
        -------
        np.ndarray
            Array of upper level energies in K
        """
        self._ensure_data_loaded()
        # Try to get from raw data first (faster)
        if hasattr(self, '_raw_lines_data') and self._raw_lines_data:
            return np.array([line_data['e_up'] for line_data in self._raw_lines_data])
        else:
            self._ensure_lines_created()
            return np.array([line.e_up for line in self.lines])
    
    def get_lower_energies(self):
        """
        Get all lower level energies from the lines.
        
        Returns
        -------
        np.ndarray
            Array of lower level energies in K
        """
        self._ensure_data_loaded()
        # Try to get from raw data first (faster)
        if hasattr(self, '_raw_lines_data') and self._raw_lines_data:
            return np.array([line_data['e_low'] for line_data in self._raw_lines_data])
        else:
            self._ensure_lines_created()
            return np.array([line.e_low for line in self.lines])
    
    def get_upper_weights(self):
        """
        Get all upper level statistical weights from the lines.
        
        Returns
        -------
        np.ndarray
            Array of upper level statistical weights
        """
        self._ensure_data_loaded()
        # Try to get from raw data first (faster)
        if hasattr(self, '_raw_lines_data') and self._raw_lines_data:
            return np.array([line_data['g_up'] for line_data in self._raw_lines_data])
        else:
            self._ensure_lines_created()
            return np.array([line.g_up for line in self.lines])
    
    def get_lower_weights(self):
        """
        Get all lower level statistical weights from the lines.
        
        Returns
        -------
        np.ndarray
            Array of lower level statistical weights
        """
        self._ensure_data_loaded()
        # Try to get from raw data first (faster)
        if hasattr(self, '_raw_lines_data') and self._raw_lines_data:
            return np.array([line_data['g_low'] for line_data in self._raw_lines_data])
        else:
            self._ensure_lines_created()
            return np.array([line.g_low for line in self.lines])
    
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
        
        # For efficiency, filter raw data first if available
        if hasattr(self, '_raw_lines_data') and self._raw_lines_data:
            # Filter raw data first
            filtered_raw = [line_data for line_data in self._raw_lines_data 
                           if lam_min <= line_data['lam'] <= lam_max]
            # Create MoleculeLine objects only for filtered data
            from iSLAT.Modules.DataTypes.MoleculeLine import MoleculeLine
            return [MoleculeLine(self.molecule_id, line_data) for line_data in filtered_raw]
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
        
        # Try to get from raw data first if possible
        if hasattr(self, '_raw_lines_data') and self._raw_lines_data and attribute_name in self._raw_lines_data[0]:
            return np.array([line_data[attribute_name] for line_data in self._raw_lines_data])
        else:
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
        """Get or create cached pandas DataFrame from raw data."""
        if not hasattr(self, '_pandas_df_cache') or self._pandas_df_cache is None:
            if hasattr(self, '_raw_lines_data') and self._raw_lines_data:
                pd = _get_pandas()
                if pd is not None:
                    self._pandas_df_cache = pd.DataFrame(self._raw_lines_data)
                else:
                    self._pandas_df_cache = None
            else:
                self._pandas_df_cache = None
        return self._pandas_df_cache