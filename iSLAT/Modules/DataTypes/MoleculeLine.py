from typing import Dict, Any, Optional, Union
import numpy as np

# Lazy import for pandas
pd = None

def _get_pandas():
    """Lazy import of pandas"""
    global pd
    if pd is None:
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Pandas is required for table functionality")
    return pd

class MoleculeLine:
    """
    Efficient representation of a single molecular line.
    
    Uses __slots__ for memory efficiency and direct attribute access for speed.
    All attributes are typed for better performance and code clarity.
    """
    __slots__ = ('molecule_id', 'nr', 'lev_up', 'lev_low', 'lam', 'freq', 
                 'a_stein', 'e_up', 'e_low', 'g_up', 'g_low')
    
    def __init__(self, molecule_id: str, line_data: Dict[str, Any], **kwargs):
        """
        Initialize a MoleculeLine object.
        
        Parameters
        ----------
        molecule_id : str
            Identifier for the molecule.
        line_data : dict
            Dictionary containing line data with keys like 'frequency', 'wavelength', 'intensity', etc.
        """
        self.molecule_id: str = molecule_id
        
        # Direct attribute assignment for better performance
        # Using .get() with proper type conversion for robustness
        self.nr: Optional[Union[int, float]] = line_data.get('nr', None)
        self.lev_up: Optional[Union[int, float]] = line_data.get('lev_up', None)
        self.lev_low: Optional[Union[int, float]] = line_data.get('lev_low', None)
        self.lam: Optional[float] = line_data.get('lam', None)
        self.freq: Optional[float] = line_data.get('freq', None)
        self.a_stein: Optional[float] = line_data.get('a_stein', None)
        self.e_up: Optional[float] = line_data.get('e_up', None)
        self.e_low: Optional[float] = line_data.get('e_low', None)
        self.g_up: Optional[Union[int, float]] = line_data.get('g_up', None)
        self.g_low: Optional[Union[int, float]] = line_data.get('g_low', None)

    def get_ndarray(self) -> np.ndarray:
        """
        Convert the line data to a numpy ndarray.

        Returns
        -------
        np.ndarray
            Numpy array containing the line data.
        """
        return np.array([self.nr, self.lev_up, self.lev_low, self.lam, self.freq,
                         self.a_stein, self.e_up, self.e_low, self.g_up, self.g_low])

    def get_pandas_table(self):
        """
        Convert the line data to a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the line data.
        """
        pd = _get_pandas()
        return pd.DataFrame({
            'nr': [self.nr],
            'lev_up': [self.lev_up],
            'lev_low': [self.lev_low],
            'lam': [self.lam],
            'freq': [self.freq],
            'a_stein': [self.a_stein],
            'e_up': [self.e_up],
            'e_low': [self.e_low],
            'g_up': [self.g_up],
            'g_low': [self.g_low]
        })
    
    def get_dict(self) -> Dict[str, Any]:
        """
        Convert the line data to a dictionary.

        Returns
        -------
        dict
            Dictionary containing the line data.
        """
        return {
            'nr': self.nr,
            'lev_up': self.lev_up,
            'lev_low': self.lev_low,
            'lam': self.lam,
            'freq': self.freq,
            'a_stein': self.a_stein,
            'e_up': self.e_up,
            'e_low': self.e_low,
            'g_up': self.g_up,
            'g_low': self.g_low
        }
    
    @property
    def line_data(self) -> 'LineDataView':
        """
        Compatibility property that returns a namedtuple-like object for legacy code.
        
        Returns
        -------
        LineDataView
            Object with namedtuple-like attribute access
        """
        return LineDataView(self)
    
    def __str__(self) -> str:
        return f"MoleculeLine(molecule={self.molecule_id}, lam={self.lam}, freq={self.freq})"

    def __repr__(self) -> str:
        return self.__str__()

class LineDataView:
    """
    A lightweight view object that provides namedtuple-like access to MoleculeLine data.
    Used for backward compatibility without the overhead of creating actual namedtuples.
    
    Uses __slots__ for memory efficiency.
    """
    __slots__ = ('_line',)
    
    def __init__(self, line: MoleculeLine):
        self._line: MoleculeLine = line
    
    @property
    def nr(self):
        return self._line.nr
    
    @property
    def lev_up(self):
        return self._line.lev_up
    
    @property
    def lev_low(self):
        return self._line.lev_low
    
    @property
    def lam(self):
        return self._line.lam
    
    @property
    def freq(self):
        return self._line.freq
    
    @property
    def a_stein(self):
        return self._line.a_stein
    
    @property
    def e_up(self):
        return self._line.e_up
    
    @property
    def e_low(self):
        return self._line.e_low
    
    @property
    def g_up(self):
        return self._line.g_up
    
    @property
    def g_low(self):
        return self._line.g_low