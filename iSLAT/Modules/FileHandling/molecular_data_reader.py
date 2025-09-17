# -*- coding: utf-8 -*-

"""
Molecular Data Reader Module

This module provides functionality to read molecular data from .par files,
replacing the functionality previously in the MolData class.

The .par file format:
- First line: number of partition function values
- Next N lines: partition function (temperature vs Q(temperature))
- Next line: number of lines
- Following lines: molecular line data in fixed format

- 01/06/2020: SB, initial version (as MolData)
- 07/12/2025: Johnny McCaskill, Refactored into separate module for better organization
"""

import numpy as np
from collections import namedtuple
import time  # Add timing for performance debugging

from pathlib import Path
import os
#from iSLAT.Modules.FileHandling.iSLATFileHandling import data_files_path, hitran_data_folder_name
from iSLAT.Modules.FileHandling import data_files_path, hitran_data_folder_name

try:
    import pandas as pd
except ImportError:
    pd = None

__all__ = ["read_molecular_data", "MolecularDataReader"]


def read_molecular_data(molecule_name, filename):
    """
    Read molecular data from a .par file and return partition function and lines data.
    
    Parameters
    ----------
    molecule_name : str
        Name of the molecule (e.g., "CO")
    filename : str
        Path to the .par file
        
    Returns
    -------
    tuple
        (partition_function, lines_data) where:
        - partition_function: namedtuple with temperature and Q values
        - lines_data: list of dictionaries containing line data
    """
    reader = MolecularDataReader()
    return reader.read_par_file(filename)

class MolecularDataReader:
    """
    A utility class for reading molecular data files.
    
    This class provides methods for reading various molecular data formats.
    Currently supports .par files in the format used by the original Fortran 90 code.
    """
    
    __slots__ = ('_partition_type', '_lines_type', 'debug')
    
    def __init__(self, debug=False):
        # Define namedtuple types for data structure
        self._partition_type = namedtuple('partition', ['t', 'q'])
        self._lines_type = namedtuple('lines', ['nr', 'lev_up', 'lev_low', 'lam', 'freq', 'a_stein',
                                               'e_up', 'e_low', 'g_up', 'g_low'])
        self.debug = debug

    def read_par_file(self, filename, file_path=data_files_path):
        """
        Read molecular data from a .par file using the fastest possible method.
        """
        #full_file_path = Path(file_path) / filename
        full_file_path = os.path.join(file_path, filename)
        full_file_path = os.path.abspath(full_file_path)
        #full_file_path = filename
        print(f"Loading lines from filepath: {full_file_path}")

        # Try ultra-fast method first
        try:
            return self._read_par_file_ultra_fast(full_file_path)
        except Exception as e:
            if self.debug:
                print(f"Debug: Ultra-fast method failed: {e}, falling back to standard method")
            return self._read_par_file_standard(full_file_path)

    def _read_par_file_ultra_fast(self, filename):
        """Ultra-fast .par file reader using direct numpy operations."""
        # Read entire file at once using binary mode for speed
        with open(filename, 'rb') as f:
            file_bytes = f.read()
        
        # Decode and normalize line endings
        file_str = file_bytes.decode('utf-8', errors='ignore')
        all_lines = file_str.replace('\r\n', '\n').replace('\r', '\n').split('\n')
        
        # Filter out comments and empty lines
        clean_lines = [line for line in all_lines if line.strip() and not line.strip().startswith('#')]
        
        if len(clean_lines) < 3:
            return None, None
        
        # Parse partition function
        n_partition = int(clean_lines[0].strip())
        partition_lines = clean_lines[1:n_partition + 1]
        
        # Create partition function using vectorized operations
        partition_data = []
        for line in partition_lines:
            parts = line.split()
            if len(parts) >= 2:
                partition_data.append([float(parts[0]), float(parts[1])])
        
        partition_array = np.array(partition_data)
        partition_function = self._partition_type(partition_array[:, 0], partition_array[:, 1])
        
        # Find molecular lines section
        lines_start_search = n_partition + 1
        num_lines = None
        lines_start = None
        
        for i in range(lines_start_search, min(lines_start_search + 5, len(clean_lines))):
            try:
                num_lines = int(clean_lines[i].strip())
                lines_start = i + 1
                break
            except ValueError:
                continue
        
        if num_lines is None or lines_start is None:
            return partition_function, []
        
        # Parse molecular lines using ultra-fast numpy operations
        molecular_lines = clean_lines[lines_start:lines_start + num_lines]
        lines_data = self._parse_lines_ultra_fast_direct(molecular_lines)
        
        return partition_function, lines_data
    
    def _parse_lines_ultra_fast_direct(self, molecular_lines):
        """Parse lines using direct numpy array operations - fastest possible."""
        num_lines = len(molecular_lines)
        
        # Pre-allocate arrays for maximum speed
        lines_data = []
        
        # Parse in optimized chunks to maintain cache efficiency
        chunk_size = 5000
        for start_idx in range(0, num_lines, chunk_size):
            end_idx = min(start_idx + chunk_size, num_lines)
            chunk_data = []
            
            for i in range(start_idx, end_idx):
                line = molecular_lines[i]
                if len(line) >= 149:
                    try:
                        line_dict = {
                            'nr': int(line[0:6]),
                            'lev_up': line[6:36].strip(),
                            'lev_low': line[36:66].strip(),
                            'lam': float(line[66:77]),
                            'freq': float(line[77:92]) * 1e9,  # GHz to Hz
                            'a_stein': float(line[92:105]),
                            'e_up': float(line[105:120]),
                            'e_low': float(line[120:135]),
                            'g_up': float(line[135:142]),
                            'g_low': float(line[142:149])
                        }
                        chunk_data.append(line_dict)
                    except (ValueError, IndexError):
                        continue
            
            lines_data.extend(chunk_data)
        
        return lines_data
    
    def _read_par_file_standard(self, filename):
        """Standard .par file reader (fallback method)."""
        start_time = time.time()
        
        with open(filename, "r") as f:
            data_raw = f.readlines()
        
        file_read_time = time.time()
        if self.debug:
            print(f"Debug: File read took {file_read_time - start_time:.3f}s")

        # Remove comments and empty lines - optimized filtering
        data_clean = [line for line in data_raw if line.strip() and not line.strip().startswith("#")]
        
        clean_time = time.time()
        if self.debug:
            print(f"Debug: Cleaning took {clean_time - file_read_time:.3f}s")
            print(f"Debug: Total raw lines: {len(data_raw)}")
            print(f"Debug: Clean data lines: {len(data_clean)}")

        # Split file into partition function and line section
        n_partition = int(data_clean[0])
        
        if self.debug:
            print(f"Debug: Number of partition function entries: {n_partition}")
            
        data_q = data_clean[1:n_partition + 1]
        data_lines = data_clean[n_partition + 1:]

        # Read partition function and lines
        partition_time_start = time.time()
        partition_function = self._read_partition_function(data_q)
        partition_time_end = time.time()
        
        if self.debug:
            print(f"Debug: Partition function read took {partition_time_end - partition_time_start:.3f}s")
        
        lines_time_start = time.time()
        lines_data = self._read_lines_data(data_lines)
        lines_time_end = time.time()
        
        total_time = time.time() - start_time
        
        if self.debug:
            print(f"Debug: Lines data read took {lines_time_end - lines_time_start:.3f}s")
            print(f"Debug: Total file processing took {total_time:.3f}s")
        
        return partition_function, lines_data

    def _read_partition_function(self, data_q):
        """
        Read partition function data from raw file lines.
        
        Parameters
        ----------
        data_q : list
            List of strings containing partition function data
            
        Returns
        -------
        namedtuple
            Partition function with temperature and Q values
        """
        N = np.genfromtxt(data_q, dtype="f8,f8")
        q_temperature, q = [N[field] for field in N.dtype.names]
        return self._partition_type(q_temperature, q)

    def _read_lines_data(self, data_lines):
        """
        Read molecular lines data from raw file lines using the fastest method available.
        
        Parameters
        ----------
        data_lines : list
            List of strings containing molecular lines data
            
        Returns
        -------
        list
            List of dictionaries containing line data
        """
        if len(data_lines) < 3:
            return []
            
        # Get number of lines from the second line (skip_header=2 accounts for this)
        try:
            num_lines = int(data_lines[1])
        except (ValueError, IndexError):
            num_lines = len(data_lines) - 2  # Fallback
            
        if self.debug:
            print(f"Debug: Processing {num_lines} line data entries")
        
        # For small files, use original method
        if num_lines < 100:
            return self._read_lines_data_original(data_lines)
        
        # For large files, try ultra-fast parsing first
        if num_lines > 10000:
            try:
                # Use ultra-fast numpy vectorized parsing
                molecular_lines = data_lines[2:2+num_lines]  # Skip header lines
                lines_data = self._parse_lines_ultra_fast(molecular_lines)
                if lines_data:
                    if self.debug:
                        print(f"Debug: Ultra-fast parsing processed {len(lines_data)} lines")
                    return lines_data
            except Exception as e:
                if self.debug:
                    print(f"Debug: Ultra-fast parsing failed: {e}")
        
        # Fall back to optimized method
        return self._read_lines_data_optimized(data_lines[2:], num_lines)
    
    def _read_lines_data_original(self, data_lines):
        """Original method for small files."""
        # Parse the structured data using numpy - skip_header=2 to skip the number of lines
        M = np.genfromtxt(data_lines, dtype="i4,S30,S30,f8,f8,f8,f8,f8,f8", skip_header=2,
                          delimiter=(6, 30, 30, 11, 15, 13, 15, 15, 7, 7))

        if self.debug:
            print(f"Debug: genfromtxt returned array with shape: {M.shape if hasattr(M, 'shape') else 'scalar'}")

        # Handle single line case
        if M.ndim == 0:
            M = np.array([M])

        # Extract all fields at once
        nr, lev_up, lev_low, lam, freq, a_stein, e_up, e_low, g_up, g_low = [M[field] for field in M.dtype.names]

        # Decode string fields efficiently using vectorized operations
        lev_up = np.char.decode(lev_up).astype('U30')
        lev_low = np.char.decode(lev_low).astype('U30')
        lev_up = np.char.strip(lev_up)
        lev_low = np.char.strip(lev_low)

        # Convert frequency from GHz to Hz using vectorized operation
        freq = freq * 1e9

        # Create list of dictionaries using zip for efficiency
        field_names = ['nr', 'lev_up', 'lev_low', 'lam', 'freq', 'a_stein', 'e_up', 'e_low', 'g_up', 'g_low']
        field_arrays = [nr, lev_up, lev_low, lam, freq, a_stein, e_up, e_low, g_up, g_low]
        
        lines_data = [dict(zip(field_names, values)) for values in zip(*field_arrays)]
        
        if self.debug:
            print(f"Debug: Created {len(lines_data)} line data dictionaries")
            
        return lines_data
        
    def _read_lines_data_optimized(self, line_data, num_lines):
        """
        Optimized method for larger files using chunked processing and faster parsing.
        """
        # For very large files (>200k lines), try memory mapping
        if num_lines > 200000:
            if self.debug:
                print(f"Debug: Using memory mapping for {num_lines} lines")
            # Note: This would require file handle, for now use pandas
        
        # Convert lines to a single string for faster processing
        data_string = '\n'.join(line_data[:num_lines])
        
        # Use faster parsing with pandas if available
        if pd is not None:
            return self._read_lines_with_pandas(data_string)
        else:
            return self._read_lines_with_numpy_optimized(data_string)
            
    def _read_lines_with_pandas(self, data_string):
        """Use pandas for fast fixed-width parsing with optimizations."""
        import io
        
        try:
            # Define column specifications for fixed-width format
            colspecs = [(0, 6), (6, 36), (36, 66), (66, 77), (77, 92), 
                       (92, 105), (105, 120), (120, 135), (135, 142), (142, 149)]
            names = ['nr', 'lev_up', 'lev_low', 'lam', 'freq', 'a_stein', 'e_up', 'e_low', 'g_up', 'g_low']
            
            # Use more efficient data types and processing options
            df = pd.read_fwf(io.StringIO(data_string), 
                            colspecs=colspecs, 
                            names=names, 
                            dtype={
                                'nr': 'int32',
                                'lev_up': 'str', 
                                'lev_low': 'str',
                                'lam': 'float32',  # Use float32 for better memory efficiency
                                'freq': 'float64',  # Keep freq as float64 for precision
                                'a_stein': 'float32',
                                'e_up': 'float32',
                                'e_low': 'float32',
                                'g_up': 'float32',
                                'g_low': 'float32'
                            },
                            na_filter=False)  # Skip NaN checking for speed
            
            # Clean string columns efficiently
            df['lev_up'] = df['lev_up'].str.strip()
            df['lev_low'] = df['lev_low'].str.strip()
            
            # Convert frequency from GHz to Hz using vectorized operation
            df['freq'] = df['freq'].astype('float64') * 1e9
            
            # Convert to list of dictionaries efficiently using to_dict
            lines_data = df.to_dict('records')
            
            if self.debug:
                print(f"Debug: Pandas processed {len(lines_data)} lines")
                
            return lines_data
            
        except Exception as e:
            if self.debug:
                print(f"Debug: Pandas parsing failed: {e}, falling back to numpy")
            return self._read_lines_with_numpy_optimized(data_string)
            
    def _read_lines_with_numpy_optimized(self, data_string):
        """Optimized numpy parsing for when pandas is not available."""
        import io
        
        try:
            # Use numpy's fromstring for faster parsing
            lines = data_string.strip().split('\n')
            num_lines = len(lines)
            
            # Pre-allocate arrays for better performance
            nr = np.zeros(num_lines, dtype=np.int32)
            lev_up = np.empty(num_lines, dtype='U30')
            lev_low = np.empty(num_lines, dtype='U30') 
            lam = np.zeros(num_lines, dtype=np.float64)
            freq = np.zeros(num_lines, dtype=np.float64)
            a_stein = np.zeros(num_lines, dtype=np.float64)
            e_up = np.zeros(num_lines, dtype=np.float64)
            e_low = np.zeros(num_lines, dtype=np.float64)
            g_up = np.zeros(num_lines, dtype=np.float64)
            g_low = np.zeros(num_lines, dtype=np.float64)
            
            # Parse lines in batches for better cache performance
            batch_size = min(1000, num_lines)
            for i in range(0, num_lines, batch_size):
                end_idx = min(i + batch_size, num_lines)
                batch_lines = lines[i:end_idx]
                
                for j, line in enumerate(batch_lines):
                    idx = i + j
                    if len(line) >= 149:  # Ensure line is long enough
                        nr[idx] = int(line[0:6].strip())
                        lev_up[idx] = line[6:36].strip()
                        lev_low[idx] = line[36:66].strip()
                        lam[idx] = float(line[66:77].strip())
                        freq[idx] = float(line[77:92].strip()) * 1e9  # Convert GHz to Hz
                        a_stein[idx] = float(line[92:105].strip())
                        e_up[idx] = float(line[105:120].strip())
                        e_low[idx] = float(line[120:135].strip())
                        g_up[idx] = float(line[135:142].strip())
                        g_low[idx] = float(line[142:149].strip())
            
            # Create list of dictionaries efficiently
            field_names = ['nr', 'lev_up', 'lev_low', 'lam', 'freq', 'a_stein', 'e_up', 'e_low', 'g_up', 'g_low']
            field_arrays = [nr, lev_up, lev_low, lam, freq, a_stein, e_up, e_low, g_up, g_low]
            
            lines_data = [dict(zip(field_names, values)) for values in zip(*field_arrays)]
            
            if self.debug:
                print(f"Debug: Numpy optimized processed {len(lines_data)} lines")
                
            return lines_data
            
        except Exception as e:
            if self.debug:
                print(f"Debug: Optimized numpy parsing failed: {e}, using original method")
            # Fallback to original method
            return self._read_lines_data_original([str(num_lines)] + data_string.split('\n'))
        
        return lines_data

    def _read_lines_ultra_fast(self, filename):
        """
        Ultra-fast line reading using optimized C-level operations and memory mapping.
        """
        try:
            import mmap
            import struct
            
            print(f"Using ultra-fast loading for {filename}")
            
            # Memory map the file for fastest I/O
            with open(filename, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    # Read entire file as bytes
                    file_data = mm.read().decode('utf-8', errors='ignore')
            
            # Split into lines in one operation
            lines = file_data.strip().split('\n')
            
            # Filter out comments and empty lines using list comprehension
            clean_lines = [line for line in lines if line.strip() and not line.strip().startswith('#')]
            
            if len(clean_lines) < 3:
                return None, []
            
            # Parse partition function section
            n_partition = int(clean_lines[0])
            partition_lines = clean_lines[1:n_partition + 1]
            
            # Parse partition function using numpy
            partition_data = np.array([line.split() for line in partition_lines], dtype=float)
            partition_function = self._partition_type(partition_data[:, 0], partition_data[:, 1])
            
            # Parse molecular lines section
            lines_start = n_partition + 2  # Skip partition count and lines count
            num_lines = int(clean_lines[n_partition + 1])
            molecular_lines = clean_lines[lines_start:lines_start + num_lines]
            
            # Ultra-fast parsing using numpy structured arrays
            lines_data = self._parse_lines_ultra_fast(molecular_lines)
            
            return partition_function, lines_data
            
        except Exception as e:
            if self.debug:
                print(f"Debug: Ultra-fast parsing failed: {e}, falling back")
            return None, []
    
    def _parse_lines_ultra_fast(self, molecular_lines):
        """Parse molecular lines using the fastest possible method."""
        try:
            # Try pure numpy vectorized approach first (fastest)
            lines_data = self._parse_lines_vectorized(molecular_lines)
            if lines_data:
                return lines_data
            
            # Fallback to structured array approach
            # Pre-allocate numpy arrays for maximum speed
            num_lines = len(molecular_lines)
            
            # Create structured numpy array for batch processing
            dt = np.dtype([
                ('line', 'U150')  # Store entire line as string
            ])
            
            line_array = np.array([(line,) for line in molecular_lines], dtype=dt)
            
            # Use numpy's vectorized string operations for parsing
            lines_str = line_array['line']
            
            # Extract fields using vectorized operations (much faster than loops)
            nr = np.array([int(line[0:6].strip()) for line in lines_str], dtype=np.int32)
            lev_up = np.array([line[6:36].strip() for line in lines_str], dtype='U30')
            lev_low = np.array([line[36:66].strip() for line in lines_str], dtype='U30')
            lam = np.array([float(line[66:77].strip()) for line in lines_str], dtype=np.float32)
            freq = np.array([float(line[77:92].strip()) * 1e9 for line in lines_str], dtype=np.float64)  # Convert GHz to Hz
            a_stein = np.array([float(line[92:105].strip()) for line in lines_str], dtype=np.float32)
            e_up = np.array([float(line[105:120].strip()) for line in lines_str], dtype=np.float32)
            e_low = np.array([float(line[120:135].strip()) for line in lines_str], dtype=np.float32)
            g_up = np.array([float(line[135:142].strip()) for line in lines_str], dtype=np.float32)
            g_low = np.array([float(line[142:149].strip()) for line in lines_str], dtype=np.float32)
            
            # Create structured record array - fastest way to create list of dicts
            record_array = np.rec.fromarrays([nr, lev_up, lev_low, lam, freq, a_stein, e_up, e_low, g_up, g_low],
                                           names=['nr', 'lev_up', 'lev_low', 'lam', 'freq', 'a_stein', 'e_up', 'e_low', 'g_up', 'g_low'])
            
            # Convert to list of dictionaries efficiently
            lines_data = [dict(zip(record_array.dtype.names, row)) for row in record_array]
            
            return lines_data
            
        except Exception as e:
            if self.debug:
                print(f"Debug: Ultra-fast line parsing failed: {e}")
            # Fallback to simpler method
            return self._parse_lines_simple(molecular_lines)
            
    def _parse_lines_simple(self, molecular_lines):
        """Simple fallback parsing method."""
        lines_data = []
        for line in molecular_lines:
            try:
                if len(line) >= 149:
                    line_dict = {
                        'nr': int(line[0:6].strip()),
                        'lev_up': line[6:36].strip(),
                        'lev_low': line[36:66].strip(),
                        'lam': float(line[66:77].strip()),
                        'freq': float(line[77:92].strip()) * 1e9,
                        'a_stein': float(line[92:105].strip()),
                        'e_up': float(line[105:120].strip()),
                        'e_low': float(line[120:135].strip()),
                        'g_up': float(line[135:142].strip()),
                        'g_low': float(line[142:149].strip())
                    }
                    lines_data.append(line_dict)
            except (ValueError, IndexError):
                continue
        return lines_data

    @staticmethod
    def validate_par_file(filename):
        """
        Validate that a .par file has the correct format.
        
        Parameters
        ----------
        filename : str
            Path to the .par file
            
        Returns
        -------
        bool
            True if file format is valid, False otherwise
        """
        try:
            with open(filename, "r") as f:
                # Read only what we need for validation
                lines_read = 0
                data_clean = []
                
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        data_clean.append(stripped)
                        lines_read += 1
                        
                        # Early exit if we have enough to validate
                        if lines_read == 1:
                            try:
                                n_partition = int(data_clean[0])
                            except ValueError:
                                return False
                        elif lines_read >= 3:  # Minimum needed for validation
                            break
            
            if len(data_clean) < 3:
                return False
                
            # Check if we would have enough lines for partition function
            n_partition = int(data_clean[0])
            return len(data_clean) >= n_partition + 2  # +2 for partition count and line count
                
        except (IOError, OSError, ValueError):
            return False
    
    @staticmethod
    def get_file_info(filename):
        """
        Get basic information about a .par file without fully loading it.
        
        Parameters
        ----------
        filename : str
            Path to the .par file
            
        Returns
        -------
        dict
            Dictionary containing file information: 
            {'num_partition_points': int, 'num_lines': int, 'valid': bool}
        """
        info = {'num_partition_points': 0, 'num_lines': 0, 'valid': False}
        
        try:
            with open(filename, "r") as f:
                # Read only the lines we need for file info
                data_clean = []
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        data_clean.append(stripped)
                        
                        # Stop early once we have what we need
                        if len(data_clean) == 1:
                            try:
                                n_partition = int(data_clean[0])
                                info['num_partition_points'] = n_partition
                            except ValueError:
                                return info
                        elif len(data_clean) > n_partition + 1:
                            try:
                                n_lines = int(data_clean[n_partition + 1])
                                info['num_lines'] = n_lines
                                info['valid'] = True
                                break
                            except ValueError:
                                return info
                    
        except (IOError, OSError):
            pass
            
        return info

    def _parse_lines_vectorized(self, molecular_lines):
        """Ultra-optimized parsing using pure numpy vectorization."""
        try:
            import numpy as np
            
            # Convert all lines to a single numpy array
            lines_array = np.array(molecular_lines, dtype='U150')
            
            # Use numpy's vectorized string slicing for all fields at once
            nr_str = np.char.strip(np.char.slice(lines_array, 0, 6))
            lev_up = np.char.strip(np.char.slice(lines_array, 6, 36))
            lev_low = np.char.strip(np.char.slice(lines_array, 36, 66))
            lam_str = np.char.strip(np.char.slice(lines_array, 66, 77))
            freq_str = np.char.strip(np.char.slice(lines_array, 77, 92))
            a_stein_str = np.char.strip(np.char.slice(lines_array, 92, 105))
            e_up_str = np.char.strip(np.char.slice(lines_array, 105, 120))
            e_low_str = np.char.strip(np.char.slice(lines_array, 120, 135))
            g_up_str = np.char.strip(np.char.slice(lines_array, 135, 142))
            g_low_str = np.char.strip(np.char.slice(lines_array, 142, 149))
            
            # Convert to numeric types using vectorized operations
            nr = nr_str.astype(np.int32)
            lam = lam_str.astype(np.float32)
            freq = freq_str.astype(np.float64) * 1e9  # Convert GHz to Hz
            a_stein = a_stein_str.astype(np.float32)
            e_up = e_up_str.astype(np.float32)
            e_low = e_low_str.astype(np.float32)
            g_up = g_up_str.astype(np.float32)
            g_low = g_low_str.astype(np.float32)
            
            # Create dictionary list efficiently using zip
            lines_data = [
                {
                    'nr': int(nr[i]),
                    'lev_up': str(lev_up[i]),
                    'lev_low': str(lev_low[i]),
                    'lam': float(lam[i]),
                    'freq': float(freq[i]),
                    'a_stein': float(a_stein[i]),
                    'e_up': float(e_up[i]),
                    'e_low': float(e_low[i]),
                    'g_up': float(g_up[i]),
                    'g_low': float(g_low[i])
                }
                for i in range(len(lines_array))
            ]
            
            return lines_data
            
        except Exception as e:
            if self.debug:
                print(f"Debug: Vectorized parsing failed: {e}")
            return None
