"""
LineSaveService - Service for saving spectral line data.

This module handles the logic for extracting and saving line information,
separated from GUI concerns to enable reuse and testing.
"""

import numpy as np
from typing import Any, Dict, Optional, Tuple

class LineSaveService:
    """
    Service class for saving spectral line data.
    
    This class handles the logic for extracting line information from
    selections and formatting it for saving.
    """
    
    def __init__(self, islat_instance):
        """
        Initialize the line save service.
        
        Parameters
        ----------
        islat_instance : iSLAT
            Reference to the main iSLAT instance for accessing data and configuration
        """
        self.islat = islat_instance
    
    def create_default_line_info(
        self,
        center_wave: float,
        line_flux: float,
        line_err: float = 0.0
    ) -> Dict[str, Any]:
        """
        Create a default line info dictionary when no specific line is selected.
        
        Parameters
        ----------
        center_wave : float
            Center wavelength of the line
        line_flux : float
            Integrated flux of the line
        line_err : float, optional
            Error on the flux
            
        Returns
        -------
        dict
            Dictionary with default line information
        """
        return {
            'lam': center_wave,
            'wavelength': center_wave,
            'flux': line_flux,
            'intensity': line_flux,
            'e': 0.0,
            'a': 0.0,
            'g': 1.0,
            'inten': line_flux,
            'up_lev': 'Unknown',
            'low_lev': 'Unknown',
            'tau': 0.0
        }
    
    def format_line_for_save(
        self,
        selected_line_info: Dict[str, Any],
        species_name: str,
        xmin: float,
        xmax: float
    ) -> Dict[str, Any]:
        """
        Format line information for saving to file.
        
        Parameters
        ----------
        selected_line_info : dict
            Dictionary with line information from selection
        species_name : str
            Name of the molecular species
        xmin : float
            Minimum wavelength of selection
        xmax : float
            Maximum wavelength of selection
            
        Returns
        -------
        dict
            Dictionary formatted for file save
        """
        return {
            'species': species_name,
            'lev_up': selected_line_info.get('up_lev', ''),
            'lev_low': selected_line_info.get('low_lev', ''),
            'lam': selected_line_info['lam'],
            'tau': selected_line_info.get('tau', 0.0),
            'intens': selected_line_info.get('inten', selected_line_info.get('intensity', 0.0)),
            'a_stein': selected_line_info.get('a', 0.0),
            'e_up': selected_line_info.get('e', 0.0),
            'g_up': selected_line_info.get('g', 1.0),
            'e_low': selected_line_info.get('e_low', 0.0),
            'g_low': selected_line_info.get('g_low', 1.0),
            'xmin': xmin,
            'xmax': xmax,
        }
    
    def get_selection_bounds(
        self,
        selected_wave: Optional[np.ndarray],
        current_selection: Tuple[float, float],
        line_wavelength: float
    ) -> Tuple[float, float]:
        """
        Get the wavelength bounds for a selection.
        
        Parameters
        ----------
        selected_wave : np.ndarray or None
            Array of selected wavelengths
        current_selection : tuple
            Current selection bounds (xmin, xmax)
        line_wavelength : float
            Center wavelength of the line (used as fallback)
            
        Returns
        -------
        tuple
            (xmin, xmax) wavelength bounds
        """
        if selected_wave is not None and len(selected_wave) > 0:
            xmin = selected_wave[0] if len(selected_wave) > 0 else line_wavelength - 0.01
            xmax = selected_wave[-1] if len(selected_wave) > 1 else line_wavelength + 0.01
        else:
            xmin, xmax = current_selection
        
        return xmin, xmax
    
    def extract_line_info_from_selection(
        self,
        main_plot: Any,
        save_type: str = "selected"
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        """
        Extract line information from the current plot selection.
        
        Parameters
        ----------
        main_plot : iSLATPlot
            Reference to the main plot instance
        save_type : str
            Type of save: "selected" or "strongest"
            
        Returns
        -------
        tuple
            (line_info_dict, error_message) - error_message is empty string on success
        """
        if not hasattr(main_plot, 'current_selection') or main_plot.current_selection is None:
            return None, "No region selected for saving."
        
        if save_type == "strongest":
            selected_line_info = main_plot.find_strongest_line_from_data()
            if selected_line_info is None:
                return None, "No valid line found in selection."
                
        elif save_type == "selected":
            selected_line_info = main_plot.selected_line
            if selected_line_info is None:
                # Fallback: create basic line info from selection bounds
                xmin, xmax = main_plot.current_selection
                center_wave = (xmin + xmax) / 2.0
                
                # Calculate flux integral in the selected range
                err_data = getattr(self.islat, 'err_data', None)
                line_flux, line_err = main_plot.flux_integral(
                    self.islat.wave_data,
                    self.islat.flux_data,
                    err_data,
                    xmin,
                    xmax
                )
                
                selected_line_info = self.create_default_line_info(
                    center_wave, line_flux, line_err
                )
        else:
            return None, "Invalid save type specified."
        
        return selected_line_info, ""