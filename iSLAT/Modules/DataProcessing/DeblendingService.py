"""
DeblendingService - Service for line deblending operations.

This module handles the logic for fitting and extracting deblended line components,
separated from GUI concerns to enable reuse and testing.
"""

import os
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Callable

import iSLAT.Modules.FileHandling.iSLATFileHandling as ifh
import iSLAT.Constants as c

class DeblendingService:
    """
    Service class for line deblending operations.
    
    This class handles the logic for fitting multiple Gaussian components to blended
    spectral lines and extracting/saving the individual component information.
    """

    def __init__(self, islat_instance):
        """
        Initialize the deblending service.
        
        Parameters
        ----------
        islat_instance : iSLAT
            Reference to the main iSLAT instance for accessing data and configuration
        """
        self.islat = islat_instance
    
    def extract_deblended_components(
        self,
        line_params: Dict[str, Any],
        line_info: List[Tuple],
        molecule_name: str
    ) -> List[Dict[str, Any]]:
        """
        Extract deblended component information from fit results.
        
        Parameters
        ----------
        line_params : dict
            Dictionary of fitted line parameters with component keys like 'component_0', 'component_1'
        line_info : list of tuple
            List of (MoleculeLine, intensity, tau) tuples from the molecule's intensity data
        molecule_name : str
            Name of the active molecule
            
        Returns
        -------
        list of dict
            List of dictionaries containing extracted component information
        """
        components = []
        component_idx = 0
        
        while f'component_{component_idx}' in line_params:
            comp_params = line_params[f'component_{component_idx}']
            
            component_data = {
                'index': component_idx,
                'center': comp_params['center'],
                'center_stderr': comp_params.get('center_stderr', 0),
                'fwhm': comp_params['fwhm'],
                'fwhm_stderr': comp_params.get('fwhm_stderr', 0),
                'area': comp_params['area'],
                'area_stderr': comp_params.get('area_stderr', 0),
                'molecule_name': molecule_name
            }
            
            # Try to match with line info
            if component_idx < len(line_info):
                try:
                    current_tripple = line_info[component_idx]
                    current_line_info = current_tripple[0].get_dict()
                    current_intens = current_tripple[1]
                    current_tau = current_tripple[2]
                    
                    # Calculate Doppler shift
                    doppler = ((comp_params['center'] - current_line_info["lam"]) / 
                              current_line_info["lam"] * c.SPEED_OF_LIGHT_KMS) if current_line_info["lam"] else np.nan
                    
                    component_data.update({
                        'lev_up': current_line_info['lev_up'],
                        'lev_low': current_line_info['lev_low'],
                        'lam': current_line_info['lam'],
                        'tau': current_tau,
                        'intens': current_intens,
                        'a_stein': current_line_info['a_stein'],
                        'e_up': current_line_info['e_up'],
                        'e_low': current_line_info['e_low'],
                        'g_up': current_line_info['g_up'],
                        'g_low': current_line_info['g_low'],
                        'doppler': doppler
                    })
                except Exception:
                    # If we can't get line info, just use fit parameters
                    pass
            
            components.append(component_data)
            component_idx += 1
        
        return components
    
    def format_component_for_save(
        self,
        component: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Format a component dictionary for saving to file.
        
        Parameters
        ----------
        component : dict
            Component data dictionary
            
        Returns
        -------
        dict
            Formatted dictionary ready for file save
        """
        return {
            'species': component.get('molecule_name', 'Unknown'),
            'lev_up': component.get('lev_up', ''),
            'lev_low': component.get('lev_low', ''),
            'lam': component.get('lam', 0.0),
            'tau': component.get('tau', 0.0),
            'intens': component.get('intens', 0.0),
            'a_stein': component.get('a_stein', 0.0),
            'e_up': component.get('e_up', 0.0),
            'e_low': component.get('e_low', 0.0),
            'g_up': component.get('g_up', 1.0),
            'g_low': component.get('g_low', 1.0),
            'Flux_fit': component['area'],
            'Err_fit': component['area_stderr'],
            'FWHM_fit': component['fwhm'],
            'FWHM_err': component['fwhm_stderr'],
            'Centr_fit': component['center'],
            'Centr_err': component['center_stderr'],
            'Doppler': component.get('doppler', np.nan)
        }
    
    def save_deblended_components(
        self,
        components: List[Dict[str, Any]],
        save_file_name: str
    ) -> int:
        """
        Save deblended components to a file.
        
        Parameters
        ----------
        components : list of dict
            List of component dictionaries
        save_file_name : str
            Output file name
            
        Returns
        -------
        int
            Number of successfully saved components
        """
        saved_count = 0
        
        for component in components:
            try:
                line_save_info = self.format_component_for_save(component)
                ifh.save_line(line_save_info, file_name=save_file_name)
                saved_count += 1
            except Exception:
                continue
        
        return saved_count
    
    def save_deblend_summary(
        self,
        fit_result_summary: Dict,
        fit_results_components: List,
        spectrum_base_name: str,
        save_directory: str
    ) -> Tuple[str, str]:
        """
        Save deblending summary statistics and models.
        
        Parameters
        ----------
        fit_result_summary : dict
            Summary statistics from the fitting engine
        fit_results_components : list
            Component data from the fitting engine
        spectrum_base_name : str
            Base name for output files
        save_directory : str
            Directory to save files
            
        Returns
        -------
        tuple of str
            (models_file_path, stats_file_path)
        """
        models_file_name = f"{spectrum_base_name}-deblend_models.csv"
        stats_file_name = f"{spectrum_base_name}-deblended_fit_statistics.json"
        
        ifh.save_deblended_fit_stats_and_models(
            deblended_data=fit_result_summary,
            components_data=fit_results_components,
            models_file_name=models_file_name,
            stats_file_name=stats_file_name
        )
        
        models_path = os.path.join(save_directory, models_file_name)
        stats_path = os.path.join(save_directory, stats_file_name)
        
        return models_path, stats_path
    
    def format_component_display(
        self,
        component: Dict[str, Any]
    ) -> List[str]:
        """
        Format component data for display.
        
        Parameters
        ----------
        component : dict
            Component data dictionary
            
        Returns
        -------
        list of str
            List of formatted display strings
        """
        messages = []
        
        # Handle None values in stderr parameters
        center_err = component.get('center_stderr', 0)
        center_err_str = f"{center_err:.5f}" if center_err is not None else "0"
        
        fwhm_err = component.get('fwhm_stderr', 0)
        fwhm_err_str = f"{fwhm_err:.1f}" if fwhm_err is not None else "0"
        
        area_err = component.get('area_stderr', 0)
        area_err_str = f"{area_err:.3e}" if area_err is not None else "0"
        
        messages.append(f"Centroid (μm) = {component['center']:.5f} +/- {center_err_str}")
        messages.append(f"FWHM (km/s) = {component['fwhm']:.1f} +/- {fwhm_err_str}")
        messages.append(f"Area (erg/s/cm2) = {component['area']:.3e} +/- {area_err_str}")
        
        return messages
    
    def format_single_gaussian_display(
        self,
        line_params: Dict[str, Any]
    ) -> List[str]:
        """
        Format single Gaussian fit results for display.
        
        Parameters
        ----------
        line_params : dict
            Dictionary of fitted line parameters
            
        Returns
        -------
        list of str
            List of formatted display strings
        """
        messages = []
        
        if 'center' not in line_params:
            messages.append("Could not extract fit parameters.")
            return messages
        
        # Handle None values in stderr parameters
        center_err = line_params.get('center_stderr', 0)
        center_err_str = f"{center_err:.5f}" if center_err is not None else "0"
        
        fwhm_err = line_params.get('fwhm_stderr', 0)
        fwhm_err_str = f"{fwhm_err:.5f}" if fwhm_err is not None else "0"
        
        area_err = line_params.get('area_stderr', 0)
        area_err_str = f"{area_err:.3e}" if area_err is not None else "0"
        
        messages.append(f"Centroid (μm) = {line_params['center']:.5f} +/- {center_err_str}")
        messages.append(f"FWHM (km/s) = {line_params['fwhm']:.5f} +/- {fwhm_err_str}")
        messages.append(f"Area (erg/s/cm2) = {line_params['area']:.3e} +/- {area_err_str}")
        
        return messages