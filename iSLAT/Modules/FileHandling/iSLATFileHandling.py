import os
import csv
import json
import pandas as pd
import numpy as np
from ...Constants import MOLECULES_DATA
from .molecular_data_reader import read_molecular_data


#from pathlib import Path

from iSLAT.Modules.FileHandling import *

from typing import Dict, List, Optional, Tuple, Callable, Any, Union, TYPE_CHECKING
if TYPE_CHECKING:
    from iSLAT.Modules.DataTypes.MoleculeDict import MoleculeDict
    from iSLAT.Modules.DataTypes.Molecule import Molecule

def load_user_settings(file_path=user_configuration_file_path, file_name=user_configuration_file_name, theme_file_path=theme_file_path):
    """ load_user_settings() loads the user settings from the UserSettings.json file."""
    file = os.path.join(file_path, file_name)
    if os.path.exists(file):
        with open(file, 'r') as f:
            user_settings = json.load(f)
    else:
        # If the file does not exist, return default settings and save them as a new json file
        default_settings = {
            "first_startup": True,
            "reload_default_files": True,
            "theme": "LightTheme"
        }
        with open(file, 'w') as f:
            json.dump(default_settings, f, indent=4)
        user_settings = default_settings
    
    # append theme information to the user settings dictonary
    theme_file = f"{theme_file_path}/{user_settings['theme']}.json"
    if os.path.exists(theme_file):
        with open(theme_file, 'r') as f:
            theme_settings = json.load(f)
        user_settings["theme"] = theme_settings
    return user_settings

def read_from_csv(file_path=save_folder_path, file_name=molsave_file_name):
    file = os.path.join(file_path, file_name)
    if os.path.exists(file):
        try:
            with open(file, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                return [row for row in reader]
        except FileNotFoundError:
            pass
    return MOLECULES_DATA

def read_default_csv(file_path=defaults_file_path, file_name=defaults_file_name):
    file = os.path.join(file_path, file_name)
    if os.path.exists(file):
        try:
            with open(file, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                return {row['Molecule Name']: row for row in reader if 'Molecule Name' in row}
        except FileNotFoundError:
            pass
    return MOLECULES_DATA

def read_from_user_csv(file_path=save_folder_path, file_name=molecule_list_file_name):
    file = os.path.join(file_path, file_name)
    defaults_file = os.path.join(defaults_file_path, defaults_file_name)
    if os.path.exists(file):
        try:
            with open(file, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                return {row['Molecule Name']: row for row in reader if 'Molecule Name' in row}
        except FileNotFoundError:
            pass
    #return MOLECULES_DATA
    elif os.path.exists(defaults_file):
        try:
            with open(defaults_file, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                return {row['Molecule Name']: row for row in reader if 'Molecule Name' in row}
        except FileNotFoundError:
            pass

def read_full_molecule_parameters(file_path=config_file_path, file_name=default_molecule_parameters_file_name):
    """
    read_default_molecule_parameters() updates the default molecule parameters from the DefaultMoleculeParameters.json file.
    """
    file = os.path.join(file_path, file_name)
    with open(file, 'r') as f:
        default_molecule_parameters = json.load(f)
    return default_molecule_parameters

def read_default_molecule_parameters(file_path=config_file_path, file_name=default_molecule_parameters_file_name):
    """
    read_default_molecule_parameters() updates the default molecule parameters from the DefaultMoleculeParameters.json file.
    """
    file = os.path.join(file_path, file_name)
    with open(file, 'r') as f:
        default_molecule_parameters = json.load(f)["default_initial_params"]
    return default_molecule_parameters

def read_initial_molecule_parameters(file_path=config_file_path, file_name=default_initial_parameters_file_name):
    """
    read_initial_molecule_parameters() updates the initial molecule parameters from the DefaultMoleculeParameters.json file.
    """
    file = os.path.join(file_path, file_name)
    with open(file, 'r') as f:
        initial_molecule_parameters = json.load(f)["initial_parameters"]
    return initial_molecule_parameters

def read_save_data(file_path = save_folder_path, file_name=molecule_list_file_name):
    """
    read_save_data() loads the save data from the SAVES folder.
    It returns a list of dictionaries with the save data.
    """
    #save_file = os.path.join("SAVES", "molecules_list.csv")
    file = os.path.join(file_path, file_name)
    defaults_file = os.path.join(defaults_file_path, defaults_file_name)
    if os.path.exists(file):
        try:
            df = pd.read_csv(file)
            savedata = {row['Molecule Name']: {col: row[col] for col in df.columns if col != 'Molecule Name'} for _, row in df.iterrows()}
            return savedata
        except Exception as e:
            print(f"Error reading save file: {e}")
            savedata = {}
            return savedata
    elif os.path.exists(defaults_file):
        try:
            df = pd.read_csv(defaults_file)
            savedata = {row['Molecule Name']: {col: row[col] for col in df.columns if col != 'Molecule Name'} for _, row in df.iterrows()}
            print("Huz:", savedata)
            return savedata
        except Exception as e:
            print(f"Error reading defaults file: {e}")
            return {}
    else:
        print("No save file found.")
        savedata = {}
        return savedata

def read_HITRAN_data(file_path):
    """
    read_HITRAN_data(file_path) reads the HITRAN .par file at the given path.
    Returns the contents as a list of lines (or processes to DataFrame if needed).
    """
    if not os.path.exists(file_path):
        #print(f"HITRAN file '{file_path}' does not exist.")
        return []

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
        #print(f"Successfully read HITRAN data from {file_path}")
        return lines
    except Exception as e:
        #print(f"Failed to read HITRAN file '{file_path}': {e}")
        return []

def read_line_saves(file_path=save_folder_path, file_name=line_saves_file_name) -> pd.DataFrame:
    if not file_path or not file_name:
        return pd.DataFrame()
    
    print(f"joining {file_path} and {file_name}")

    filename = os.path.join(file_path, file_name)
    if os.path.exists(filename):
        try:
            return pd.read_csv(filename)
        except Exception as e:
            print(f"Error reading line saves file: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def save_line(line_info, file_path=line_saves_file_path, file_name=line_saves_file_name, overwritefile=False):
    """Save a line to the line saves file."""
    filename = os.path.join(file_path, file_name)
    
    # Sanitize line_info to ensure no objects get saved as strings
    clean_line_info = {}
    for key, value in line_info.items():
        if key == 'species' and hasattr(value, 'name'):
            # If species is a Molecule object, extract the name
            clean_line_info[key] = str(value.name)
        elif isinstance(value, (int, float, str, bool)) or value is None:
            # Only save basic types
            clean_line_info[key] = value
        else:
            # Convert other types to string but warn
            print(f"Warning: Converting {key}={type(value)} to string in saved line")
            clean_line_info[key] = str(value)
    
    #print(f"Saving line to {filename}")
    df = pd.DataFrame([clean_line_info])

    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)
    
    # check to see if the file is empty
    if os.path.exists(filename) and os.path.getsize(filename) > 0:
        do_header = False
    else:
        do_header = True

    # Save the line to the CSV file
    if overwritefile and os.path.exists(filename):
        df.to_csv(filename, mode='w', header=True, index=False)
    else:
        df.to_csv(filename, mode='a', header=do_header, index=False)
    print(f"Saved line at ~{clean_line_info['lam']:.4f} μm to {filename}")

def save_fit_results(fit_results_data, file_path = line_saves_file_path, file_name= fit_save_lines_file_name, overwritefile=True):
    """
    Save fit results data to CSV file.
    
    Parameters
    ----------
    fit_results_data : list of dict
        List of dictionaries containing fit results
    file_path : str
        Directory path to save the file
    file_name : str
        Name of the CSV file
        
    Returns
    -------
    str
        Full path to the saved file
    """
    
    full_path = os.path.join(file_path, file_name)
    
    # Save each fit result using the existing save_line function
    # Overwrite the file only for the first entry if overwritefile is True
    if overwritefile and os.path.exists(full_path):
        # Clear the output but do not delete the file
        open(full_path, 'w').close()
    for fit_result in fit_results_data:
        save_line(fit_result, file_path=file_path, file_name=file_name)
    
    return full_path

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.inf):
            return "Infinity"
        return json.JSONEncoder.default(self, obj)

def save_deblended_models(deblended_data, file_path=deblend_models_file_path, file_name=deblend_models_file_name):
    """
    Save deblended model data to CSV file.
    
    Parameters
    ----------
    deblended_data : dict from LMFIT summary function
        List of dictionaries containing deblended model data
    file_path : str
        Directory path to save the file
    file_name : str
        Name of the CSV file
        
    Returns
    -------
    str
        Full path to the saved file
    """
    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)

    # Save the deblended data to the CSV file
    df = pd.DataFrame(deblended_data)
    df.to_csv(os.path.join(file_path, file_name), index=False)

    return os.path.join(file_path, file_name)

def save_deblended_fit_stats(deblended_data, file_path=deblended_fit_stats_file_path, file_name=deblended_fit_stats_file_name):
    """
    Save deblended model data to JSON file.
    
    Parameters
    ----------
    deblended_data : dict from LMFIT summary function
        List of dictionaries containing deblended model data
    file_path : str
        Directory path to save the file
    file_name : str
        Name of the JSON file
        
    Returns
    -------
    str
        Full path to the saved file
    """
    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)

    # Save the deblended data to the JSON file
    with open(os.path.join(file_path, file_name), 'w') as f:
        json.dump(deblended_data, f, indent=4, cls=NpEncoder)

    return os.path.join(file_path, file_name)

def save_deblended_fit_stats_and_models(deblended_data, components_data, file_path=deblended_fit_stats_file_path, stats_file_name=deblended_fit_stats_file_name, models_file_name=deblend_models_file_name):
    """
    Save deblended fit statistics to JSON file and models to CSV file.
    
    Parameters
    ----------
    deblended_data : dict from LMFIT summary function
        Dictionary containing deblended fit statistics
    models_data : dict from LMFIT summary function
        Dictionary containing deblended model data
    file_path : str
        Directory path to save the files
    stats_file_name : str
        Name of the JSON file for fit statistics
    models_file_name : str
        Name of the CSV file for models
    """
    # parse deblended_data into stats and models
    #stats_data = {k: v for k, v in deblended_data.items() if k != 'params'}
    #models_data = deblended_data.get('params', {})
    stats_data = deblended_data
    models_data = components_data

    save_deblended_fit_stats(stats_data, file_path=file_path, file_name=stats_file_name)
    save_deblended_models(models_data, file_path=file_path, file_name=models_file_name)

def read_spectral_data(file_path : str):
    """
    read_spectral_data() reads the spectral data from the provided file path.
    Returns a DataFrame with the spectral data.
    """
    if os.path.exists(file_path):
        try:
            if (file_path.endswith('.csv') or file_path.endswith('.txt')) and os.path.isfile(file_path):
                df = pd.read_csv(file_path)
                return df
            elif file_path.endswith('.dat') and os.path.isfile(file_path):
                columns_to_load = ['wave', 'flux']
                df = pd.DataFrame(np.loadtxt(file_path), columns=columns_to_load)
                return df
        except Exception as e:
            print(f"Error reading spectral data: {e}")
            return pd.DataFrame()
    else:
        print("Spectral data file does not exist.")
        return pd.DataFrame()

def write_molecules_to_csv(molecules_dict, file_path=save_folder_path, file_name=None, loaded_spectrum_name=None):
    """
    Write molecule parameters to CSV file.
    
    Parameters:
    -----------
    molecules_dict : MoleculeDict
        Dictionary containing molecule objects
    file_path : str
        Path to save folder
    file_name : str, optional
        Name of the CSV file. If None, uses molecule_list_file_name
    loaded_spectrum_name : str, optional
        Name of the currently loaded spectrum file. If provided, prefixes the filename
    """
    # Determine the base filename
    base_file_name = file_name if file_name is not None else molecule_list_file_name
    
    # Create filename based on loaded spectrum if provided
    if loaded_spectrum_name:
        spectrum_base_name = os.path.splitext(loaded_spectrum_name)[0]
        csv_filename = os.path.join(file_path, f"{spectrum_base_name}-{base_file_name}")
    else:
        csv_filename = os.path.join(file_path, base_file_name)
    
    # Ensure the directory exists
    os.makedirs(file_path, exist_ok=True)
    
    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Use field names compatible with Molecule class expectations
            header = ['Molecule Name', 'File Path', 'Molecule Label', 'Temp', 'Rad', 'N_Mol', 'Color', 'Vis', 'Dist', 'StellarRV', 'FWHM', 'Broad', 'RV Shift']
            writer.writerow(header)
            
            for mol_name, mol_obj in molecules_dict.items():
                row = [
                    mol_name,
                    getattr(mol_obj, 'filepath', getattr(mol_obj, 'hitran_data', '')),
                    getattr(mol_obj, 'displaylabel', mol_name),
                    getattr(mol_obj, 'temp', 600),
                    getattr(mol_obj, 'radius', 0.5),
                    getattr(mol_obj, 'n_mol', 1e17),
                    getattr(mol_obj, 'color', '#FF0000'),
                    getattr(mol_obj, 'is_visible', True),
                    getattr(molecules_dict, '_global_dist', 140),
                    getattr(molecules_dict, '_global_stellar_rv', 0),
                    getattr(mol_obj, 'fwhm', 200),       # Default FWHM in km/s
                    getattr(mol_obj, 'broad', 2.5),       # Default broadening
                    getattr(mol_obj, 'rv_shift', 0),      # Default RV shift
                ]
                writer.writerow(row)
        
        return csv_filename
        
    except Exception as e:
        print(f"Error saving molecule parameters: {e}")
        return None

def load_atomic_lines(file_path=atomic_lines_file_name):
    """
    Load atomic line database from CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to the atomic lines CSV file
        
    Returns
    -------
    pandas.DataFrame
        DataFrame containing atomic line data with columns: wave, species, line
    """
    try:
        # Try to find the atomic lines file relative to the current working directory
        if not os.path.exists(file_path):
            # Try alternative paths
            alt_paths = [
                os.path.join('iSLAT', file_path),
                os.path.join('..', file_path),
                os.path.join(os.path.dirname(__file__), '..', file_path)
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    file_path = alt_path
                    break
            else:
                print(f"Warning: Atomic lines file not found at {file_path} or alternative paths")
                return pd.DataFrame()
        
        atomic_lines = pd.read_csv(file_path)
        print(f"Loaded {len(atomic_lines)} atomic lines from {file_path}")
        return atomic_lines
        
    except Exception as e:
        print(f"Error loading atomic lines: {str(e)}")
        return pd.DataFrame()

def load_molecular_data_from_par(molecule_name, filename):
    """
    Load molecular data from a .par file.
    
    This function provides a convenient interface to load molecular line data
    and partition functions from .par format files.
    
    Parameters
    ----------
    molecule_name : str
        Name of the molecule (e.g., "CO", "H2O")
    filename : str
        Path to the .par file
        
    Returns
    -------
    tuple
        (partition_function, lines_data) where:
        - partition_function: namedtuple with temperature and Q values
        - lines_data: list of dictionaries containing line data
        
    Examples
    --------
    >>> partition, lines = load_molecular_data_from_par("H2O", "path/to/h2o.par")
    >>> print(f"Loaded {len(lines)} lines for {molecule_name}")
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Molecular data file not found: {filename}")
    
    return read_molecular_data(molecule_name, filename)

def save_output_line_measurements(output_line_measurements, file_path=None, file_name=None):
    from tkinter import filedialog
    
    # Define appropriate file types for measurements
    filetypes = [
        #('Text Files', '*.txt'),
        ('CSV Files', '*.csv'),
        #('DAT Files', '*.dat'),
        ('All Files', '*.*')
    ]
    
    if not os.path.exists(set_output_file_folder_path):
        os.makedirs(set_output_file_folder_path)

    # Open file dialog
    file_path = filedialog.asksaveasfilename(
        title="Select Output Line Measurements File",
        filetypes=filetypes,
        initialdir=set_output_file_folder_path,
        defaultextension=".csv",
    )
    
    if file_path:
        # Store the file path in the islat_class
        file_name = os.path.basename(file_path)
        print(f"Output line measurements loaded: {file_name}")
    else:
        print("No output line measurements file selected.")
        return
    
    return file_path, file_name

def load_input_line_list(file_path=None, file_name=None):
    from tkinter import filedialog
    
    # Define appropriate file types for line lists
    filetypes = [
        ('CSV Files', '*.csv'),
        #('DAT Files', '*.dat'),
        ('All Files', '*.*')
    ]
    
    # Open file dialog
    file_path = filedialog.askopenfilename(
        title="Select Input Line List File",
        filetypes=filetypes,
        initialdir=set_input_file_folder_path,
        defaultextension=".csv",
    )

    print(f"file_path is {file_path}")
    
    if file_path:
        # Store the file path in the islat_class
        file_name = os.path.basename(file_path)
        print(f"Input line list loaded: {file_name}")
    else:
        print("No input line list file selected.")
        return
    
    return file_path, file_name

def load_control_panel_fields_config(file_path=None, file_name="ControlPanelFields.json"):
    """
    load_control_panel_fields_config() loads the control panel field definitions from the ControlPanelFields.json file.
    Returns a dictionary containing global_fields and molecule_fields configurations.
    """
    if file_path is None:
        # Get the directory of this module and construct the path to the config directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to iSLAT directory and then to DATAFILES/CONFIG
        file_path = os.path.join(current_dir, "..", "..", "DATAFILES", "CONFIG")
    
    file = os.path.join(file_path, file_name)
    try:
        with open(file, 'r') as f:
            config = json.load(f)
        
        # Convert string datatype names to actual types
        for field_dict in [config.get('global_fields', {}), config.get('molecule_fields', {})]:
            for field_key, field_config in field_dict.items():
                # Skip documentation fields that start with underscore
                if 'datatype' in field_config:
                    datatype_str = field_config['datatype']
                    if datatype_str == 'float':
                        field_config['datatype'] = float
                    elif datatype_str == 'int':
                        field_config['datatype'] = int
                    elif datatype_str == 'str':
                        field_config['datatype'] = str
                    # Add other datatypes as needed
        
        return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading control panel fields config: {e}")
        # Return default configuration if file is missing or invalid
        return {"global_fields": {}, "molecule_fields": {}}

def generate_all_csv(molecules_data: 'MoleculeDict', output_dir=models_folder_path, wave_data=None):
    """
    Generate CSV files for all molecules in the MoleculeDict.
    
    Parameters:
    -----------
    molecules_data : MoleculeDict
        Dictionary containing molecule objects
    output_dir : str
        Output directory path
    wave_data : np.ndarray, optional
        Wavelength array to use for flux calculation
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Export individual molecule spectra
    for mol_name, molecule in molecules_data.items():
        # Skip if molecule is not visible or doesn't have plot data
        if not molecule.is_visible:
            continue
            
        # Get wavelength and flux data from molecule using get_flux method
        if wave_data is not None:
            try:
                lambdas, fluxes = molecule.get_flux(
                    wavelength_array=wave_data, 
                    return_wavelengths=True, 
                    interpolate_to_input=False
                )
            except Exception as e:
                print(f"Error getting flux data for {mol_name}: {e}")
                continue
        else:
            print(f"No wavelength data available for {mol_name}")
            continue

        # Check if we have valid data
        if (lambdas is None or fluxes is None or 
            len(lambdas) == 0 or len(fluxes) == 0 or 
            len(lambdas) != len(fluxes)):
            print(f"Invalid data for {mol_name}, skipping...")
            continue

        # Write individual molecule CSV
        csv_file_path = os.path.join(output_dir, f"{mol_name}_spec_output.csv")
        try:
            with open(csv_file_path, "w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["wave", "flux"])
                for wave, flux in zip(lambdas, fluxes):
                    csv_writer.writerow([wave, flux])
            print(f"Exported {mol_name} spectrum to {csv_file_path}")
        except Exception as e:
            print(f"Error writing CSV for {mol_name}: {e}")

    # Generate summed spectrum if we have visible molecules and wave data
    visible_molecules = list(molecules_data.get_visible_molecules())
    if visible_molecules and wave_data is not None:
        try:
            # Get summed flux from MoleculeDict - now returns (wavelengths, flux)
            summed_wavelengths, summed_flux = molecules_data.get_summed_flux(wave_data, visible_only=True)
            
            if summed_flux is not None and len(summed_flux) > 0:
                # Write summed spectrum CSV using the combined wavelength grid
                csv_file_path = os.path.join(output_dir, "SUM_spec_output.csv")
                with open(csv_file_path, "w", newline="") as csv_file:
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(["wave", "flux"])
                    for wave, flux in zip(summed_wavelengths, summed_flux):
                        csv_writer.writerow([wave, flux])
                print(f"Exported summed spectrum to {csv_file_path}")
            else:
                print("No valid summed flux data to export")
        except Exception as e:
            print(f"Error generating summed spectrum: {e}")
    
    print(f'All models exported to {output_dir}')

def generate_csv(molecules_data: 'MoleculeDict', mol_name: str, data_field, output_dir=models_folder_path, wave_data=None):
    """
    Generate CSV file for a specific molecule or summed spectrum.
    
    Parameters:
    -----------
    molecules_data : MoleculeDict
        Dictionary containing molecule objects
    mol_name : str
        Name of the molecule to export, or "SUM" for summed spectrum, or "ALL" for all molecules
    output_dir : str
        Output directory path
    wave_data : np.ndarray, optional
        Wavelength array to use for flux calculation
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    if mol_name == "SUM":
        # Generate summed spectrum
        if wave_data is None:
            print("Wave data is required to generate summed spectrum")
            return
            
        try:
            summed_wavelengths, summed_flux = molecules_data.get_summed_flux(wave_data, visible_only=True)
            
            if summed_flux is None or len(summed_flux) == 0:
                print("No visible molecules or valid flux data for summed spectrum")
                return

            # Write summed spectrum CSV using the combined wavelength grid
            csv_file_path = os.path.join(output_dir, "SUM_spec_output.csv")
            with open(csv_file_path, "w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["wave", "flux"])
                for wave, flux in zip(summed_wavelengths, summed_flux):
                    csv_writer.writerow([wave, flux])
            
            print(f'SUM model exported to {output_dir}')
            
        except Exception as e:
            print(f"Error generating summed spectrum: {e}")

    elif mol_name == "ALL":
        # Generate all molecule CSVs
        generate_all_csv(molecules_data, output_dir, wave_data)

    else:
        # Generate CSV for specific molecule
        if mol_name not in molecules_data:
            print(f"Molecule '{mol_name}' not found in molecules_data")
            return

        molecule: Molecule = molecules_data[mol_name]

        # Get wavelength and flux data from molecule using get_flux method
        if wave_data is not None:
            try:
                lambdas, fluxes = molecule.get_flux(
                    wavelength_array=wave_data, 
                    return_wavelengths=True, 
                    interpolate_to_input=False
                )
            except Exception as e:
                print(f"Error getting flux data for {mol_name}: {e}")
                return
        else:
            print(f"No wavelength data available for {mol_name}")
            return

        # Check if we have valid data
        if (lambdas is None or fluxes is None or 
            len(lambdas) == 0 or len(fluxes) == 0 or 
            len(lambdas) != len(fluxes)):
            print(f"Invalid data for {mol_name}")
            return

        try:
            # Write spectrum CSV
            csv_file_path = os.path.join(output_dir, f"{mol_name}_spec_output.csv")
            with open(csv_file_path, "w", newline="") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["wave", "flux"])
                for wave, flux in zip(lambdas, fluxes):
                    csv_writer.writerow([wave, flux])
            
            # Export line parameters if available
            if hasattr(molecule, 'intensity') and molecule.intensity is not None:
                if hasattr(molecule.intensity, 'get_table'):
                    try:
                        line_params = molecule.intensity.get_table
                        if hasattr(line_params, 'to_csv'):
                            line_params_path = os.path.join(output_dir, f"{mol_name}_line_params.csv")
                            line_params.to_csv(line_params_path, index=False)
                            print(f"Exported line parameters for {mol_name}")
                    except Exception as e:
                        print(f"Error exporting line parameters for {mol_name}: {e}")


            data_field.insert_text(f'{mol_name} model exported to {output_dir}')
            print(f'{mol_name} model exported to {output_dir}')
            
        except Exception as e:
            data_field.insert_text(f"Error writing CSV for {mol_name}: {e}")
            print(f"Error writing CSV for {mol_name}: {e}")