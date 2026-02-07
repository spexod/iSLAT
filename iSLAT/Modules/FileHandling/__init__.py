from pathlib import Path
import os

# Get the absolute path to the iSLAT package directory
# This ensures paths are correct regardless of where the code is called from
_ISLAT_PACKAGE_DIR = Path(__file__).resolve().parent.parent.parent  # Go up from FileHandling -> Modules -> iSLAT
_ABSOLUTE_DATA_FILES_PATH = _ISLAT_PACKAGE_DIR / "DATAFILES"

# Relative path for backward compatibility with any code expecting relative paths
data_files_path = Path("DATAFILES")
# Absolute path based on package location - use this for reliable file access
absolute_data_files_path = str(_ABSOLUTE_DATA_FILES_PATH)

save_folder_path = data_files_path / "SAVES"
user_configuration_file_path = config_file_path = data_files_path / "CONFIG"
theme_file_path = data_files_path / "CONFIG" / "GUIThemes"
user_configuration_file_name = "UserSettings.json"

hitran_data_folder_name = Path("HITRANdata")
# Use absolute path for HITRAN data to ensure cache works from anywhere
hitran_data_folder_path = str(_ABSOLUTE_DATA_FILES_PATH / "HITRANdata")
hitran_cache_folder_path = os.path.join(hitran_data_folder_path, "cache")

example_data_folder_path = data_files_path / "EXAMPLE-data"

molsave_file_name = "molsave.csv"
molecule_list_file_name = "molecules_list.csv"

defaults_file_name = "default.csv"
defaults_file_path = config_file_path

default_molecule_parameters_file_name = "DefaultMoleculeParameters.json"
default_initial_parameters_file_name = "DefaultMoleculeParameters.json"

line_saves_file_name = "deblended_linecenter.csv"
line_saves_file_path = data_files_path / "LINESAVES"

deblend_models_file_name = "deblend_models.csv"
deblend_models_file_path = line_saves_file_path

deblended_fit_stats_file_name = "deblended_fit_statistics.json"
deblended_fit_stats_file_path = line_saves_file_path

fit_save_lines_file_name = "fit_save_lines.csv"
atomic_lines_file_name = data_files_path / "LINELISTS" / "Atomic_lines.csv"
models_folder_path = data_files_path / "MODELS"

set_input_file_folder_path = data_files_path / "LINELISTS"

set_output_file_folder_path = line_saves_file_path
set_output_file_name = "line_outputs.csv"

__all__ = [
    "data_files_path",
    "absolute_data_files_path",
    "save_folder_path",
    "config_file_path",
    "user_configuration_file_path",
    "theme_file_path",
    "user_configuration_file_name",
    "hitran_data_folder_name",
    "hitran_data_folder_path",
    "hitran_cache_folder_path",
    "molsave_file_name",
    "molecule_list_file_name",
    "defaults_file_name",
    "defaults_file_path",
    "default_molecule_parameters_file_name",
    "default_initial_parameters_file_name",
    "line_saves_file_name",
    "line_saves_file_path",
    "fit_save_lines_file_name",
    "atomic_lines_file_name",
    "models_folder_path",
    "set_input_file_folder_path",
    "set_output_file_folder_path",
    "set_output_file_name",
    "deblend_models_file_name",
    "deblend_models_file_path",
    "deblended_fit_stats_file_name",
    "deblended_fit_stats_file_path",
    "example_data_folder_path"
]