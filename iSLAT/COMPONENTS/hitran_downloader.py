import os
import datetime
from COMPONENTS.Hitran_data import get_Hitran_data
from COMPONENTS.partition_function_writer import write_partition_function
from COMPONENTS.line_data_writer import write_line_data

def download_hitran_data(mols, basem, isot):
    #mols = ["H2", "HD", "H2O", "H218O", "CO2", "13CO2", "CO", "13CO", "C18O", "CH4", "HCN", "H13CN", "NH3", "OH", "C2H2", "13CCH2", "C2H4", "C4H2", "C2H6", "HC3N"]
    #basem = ["H2", "H2", "H2O", "H2O", "CO2", "CO2", "CO", "CO", "CO", "CH4", "HCN", "HCN", "NH3", "OH", "C2H2", "C2H2", "C2H4", "C4H2", "C2H6", "HC3N"]
    #isot = [1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1]

    #mols = ["O2"]
    #basem = ["O2", "O2"]
    #isot = [1, 2]

    min_wave = 0.3  # micron
    max_wave = 1000  # micron

    min_vu = 1 / (min_wave / 1E6) / 100.
    max_vu = 1 / (max_wave / 1E6) / 100.

    print(' ')
    print ('Checking for HITRAN files: ...')

    for mol, bm, iso in zip(mols, basem, isot):
        save_folder = 'HITRANdata'
        file_path = os.path.join(save_folder, "data_Hitran_2020_{:}.par".format(mol))

        if os.path.exists(file_path):
            print("File already exists for mol: {:}. Skipping.".format(mol))
            continue

        print("Downloading data for mol: {:}".format(mol))
        Htbl, qdata, M, G = get_Hitran_data(bm, iso, min_vu, max_vu)
        os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

        with open(file_path, 'w') as fh:
            fh.write("# HITRAN 2020 {:}; id:{:}; iso:{:};gid:{:}\n".format(mol, M, iso, G))
            fh.write("# Downloaded from the Hitran website\n")
            fh.write("# {:s}\n".format(str(datetime.date.today())))
            fh = write_partition_function(fh, qdata)
            fh = write_line_data(fh, Htbl)

        print("Data for Mol: {:} downloaded and saved.".format(mol))
