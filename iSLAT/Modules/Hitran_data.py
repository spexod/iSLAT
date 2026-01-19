from astroquery import hitran
import pandas as pd
import urllib.request
import ssl
#from .hitran_utils import get_molecule_identifier
#from .global_identifier import get_global_identifier
from astropy import units as un

from typing import List, Tuple
import os
import datetime
#from .Hitran_data import get_Hitran_data
from .FileHandling.partition_function_writer import write_partition_function
from .FileHandling.line_data_writer import write_line_data

context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

def get_global_identifier(molecule_name, isotopologue_number=1):
    '''                                                                                                                                
    For a given input molecular formula, return the corresponding HITRAN *global* identifier number.
    For more info, see https://hitran.org/docs/iso-meta/ 
                                                                                                                                       
    Parameters                                                                                                                         
    ----------                                                                                                                         
    molecular_formula : str                                                                                                            
        The string describing the molecule.              
    isotopologue_number : int, optional
        The isotopologue number, from most to least common.                                                                              
                                                                                                                                       
    Returns                                                                                                                            
    -------                                                                                                                            
    G : int                                                                                                                            
        The HITRAN global identifier number.                                                                                        
    '''

    mol_isot_code = molecule_name + '_' + str(isotopologue_number)

    trans = { 'H2O_1':1, 'H2O_2':2, 'H2O_3':3, 'H2O_4':4, 'H2O_5':5, 'H2O_6':6, 'H2O_7':129,
               'CO2_1':7,'CO2_2':8,'CO2_3':9,'CO2_4':10,'CO2_5':11,'CO2_6':12,'CO2_7':13,'CO2_8':14,
               'CO2_9':121,'CO2_10':15,'CO2_11':120,'CO2_12':122,
               'O3_1':16,'O3_2':17,'O3_3':18,'O3_4':19,'O3_5':20,
               'N2O_1':21,'N2O_2':22,'N2O_3':23,'N2O_4':24,'N2O_5':25,
               'CO_1':26,'CO_2':27,'CO_3':28,'CO_4':29,'CO_5':30,'CO_6':31,
               'CH4_1':32,'CH4_2':33,'CH4_3':34,'CH4_4':35,
               'O2_1':36,'O2_2':37,'O2_3':38,
               'NO_1':39,'NO_2':40,'NO_3':41,
               'SO2_1':42,'SO2_2':43,
               'NO2_1':44,
               'NH3_1':45,'NH3_2':46,
               'HNO3_1':47,'HNO3_2':117,
               'OH_1':48,'OH_2':49,'OH_3':50,
               'HF_1':51,'HF_2':110,
               'HCl_1':52,'HCl_2':53,'HCl_3':107,'HCl_4':108,
               'HBr_1':54,'HBr_2':55,'HBr_3':111,'HBr_4':112,
               'HI_1':56,'HI_2':113,
               'ClO_1':57,'ClO_2':58,
               'OCS_1':59,'OCS_2':60,'OCS_3':61,'OCS_4':62,'OCS_5':63,
               'H2CO_1':64,'H2CO_2':65,'H2CO_3':66,
               'HOCl_1':67,'HOCl_2':68,
               'N2_1':69,'N2_2':118,
               'HCN_1':70,'HCN_2':71,'HCN_3':72,
               'CH3Cl_1':73,'CH3Cl_2':74,
               'H2O2_1':75,
               'C2H2_1':76,'C2H2_2':77,'C2H2_3':105,
               'C2H6_1':78,'C2H6_2':106,
               'PH3_1':79,
               'COF2_1':80,'COF2_2':119,
               'SF6_1':126,
               'H2S_1':81,'H2S_2':82,'H2S_3':83,
               'HCOOH_1':84,
               'HO2_1':85,
               'O_1':86,
               'ClONO2_1':127,'ClONO2_2':128,
               'NO+_1':87,
               'HOBr_1':88,'HOBr_2':89,
               'C2H4_1':90,'C2H4_2':91,
               'CH3OH_1':92,
               'CH3Br_1':93,'CH3Br_2':94,
               'CH3CN_1':95,
               'CF4_1':96,
               'C4H2_1':116,
               'HC3N_1':109,
               'H2_1':103,'H2_2':115,
               'CS_1':97,'CS_2':98,'CS_3':99,'CS_4':100,
               'SO3_1':114,
               'C2N2_1':123,
               'COCl2_1':124,'COCl2_2':125,
               'SO_1': 146, 'SO_2': 147,
               'CH3F_1': 144,
               'GeH4_1': 139, 'GeH4_2': 140, 'GeH4_3': 141, 'GeH4_4': 142, 'GeH4_5': 143,
               'CS2_1': 131, 'CS2_2': 132, 'CS2_3': 133, 'CS2_4': 134,
               'CH3I_1': 145,
               'NF3_1': 136,
               }
 
    return trans[mol_isot_code]

def get_molecule_identifier(molecule_name):
    '''                                                                                                                                
    For a given input molecular formula, return the corresponding HITRAN molecule identifier number.                                   
                                                                                                                                       
    Parameters                                                                                                                         
    ----------                                                                                                                         
    molecular_formula : str                                                                                                            
        The string describing the molecule.                                                                                            
                                                                                                                                       
    Returns                                                                                                                            
    -------                                                                                                                            
    M : int                                                                                                                            
        The HITRAN molecular identifier number.                                                                                        
    '''
    
    trans = { '1':'H2O',    '2':'CO2',   '3':'O3',      '4':'N2O',   '5':'CO',    '6':'CH4',   '7':'O2',     '8':'NO',
              '9':'SO2',   '10':'NO2',  '11':'NH3',    '12':'HNO3', '13':'OH',   '14':'HF',   '15':'HCl',   '16':'HBr',
             '17':'HI',    '18':'ClO',  '19':'OCS',    '20':'H2CO', '21':'HOCl', '22':'N2',   '23':'HCN',   '24':'CH3Cl',
             '25':'H2O2',  '26':'C2H2', '27':'C2H6',   '28':'PH3',  '29':'COF2', '30':'SF6',  '31':'H2S',   '32':'HCOOH',
             '33':'HO2',   '34':'O',    '35':'ClONO2', '36':'NO+',  '37':'HOBr', '38':'C2H4', '39':'CH3OH', '40':'CH3Br',
             '41':'CH3CN', '42':'CF4',  '43':'C4H2',   '44':'HC3N', '45':'H2',   '46':'CS',   '47':'SO3',   '48':'C2N2',
            '49':'COCl2',   '50':'SO',   '51':'CH3F',   '52':'GeH4',   '53':'CS2',   '54':'CH3I',   '55':'NF3'}
    ## Invert the dictionary.                                                                                                          
    trans = {v:k for k,v in trans.items()}
    return(int(trans[molecule_name]))

def get_molar_mass(molecule_name, isotopologue_number):
    M = get_molecule_identifier(molecule_name)
    ISO_Info = hitran.Hitran.ISO[(M, isotopologue_number)]
    molar_mass = ISO_Info[-2]
    return molar_mass

def get_Hitran_data(Molecule_name, isotopologue_number, min_vu, max_vu):
    try:
        M = get_molecule_identifier(Molecule_name)

        Htbl = hitran.Hitran.query_lines(molecule_number=M, 
                                         isotopologue_number=isotopologue_number,
                                         min_frequency=min_vu/un.cm, 
                                         max_frequency=max_vu/un.cm)

        G = get_global_identifier(Molecule_name, isotopologue_number=isotopologue_number)

        qurl = 'https://hitran.org/data/Q/' + 'q' + str(G) + '.txt'
        handle = urllib.request.urlopen(qurl, context=context)
        qdata = pd.read_csv(handle, sep=' ', skipinitialspace=True, names=['temp', 'q'], header=None)

        #print(f'Downloaded HITRAN data for molecule {Molecule_name} (isotopologue {isotopologue_number}) with global ID {G} and molecule ID {M}.')
        #print(f'Htbl:\n{Htbl}')
        #print(f'qdata:\n{qdata}')
        #print(f'Number of lines retrieved for {Molecule_name}: {len(Htbl)}')
        #ISO_Info = hitran.Hitran.ISO[(M, isotopologue_number)]
        #print(f'Isotopologue info: {ISO_Info}')
        #molar_mass = ISO_Info[-2]
        #print(f'Molar mass (g/mol): {molar_mass}')

        return Htbl, qdata, M, G
        
    except KeyError as e:
        error_msg = f"HITRAN data not available for molecule '{Molecule_name}' with isotopologue number {isotopologue_number}. This combination may not exist in the HITRAN database."
        print(f"Error: {error_msg}")
        raise ValueError(error_msg) from e
        
    except Exception as e:
        error_msg = f"Failed to download HITRAN data for molecule '{Molecule_name}' (isotopologue {isotopologue_number}): {str(e)}"
        print(f"Error: {error_msg}")
        raise RuntimeError(error_msg) from e

def download_hitran_data(mols, basem, isot) -> List[Tuple]:
    #mols = ["H2", "HD", "H2O", "H218O", "CO2", "13CO2", "CO", "13CO", "C18O", "CH4", "HCN", "H13CN", "NH3", "OH", "C2H2", "13CCH2", "C2H4", "C4H2", "C2H6", "HC3N"]
    #basem = ["H2", "H2", "H2O", "H2O", "CO2", "CO2", "CO", "CO", "CO", "CH4", "HCN", "HCN", "NH3", "OH", "C2H2", "C2H2", "C2H4", "C4H2", "C2H6", "HC3N"]
    #isot = [1, 2, 1, 2, 1, 2, 1, 2, 3, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1]

    #mols = ["O2"]
    #basem = ["O2", "O2"]
    #isot = [1, 2]

    missed_mols = []

    min_wave = 0.3  # micron
    max_wave = 1000  # micron

    min_vu = 1 / (min_wave / 1E6) / 100.
    max_vu = 1 / (max_wave / 1E6) / 100.

    print(' ')
    print ('Checking for HITRAN files: ...')

    for mol, bm, iso in zip(mols, basem, isot):
        save_folder = 'DATAFILES/HITRANdata'
        file_path = os.path.join(save_folder, "data_Hitran_{:}.par".format(mol))

        if os.path.exists(file_path):
            print("File already exists for mol: {:}. Skipping.".format(mol))
            continue

        print("Downloading data for mol: {:}".format(mol))
        try:
            Htbl, qdata, M, G = get_Hitran_data(bm, iso, min_vu, max_vu)
            os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

            with open(file_path, 'w') as fh:
                fh.write("# HITRAN {:}; id:{:}; iso:{:};gid:{:}\n".format(mol, M, iso, G))
                fh.write("# Molar Mass: {:}\n".format(get_molar_mass(bm, iso)))
                fh.write("# Downloaded from the Hitran website\n")
                fh.write("# {:s}\n".format(str(datetime.date.today())))
                fh = write_partition_function(fh, qdata)
                fh = write_line_data(fh, Htbl)

            print("Data for Mol: {:} downloaded and saved.".format(mol))
            
        except (ValueError, RuntimeError) as e:
            print(f"Skipping molecule {mol}: {str(e)}")
            missed_mols.append((bm, mol, isot))
            continue
        except Exception as e:
            print(f"Unexpected error downloading {mol}: {str(e)}")
            continue

    return missed_mols