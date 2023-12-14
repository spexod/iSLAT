from astroquery import hitran
import numpy
from astropy import units as un
from scipy import constants as con
import datetime
import urllib
import pandas as pd

#Using code from Nathan Hagen
#https://github.com/nzhagen/hitran
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
             '41':'CH3CN', '42':'CF4',  '43':'C4H2',   '44':'HC3N', '45':'H2',   '46':'CS',   '47':'SO3'}
    ## Invert the dictionary.                                                                                                          
    trans = {v:k for k,v in trans.items()}
    return(int(trans[molecule_name]))

def get_global_identifier(molecule_name,isotopologue_number=1):
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

    mol_isot_code=molecule_name+'_'+str(isotopologue_number)

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
               'CH3Cl_1':73,'CH3CL_2':74,
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
               'COCl2_1':124,'COCl2_2':125}
 
    return trans[mol_isot_code]


def get_Hitran_data(Molecule_name, isotopologue_number, min_vu, max_vu):

    M = get_molecule_identifier(Molecule_name)

    Htbl = hitran.Hitran.query_lines(molecule_number = M, 
                                isotopologue_number = isotopologue_number,
                                min_frequency = min_vu/un.cm, max_frequency = max_vu/un.cm)


    G = get_global_identifier(Molecule_name, isotopologue_number=isotopologue_number)

    qurl='https://hitran.org/data/Q/'+'q'+str(G)+'.txt'
    handle = urllib.request.urlopen(qurl)
    qdata = pd.read_csv(handle,sep=' ',skipinitialspace=True,names=['temp','q'],header=None)

    return Htbl, qdata, M, G

def write_partition_function(fh, qdata):

    fh.write("# Number of Partition function entries\n")
    fh.write("{:d}\n".format(len(qdata['temp'])))

    fh.write("# Temperature    Q(T)\n")
    fh.write("# [K]\n")

    for t, q in zip(qdata['temp'], qdata['q']):
        fh.write("{:>6.1f}  {:>15.6f}\n".format(t,q))
    

    return fh

def write_line_data(fh, Htbl):


    numlines = len(Htbl['molec_id'])

    fh.write("Number of lines\n")
    fh.write(str(numlines) + '\n')

    fh.write("#    Nr                        Lev_up                       Lev_low   Lambda    Frequency")
    fh.write("       Einstein-A     E_up           E_low        g_up   g_low\n")
    fh.write("#                                                                    [micron]  [GHz]    ")
    fh.write("       [s**-1]        [K]            [K]        \n")
  
    freqs = Htbl['nu'] * 100. *con.c
    freqsG = freqs/1E9

    waves = con.c/freqs*1E6

    Elo = con.h * con.c * Htbl['elower']*100 / con.k
    Eup = con.h * con.c * (Htbl['elower'] + Htbl['nu'])*100 / con.k


    for i in range(numlines):


        qqup = "_".join(Htbl['global_upper_quanta'][i].strip().split()) + "|" +\
               "_".join(Htbl['local_upper_quanta'][i].strip().split())
        qqlow = "_".join(Htbl['global_lower_quanta'][i].strip().split()) + "|" +\
               "_".join(Htbl['local_lower_quanta'][i].strip().split())

        fh.write("{:6d}{:>30s}{:>30s}{:11.5f}{:15.8f}{:13.4e}{:15.5f}{:15.5f}{:7.1f}{:7.1f}\n".format(
                  i, qqup, qqlow, waves[i], freqsG[i],  Htbl['a'][i], Eup[i], Elo[i], Htbl['gp'][i], Htbl['gpp'][i]
                ))
    return fh


if __name__ == "__main__":

    mols = ["H2O", "CO2", "13CO2", "CO", "13CO", "C18O", "CH4", "HCN", "NH3", "OH", "C2H2", "C4H2", "SO2", "H2", "HD"]
    basem =["H2O", "CO2", "CO2" , "CO", "CO",   "CO"  , "CH4", "HCN", "NH3", "OH", "C2H2", "C4H2", "SO2", "H2", "H2"]
    isot = [ 1,     1,      2,      1,    2,     3,        1,    1,     1,    1,     1   ,    1  ,   1  ,   1 ,   2 ]

    min_wave = 0.3 # micron
    max_wave = 1000 # micron

    min_vu = 1/(min_wave/1E6)/100.
    max_vu = 1/(max_wave/1E6)/100.

    for mol, bm, iso in zip(mols, basem, isot):
        print("Doing mol: {:}".format(mol))
        Htbl, qdata, M, G = get_Hitran_data(bm, iso, min_vu, max_vu)
        fh = open("./HITRANdata/data_Hitran_2020_{:}.par".format(mol), 'w')
        fh.write("# HITRAN 2020 {:}; id:{:}; iso:{:};gid:{:}\n".format(mol, M, iso, G))
        fh.write("# Downloaded from the Hitran website\n")
        fh.write("# {:s}\n".format(str(datetime.date.today())))
        fh = write_partition_function(fh, qdata)

        fh = write_line_data(fh, Htbl)
        print("Mol: {:}, done".format(mol))

