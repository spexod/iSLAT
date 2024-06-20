from astroquery import hitran
import pandas as pd
import urllib.request
import ssl
from COMPONENTS.hitran_utils import get_molecule_identifier
from COMPONENTS.global_identifier import get_global_identifier
from astropy import units as un

context = ssl.create_default_context()
context.check_hostname = False
context.verify_mode = ssl.CERT_NONE

def get_Hitran_data(Molecule_name, isotopologue_number, min_vu, max_vu):
    M = get_molecule_identifier(Molecule_name)

    Htbl = hitran.Hitran.query_lines(molecule_number=M, 
                                     isotopologue_number=isotopologue_number,
                                     min_frequency=min_vu/un.cm, 
                                     max_frequency=max_vu/un.cm)

    G = get_global_identifier(Molecule_name, isotopologue_number=isotopologue_number)

    qurl = 'https://hitran.org/data/Q/' + 'q' + str(G) + '.txt'
    handle = urllib.request.urlopen(qurl, context=context)
    qdata = pd.read_csv(handle, sep=' ', skipinitialspace=True, names=['temp', 'q'], header=None)

    return Htbl, qdata, M, G
