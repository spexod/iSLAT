import scipy.constants as con

def write_line_data(fh, Htbl):
    """
    Write line data to a file.

    Parameters
    ----------
    fh : file handle
        The file handle to write the data to.
    Htbl : DataFrame
        A pandas DataFrame containing HITRAN line data.

    Returns
    -------
    fh : file handle
        The file handle after writing the data.
    """
    numlines = len(Htbl['molec_id'])

    fh.write("Number of lines\n")
    fh.write(str(numlines) + '\n')

    fh.write("#    Nr                        Lev_up                       Lev_low   Lambda    Frequency")
    fh.write("       Einstein-A     E_up           E_low        g_up   g_low\n")
    fh.write("#                                                                    [micron]  [GHz]    ")
    fh.write("       [s**-1]        [K]            [K]        \n")

    freqs = Htbl['nu'] * 100. * con.c
    freqsG = freqs / 1E9
    waves = con.c / freqs * 1E6
    Elo = con.h * con.c * Htbl['elower'] * 100 / con.k
    Eup = con.h * con.c * (Htbl['elower'] + Htbl['nu']) * 100 / con.k

    for i in range(numlines):
        qqup = "_".join(Htbl['global_upper_quanta'][i].strip().split()) + "|" +\
               "_".join(Htbl['local_upper_quanta'][i].strip().split())
        qqlow = "_".join(Htbl['global_lower_quanta'][i].strip().split()) + "|" +\
                "_".join(Htbl['local_lower_quanta'][i].strip().split())

        fh.write("{:6d}{:>30s}{:>30s}{:11.5f}{:15.8f}{:13.4e}{:15.5f}{:15.5f}{:7.1f}{:7.1f}\n".format(
            i, qqup, qqlow, waves[i], freqsG[i], Htbl['a'][i], Eup[i], Elo[i], Htbl['gp'][i], Htbl['gpp'][i]
        ))

    return fh