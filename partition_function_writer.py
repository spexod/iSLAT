def write_partition_function(fh, qdata):
    """
    Write partition function data to a file.

    Parameters
    ----------
    fh : file handle
        The file handle to write the data to.
    qdata : DataFrame
        A pandas DataFrame containing temperature and partition function data.

    Returns
    -------
    fh : file handle
        The file handle after writing the data.
    """
    fh.write("# Number of Partition function entries\n")
    fh.write("{:d}\n".format(len(qdata['temp'])))

    fh.write("# Temperature    Q(T)\n")
    fh.write("# [K]\n")

    for t, q in zip(qdata['temp'], qdata['q']):
        fh.write("{:>6.1f}  {:>15.6f}\n".format(t, q))

    return fh
