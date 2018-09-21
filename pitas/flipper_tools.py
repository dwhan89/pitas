#-
# flipper_tools.py
#-
#
import numpy as np


def read_bin_edges(bin_file):
    print "[flipper_tools] loading %s" %bin_file
    from flipper import fftTools
    (lower, upper, center) = fftTools.readBinningFile(bin_file)
    return np.concatenate((lower[0:1], upper))

