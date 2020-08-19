#-
# flipper_tools.py
#-
#
import numpy as np
import os

def read_bin_edges(bin_file, skiprows=0):
    print("[flipper_tools] loading %s" %bin_file)
    assert(os.path.exists(bin_file))
    (lower, upper, center) = np.loadtxt(bin_file, skiprows=skiprows, unpack=True)
    return np.concatenate((lower[0:1], upper))

