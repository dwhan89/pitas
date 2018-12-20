#--
# act_analysis.py
#--
# act specific mask and beam 
# written by DW
#
#

import pitas
import os, numpy as np


def get_beam_transfer(beam_file):
    '''
        input  : path to act beam file
        output : tuple (ell, f_ell)
    '''
    assert(os.path.exists(beam_file))
    print("loading beam file %s" %(beam_file))

    ell, f_ell = np.transpose(np.loadtxt(beam_file))[0:2,:]
    return  (ell, f_ell)



