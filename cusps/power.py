#
# power.py
#- 
#

from enlib import enmap, curvedsky
import healpy
import numpy as np

def get_raw_power(imap1, imap2=None, lmax=5000):
    # healpy expects complex input
    alm1 = curvedsky.map2alm(imap1, lmax=lmax).astype(np.complex128)
    alm2 = alm1 if imap2 is None else curvedsky.map2alm(imap2, lmax=lmax).astype(np.complex128)

    cl  = healpy.alm2cl(alm1, alm2, lmax=lmax)
    l   = np.arange(len(cl))
    cl  *= (2.*l+1.)

    return (l, cl)
