#
# power.py
#- 
#
import cusps
from cusps.mcm_core import mcm_core
import os, numpy as np
from orphics import stats

def get_mcm_inv(mcm_identifier, window_temp=None, window_pol=None, bin_edges=None, output_dir=None, lmax=None, transfer=None, overwrite=False):
    if output_dir is None: output_dir = cusps.config.get_output_dir()
    if mcm_identifier is not None: mcm_dir = os.path.join(output_dir, mcm_identifier)
    if cusps.mpi.rank == 0: print "mcm directory: %s" %mcm_dir

    mbb_tt_inv = mbb_tp_inv = mbb_pp_inv = None
    def load_mbb_inv(key):
        file_name = 'curved_full_%s.dat' %key
        file_name = os.path.join(mcm_dir, file_name)
        
        print "trying to load %s" %file_name
        return np.loadtxt(file_name)

    try:
        assert(not overwrite)
        mbb_tt_inv = load_mbb_inv('TT_inv')
        mbb_tp_inv = load_mbb_inv('TP_inv')
        mbb_pp_inv = load_mbb_inv('PP_inv')
    except:
        if cusps.mpi.rank == 0: 
            print "failed to load. calculating mcm"
            cusps.util.check_None(window_temp, window_pol, bin_edges, mcm_dir)
            mcm_core.generate_mcm(window_temp, window_pol, bin_edges, mcm_dir, lmax=lmax, transfer=transfer)
        else: pass
        cusps.mpi.barrier()

        mbb_tt_inv, mbb_tp_inv, mbb_pp_inv = get_mcm_inv(mcm_identifier, overwrite=False)

    return (mbb_tt_inv, mbb_tp_inv, mbb_pp_inv)




def get_power(emap1, mcm_identifier, window_temp, window_pol, bin_edges, polcomb='00', lmax=5000, transfer=None, emap2=None, overwrite=False):
    #
    # This function most likely works, but very clunky.
    #

    polcomb = ''.join(sorted(polcomb.upper()))
    assert(polcomb in ['00', 'TT', 'ET', 'BT', 'EE', 'BE', 'BB'])
    # 00 = scalar field x scalar field

    # binner set-up 
    lmax      = int(np.max(bin_edges) if lmax is None else lmax) 
    bin_edges = bin_edges.astype(np.int)
    if bin_edges[0] < 2: bin_edges[0] = 2
    bin_edges = bin_edges[np.where(bin_edges <= lmax)]
    binner    = stats.bin1D(bin_edges)

    # take and bin raw spectra
    l, cl        = get_raw_power(emap1, emap2, lmax=lmax, normalize=True)
    lbin, clbin  = binner.binned(l,cl)

    del l, cl

    mcm_dicts = {'00': 0, 'TT': 0, 'ET': 1, 'BT': 1, 'EE': 2, 'BE': 2, 'BB': 2}
    # watch out of indexing
    mcm_inv = list(get_mcm_inv(mcm_identifier, window_temp, window_pol, \
            bin_edges, None, lmax=lmax, transfer=transfer, overwrite=overwrite))[mcm_dicts[polcomb]]
   
    clbin   = np.dot(mcm_inv, clbin)
    
    return (lbin, clbin)

def get_raw_power(emap1, emap2=None, lmax=None, normalize=True):
    l, cl = cusps.util.get_spectra(emap1, emap2, lmax=lmax)

    if normalize:
        fsky  = cusps.util.get_fsky(emap1)
        cl    /= fsky
    else: pass

    return (l, cl)
