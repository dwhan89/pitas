#
# power.py
#- 
#
import cusps
import cusps.modecoupling as mcm
import os, numpy as np
from orphics import stats

class CUSPS(object):
    def __init__(self, mcm_identifier, window_temp, window_pol, bin_edges, lmax=None, transfer=None, overwrite=False):
        self.mcm_identifier = mcm_identifier
        self.window_temp   = window_temp
        self.window_pol    = window_pol

        lmax               = int(np.max(bin_edges) if lmax is None else lmax) 
        self.lmax          = lmax

        # binner set-up 
        bin_edges       = bin_edges.astype(np.int)
        if bin_edges[0] < 2: bin_edges[0] = 2
        self.bin_edges  = bin_edges[np.where(bin_edges <= lmax)]
        self.bin_center = (bin_edges[1:] + bin_edges[:-1])/2.
        self.binner     = stats.bin1D(bin_edges)
        

        self.transfer      = transfer
        
        ret = get_mcm_inv(self.mcm_identifier, self.window_temp, self.window_pol, self.bin_edges,\
            None, lmax, transfer, overwrite)
        
        self.mcm_tt_inv    = ret[0].copy()
        self.mcm_tp_inv    = ret[1].copy()
        self.mcm_pp_inv    = ret[2].copy()
        del ret

        self.bbl           = get_bbl(self.mcm_identifier, self.window_temp, self.window_pol, self.bin_edges,\
                            None, lmax, transfer, False)

        mcm_invs           = {} 
        mcm_invs[0]        = self.mcm_tt_inv
        mcm_invs[1]        = self.mcm_tp_inv
        mcm_invs[2]        = self.mcm_pp_inv

        self.mcm_invs      = mcm_invs

    def get_cached_mcm_inv(self, polcomb='00', pure_eb=True):
        polcomb = ''.join(sorted(polcomb.upper()))
        assert(polcomb in ['00', 'TT', 'ET', 'BT', 'EE', 'BE', 'BB', 'PP'])
        # 00 = scalar field x scalar field
    
        mcm_dicts = {'00': 0, 'TT': 0, 'ET': 1, 'BT': 1, 'EE': 2, 'BE': 2, 'BB': 2, 'PP': 2}
        # watch out of indexing
        
        mcm_idx = 0 if not pure_eb else mcm_dicts[polcomb]
        
        mcm_inv = self.mcm_invs[mcm_idx]

        '''
        if mcm_idx == 2:
            pol_dicts = {'EE': 0, 'BE': 1, 'BB': 2}
            pol_idx   = pol_dicts[polcomb]
            nbin      = mcm_inv.shape[0]/3
            mcm_inv   = mcm_inv[:, nbin*pol_idx: nbin*(pol_idx+1)]
        else: pass
        '''

        return mcm_inv
    
    def bin_theory(self, l_th, cl_th):
        assert(l_th[0] == 0)
        clbin_th = np.dot(self.bbl, cl_th[:self.lmax])

        return (self.bin_center.copy(), clbin_th)

    def get_power(self, emap1, emap2=None, polcomb='00', pure_eb=True):
        #
        # This function most likely works, but very clunky.
        #

        # take and bin raw spectra
        assert(polcomb in ['EE', 'EB', 'BB', 'PP'] or pure_eb) # pol part is a bit broken now ...
        l, cl        = get_raw_power(emap1, emap2, lmax=self.lmax, normalize=False)
        lbin, clbin  = self.binner.binned(l,cl)

        del l, cl
        mcm_inv = self.get_cached_mcm_inv(polcomb, pure_eb)
        clbin   = np.dot(mcm_inv, clbin)
        
        return (lbin, clbin)

    def get_pureeb_power(self, emap, bmap):
        l, clee        = get_raw_power(emap, emap, lmax=self.lmax, normalize=False)
        lbin, cleebin  = self.binner.binned(l,clee)

        l, cleb       = get_raw_power(emap, bmap, lmax=self.lmax, normalize=False)
        lbin, clebbin  = self.binner.binned(l,cleb)

        l, clbb        = get_raw_power(bmap, bmap, lmax=self.lmax, normalize=False)
        lbin, clbbbin  = self.binner.binned(l,clbb)

        clpol   = np.concatenate([cleebin, clebbin, clbbbin])
        mcm_inv = self.get_cached_mcm_inv(polcomb='PP', pure_eb=True)
        del l, clee, cleb, clbb, cleebin, clebbin, clbbbin

        nbin    = len(lbin)
        clpol   = np.dot(mcm_inv, clpol)
        
        clee, cleb, clbb = (clpol[:nbin], clpol[nbin:2*nbin], clpol[2*nbin:3*nbin])
        
        return (lbin, clee, cleb, clbb)

'''
def get_power(emap1, mcm_identifier, window_temp, window_pol, bin_edges, polcomb='00', lmax=5000, transfer=None, emap2=None, overwrite=False, pure_eb=True):
    #
    # This function most likely works, but very clunky.
    #

    polcomb = ''.join(sorted(polcomb.upper()))
    assert(polcomb in ['00', 'TT', 'ET', 'BT', 'EE', 'BE', 'BB'])
    # 00 = scalar field x scalar field


    mcm_dicts = {'00': 0, 'TT': 0, 'ET': 1, 'BT': 1, 'EE': 2, 'BE': 2, 'BB': 2}
    # watch out of indexing
    
    mcm_idx = 0 if not pure_eb else mcm_dicts[polcomb]
    
    lmax      = int(np.max(bin_edges) if lmax is None else lmax) 
    bin_edges = bin_edges.astype(np.int)
    if bin_edges[0] < 2: bin_edges[0] = 2
    bin_edges = bin_edges[np.where(bin_edges <= lmax)]
    binner    = stats.bin1D(bin_edges)
    
    if mcm_idx <= 2:
        # binner set-up 

        # take and bin raw spectra
        l, cl        = get_raw_power(emap1, emap2, lmax=lmax, normalize=False)
        lbin, clbin  = binner.binned(l,cl)

        del l, cl
        
        mcm_inv = list(get_mcm_inv(mcm_identifier, window_temp, window_pol, \
                bin_edges, None, lmax=lmax, transfer=transfer, overwrite=overwrite))[mcm_idx]
     
        clbin   = np.dot(mcm_inv, clbin)
    elif mcm_idx == 2:
        assert(False) ## pp part is not ready yet
        pol_dicts = {'EE': 0, 'BE': 1, 'BB': 2}
        pol_idx   = pol_dicts[polcomb]
        nbin      = mcm_inv.shape[0]/3
        mcm_inv   = mcm_inv[:, nbin*pol_idx: nbin*(pol_idx+1)]
    else: pass


    return (lbin, clbin)
'''

def get_bbl(mcm_identifier, window_temp=None, window_pol=None, bin_edges=None, output_dir=None, lmax=None, transfer=None, overwrite=False): 
    if output_dir is None: output_dir = cusps.config.get_output_dir()
    if mcm_identifier is not None: mcm_dir = os.path.join(output_dir, mcm_identifier)
    if cusps.mpi.rank == 0: print "[get_bbl] mcm directory: %s" %mcm_dir

    file_name = os.path.join(mcm_dir, 'curved_full_BBL.dat')
    bbl = None
    try:
        assert(not overwrite) 
        print "trying to load %s" %file_name
        bbl = np.loadtxt(file_name)
    except:
        if cusps.mpi.rank == 0: 
            print "failed to load mcm. calculating mcm"
            cusps.util.check_None(window_temp, window_pol, bin_edges, mcm_dir)
            mcm.generate_mcm(window_temp, window_pol, bin_edges, mcm_dir, lmax=lmax, transfer=transfer)
            print "finish calculating mcm"
        else: pass
        cusps.mpi.barrier()

        bbl = np.loadtxt(file_name)
    return bbl

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
            print "failed to load mcm. calculating mcm"
            cusps.util.check_None(window_temp, window_pol, bin_edges, mcm_dir)
            mcm.generate_mcm(window_temp, window_pol, bin_edges, mcm_dir, lmax=lmax, transfer=transfer)
            print "finish calculating mcm"
        else: pass
        cusps.mpi.barrier()

        mbb_tt_inv, mbb_tp_inv, mbb_pp_inv = get_mcm_inv(mcm_identifier, overwrite=False)

    return (mbb_tt_inv, mbb_tp_inv, mbb_pp_inv)

def get_raw_power(emap1, emap2=None, lmax=None, normalize=True):
    l, cl = cusps.util.get_spectra(emap1, emap2, lmax=lmax)

    if normalize:
        fsky  = cusps.util.get_fsky(emap1)
        cl    /= fsky
    else: pass

    return (l, cl)
