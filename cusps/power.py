#
# power.py
#- 
#
import cusps
import cusps.modecoupling as mcm
import os, numpy as np
from orphics import stats
import warnings

class CUSPS(object):
    def __init__(self, mcm_identifier, window_scalar, window_pol, bin_edges, lmax=None, transfer=None, overwrite=False):
        self.mcm_identifier = mcm_identifier
        self.window_scalar    = window_scalar
        self.window_pol     = window_pol

        binner              = mcm.CUSPS_BINNER(bin_edges, lmax)
        self.binner         = binner
        self.lmax           = binner.lmax
        self.bin_edges      = binner.bin_edges
        self.bin_center     = binner.bin_center
        self.nbin           = len(self.bin_center)        

        self.transfer       = transfer
        
        ret = get_mcm_inv(self.mcm_identifier, self.window_scalar, self.window_pol, self.bin_edges,\
            None, lmax, transfer, overwrite)
        
        self.mcm_tt_inv     = ret[0].copy()
        self.mcm_tp_inv     = ret[1].copy()
        self.mcm_pp_inv     = ret[2].copy()
        del ret

        ret                 = get_bbl(self.mcm_identifier, self.window_scalar, self.window_pol, bin_edges,\
                             None, lmax, transfer, False)

        self.bbl_tt         = ret[0].copy()
        self.bbl_tp         = ret[1].copy()
        self.bbl_pp         = ret[2].copy()
        del ret

        mcm_invs            = {} 
        mcm_invs[0]         = self.mcm_tt_inv
        mcm_invs[1]         = self.mcm_tp_inv
        mcm_invs[2]         = self.mcm_pp_inv

        self.mcm_invs       = mcm_invs

    def bin_theory_scalarxscalar(self, l_th, cl_th):
        assert(l_th[0] == 2)
        clbin_th = np.dot(self.bbl_tt, cl_th[:self.lmax])

        return (self.bin_center.copy(), clbin_th)

    def bin_theory_scalarxvector(self, l_th, cl_th):
        assert(l_th[0] == 2)
        clbin_th = np.dot(self.bbl_tp, cl_th[:self.lmax])

        return (self.bin_center.copy(), clbin_th)

    def bin_theory_pureeb(self, l_th, clee_th, cleb_th, clbb_th):
        assert(l_th[0] == 2)

        clpp_th  = np.concatenate([clee_th[:self.lmax],cleb_th[:self.lmax],clbb_th[:self.lmax]])
        clbin_th = np.dot(self.bbl_pp, clpp_th)

        nbin     = self.nbin
        cleebin_th, clebbin_th, clbbbin_th = (clbin_th[:nbin], clbin_th[nbin:2*nbin], clbin_th[2*nbin:3*nbin])

        return (self.bin_center.copy(), cleebin_th, clebbin_th, clbbbin_th) 


    def get_power(self, emap1, emap2=None, polcomb='SS', pure_eb=True):
        # take and bin raw spectra
        polcomb = ''.join(sorted(polcomb.upper()))

        lbin, clbin = (None, None)
        if polcomb in ['SS', 'TT', 'KK', 'QQ', 'UU']:
            lbin, clbin = self.get_power_scalarXscalar(emap1, emap2)
        elif polcomb in ['PT', 'ET', 'BT']:
            lbin, clbin = self.get_power_scalarXvector(emap1, emap2)
        elif polcomb in ['EE', 'BB', 'BE']:
            # vector x vector
            if pure_eb:
                warning.warn('[CUSPS/POWER] map1 should be emap, map2 should be bmap')
                idx2pp = {'EE': 1, 'EB': 2, 'BB': 3}
                ret    = self.get_power_pureeb(emap1, emap2)
                lbin   = ret[0]
                clbin  = ret[idx2pp[polcomb]]
            else:
                lbin, clbin = self.get_power_scalarXscalar(emap1, emap2)
        else:
            warning.warn('[CUSPS/POWER] polcomb is not specified. assume scalar x scalar]')
            lbin, clbin = self.get_power_scalarXscalar(emap1, emap2)
           
        return lbin, clbin 

    def get_power_scalarXscalar(self, emap1, emap2):
        l, cl        = get_raw_power(emap1, emap2, lmax=self.lmax, normalize=False)
        lbin, clbin  = self.binner.bin(l,cl)
        del l, cl

        mcm_inv = self.mcm_invs[0]
        clbin   = np.dot(mcm_inv, clbin)
        
        return (lbin, clbin)

    def get_power_scalarXvector(self, emap1, emap2):
        l, cl        = get_raw_power(emap1, emap2, lmax=self.lmax, normalize=False)
        lbin, clbin  = self.binner.bin(l,cl)
        del l, cl

        mcm_inv = self.mcm_invs[1]
        clbin   = np.dot(mcm_inv, clbin)
        
        return (lbin, clbin)


    def get_power_pureeb(self, emap, bmap):
        l, clee        = get_raw_power(emap, emap, lmax=self.lmax, normalize=False)
        lbin, cleebin  = self.binner.bin(l,clee)

        l, cleb        = get_raw_power(emap, bmap, lmax=self.lmax, normalize=False)
        lbin, clebbin  = self.binner.bin(l,cleb)

        l, clbb        = get_raw_power(bmap, bmap, lmax=self.lmax, normalize=False)
        lbin, clbbbin  = self.binner.bin(l,clbb)

        clpol   = np.concatenate([cleebin, clebbin, clbbbin])
        mcm_inv = self.mcm_invs[2]
        del l, clee, cleb, clbb, cleebin, clebbin, clbbbin

        nbin    = self.nbin
        clpol   = np.dot(mcm_inv, clpol)
        
        clee, cleb, clbb = (clpol[:nbin], clpol[nbin:2*nbin], clpol[2*nbin:3*nbin])
        
        return (lbin, clee, cleb, clbb)

def get_bbl(mcm_identifier, window_scalar=None, window_pol=None, bin_edges=None, output_dir=None, lmax=None, transfer=None, overwrite=False): 
    if output_dir is None: output_dir = cusps.config.get_output_dir()
    if mcm_identifier is not None: mcm_dir = os.path.join(output_dir, mcm_identifier)
    if cusps.mpi.rank == 0: print "[get_bbl] mcm directory: %s" %mcm_dir

    bbl_tt, bbl_tp, bbl_pp = (None, None, None)
    def load_bbl(key):
        file_name = 'curved_full_BBL_%s.dat' %key
        file_name = os.path.join(mcm_dir, file_name)
        
        print "trying to load %s" %file_name
        return np.loadtxt(file_name)

    try:
        assert(not overwrite) 
        bbl_tt = load_bbl('TT') 
        bbl_tp = load_bbl('TP')
        bbl_pp = load_bbl('PP')
    except:
        if cusps.mpi.rank == 0: 
            print "failed to load mcm. calculating mcm"
            cusps.util.check_None(window_scalar, window_pol, bin_edges, mcm_dir)
            mcm.generate_mcm(window_scalar, window_pol, bin_edges, mcm_dir, lmax=lmax, transfer=transfer)
            print "finish calculating mcm"
        else: pass
        cusps.mpi.barrier()

        bbl_tt = load_bbl('TT') 
        bbl_tp = load_bbl('TP')
        bbl_pp = load_bbl('PP')
    return (bbl_tt, bbl_tp, bbl_pp)

def get_mcm_inv(mcm_identifier, window_scalar=None, window_pol=None, bin_edges=None, output_dir=None, lmax=None, transfer=None, overwrite=False):
    if output_dir is None: output_dir = cusps.config.get_output_dir()
    if mcm_identifier is not None: mcm_dir = os.path.join(output_dir, mcm_identifier)
    if cusps.mpi.rank == 0: print "mcm directory: %s" %mcm_dir

    mbb_tt_inv, mbb_tp_inv, mbb_pp_inv = (None, None, None)
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
            cusps.util.check_None(window_scalar, window_pol, bin_edges, mcm_dir)
            mcm.generate_mcm(window_scalar, window_pol, bin_edges, mcm_dir, lmax=lmax, transfer=transfer)
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
