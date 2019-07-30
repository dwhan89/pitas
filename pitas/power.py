#
# power.py
#- 
#
import pitas
import pitas.modecoupling as mcm
import os, numpy as np
import warnings
from pixell import enmap

class PITAS(object):
    def __init__(self, mcm_identifier, window_scalar, window_pol, bin_edges, lmax=None, transfer=None, output_dir=None, overwrite=False):
        self.mcm_identifier = mcm_identifier
        self.window_scalar    = window_scalar
        self.window_pol     = window_pol

        binner              = mcm.PITAS_BINNER(bin_edges, lmax)
        self.binner         = binner
        self.lmax           = binner.lmax
        self.bin_edges      = binner.bin_edges
        self.bin_center     = binner.bin_center
        self.nbin           = len(self.bin_center)        

        self.transfer       = transfer
        
        ret = get_mcm_inv(self.mcm_identifier, self.window_scalar, self.window_pol, self.bin_edges,\
            output_dir, lmax, transfer, overwrite)
        
        self.mcm_cltt_inv     = ret[0].copy(); self.mcm_cltp_inv     = ret[1].copy()
        self.mcm_clpp_inv     = ret[2].copy(); self.mcm_dltt_inv     = ret[3].copy()
        self.mcm_dltp_inv     = ret[4].copy(); self.mcm_dlpp_inv     = ret[5].copy()
         
        del ret

        ret                 = get_bbl(self.mcm_identifier, self.window_scalar, self.window_pol, bin_edges,\
                             output_dir, lmax, transfer, False)

        self.bbl_cltt = ret[0].copy();  self.bbl_cltp = ret[1].copy()
        self.bbl_clpp = ret[2].copy();  self.bbl_dltt = ret[3].copy()
        self.bbl_dltp = ret[4].copy();  self.bbl_dlpp = ret[5].copy()
        del ret

    def bin_theory_scalarxscalar(self, l_th, cl_th, ret_dl=False):
        ''' return dl=l(l+1)cl/(2pi) if ret_dl == True '''
        assert(l_th[0] == 2)
        bbl = self.bbl_cltt if ret_dl == False else self.bbl_dltt
        if ret_dl: cl_th = l_th*(l_th+1.)/(2.*np.pi)*cl_th
        spec_bin = np.dot(bbl, cl_th[:self.lmax])

        return (self.bin_center.copy(), spec_bin)

    def bin_theory_scalarxvector(self, l_th, cl_th, ret_dl=False):
        ''' return dl=l(l+1)cl/(2pi) if ret_dl == True '''
        assert(l_th[0] == 2)
        bbl = self.bbl_cltp if ret_dl == False else self.bbl_dltp
        if ret_dl: cl_th = l_th*(l_th+1.)/(2.*np.pi)*cl_th
        spec_bin = np.dot(bbl, cl_th[:self.lmax])

        return (self.bin_center.copy(), spec_bin)

    def bin_theory_pureeb(self, l_th, clee_th, cleb_th, clbb_th, ret_dl=False):
        ''' return dl=l(l+1)cl/(2pi) if ret_dl == True '''
        assert(l_th[0] == 2)
        bbl = self.bbl_clpp if ret_dl == False else self.bbl_dlpp

        if ret_dl:
            clee_th = l_th*(l_th+1.)/(2.*np.pi)*clee_th
            cleb_th = l_th*(l_th+1.)/(2.*np.pi)*cleb_th
            clbb_th = l_th*(l_th+1.)/(2.*np.pi)*clbb_th
        else: pass

        clpp_th  = np.concatenate([clee_th[:self.lmax],cleb_th[:self.lmax],clbb_th[:self.lmax]])
        spec_bin = np.dot(bbl , clpp_th)

        nbin     = self.nbin
        eebin_th, ebbin_th, bbbin_th = (spec_bin[:nbin], spec_bin[nbin:2*nbin], spec_bin[2*nbin:3*nbin])

        return (self.bin_center.copy(), eebin_th, ebbin_th, bbbin_th) 


    def get_power(self, emap1, emap2=None, polcomb='SS', pure_eb=True, ret_dl=False):
        ''' return dl=l(l+1)cl/(2pi) if ret_dl == True '''
        # take and bin raw spectra
        polcomb = ''.join(sorted(polcomb.upper()))

        lbin, clbin = (None, None)
        if polcomb in ['SS', 'TT', 'KK', 'QQ', 'UU']:
            lbin, spec_bin = self.get_power_scalarXscalar(emap1, emap2, ret_dl=ret_dl)
        elif polcomb in ['PT', 'ET', 'BT']:
            lbin, spec_bin = self.get_power_scalarXvector(emap1, emap2, ret_dl=ret_dl)
        elif polcomb in ['EE', 'BB', 'BE']:
            # vector x vector
            if pure_eb:
                warning.warn('[PITAS/POWER] map1 should be emap, map2 should be bmap')
                idx2pp = {'EE': 1, 'EB': 2, 'BB': 3}
                ret    = self.get_power_pureeb(emap1, emap2, ret_dl=ret_dl)
                lbin   = ret[0]
                spec_bin  = ret[idx2pp[polcomb]]
            else:
                lbin, spec_bin = self.get_power_scalarXscalar(emap1, emap2, ret_dl=ret_dl)
        else:
            warning.warn('[PITAS/POWER] polcomb is not specified. assume scalar x scalar]')
            lbin, spec_bin = self.get_power_scalarXscalar(emap1, emap2, ret_dl=ret_dl)
           
        return lbin, spec_bin 

    def get_power_scalarXscalar(self, emap1, emap2, ret_dl=False):
        ''' return dl=l(l+1)cl/(2pi) if ret_dl == True '''
        l, cl        = get_raw_power(emap1, emap2, lmax=self.lmax, normalize=False, ret_dl=ret_dl)
        lbin, clbin  = self.binner.bin(l,cl)
        del l, cl

        mcm_inv = self.mcm_cltt_inv if ret_dl == False else self.mcm_dltt_inv
        spec_bin   = np.dot(mcm_inv, clbin)
        
        return (lbin, spec_bin)

    def get_power_scalarXvector(self, emap1, emap2, ret_dl=False):
        ''' return dl=l(l+1)cl/(2pi) if ret_dl == True '''
        l, cl        = get_raw_power(emap1, emap2, lmax=self.lmax, normalize=False, ret_dl=ret_dl)
        lbin, clbin  = self.binner.bin(l,cl)
        del l, cl

        mcm_inv = self.mcm_cltp_inv if ret_dl == False else self.mcm_dltp_inv
        spec_bin   = np.dot(mcm_inv, clbin)
        
        return (lbin, spec_bin)

    def get_power_pureeb(self, emap1, bmap1, emap2=None, bmap2=None, ret_dl=False):
        ''' return dl=l(l+1)cl/(2pi) if ret_dl == True '''
        l, clee        = get_raw_power(emap1, emap2, lmax=self.lmax, normalize=False, ret_dl=ret_dl)
        lbin, cleebin  = self.binner.bin(l,clee)

        cleb           = None
        if emap2 is None or bmap2 is None:
            l, cleb        = get_raw_power(emap1, bmap1, lmax=self.lmax, normalize=False, ret_dl=ret_dl)
        else:
            l, cleb1       = get_raw_power(emap1, bmap2, lmax=self.lmax, normalize=False, ret_dl=ret_dl)
            l, cleb2       = get_raw_power(emap2, bmap1, lmax=self.lmax, normalize=False, ret_dl=ret_dl)
            cleb           = (cleb1+cleb2)/2. 
            del cleb1, cleb2
        lbin, clebbin  = self.binner.bin(l,cleb)

        l, clbb        = get_raw_power(bmap1, bmap2, lmax=self.lmax, normalize=False, ret_dl=ret_dl)
        lbin, clbbbin  = self.binner.bin(l,clbb)

        clpol   = np.concatenate([cleebin, clebbin, clbbbin])
        mcm_inv = self.mcm_clpp_inv if ret_dl == False else self.mcm_dlpp_inv
        del l, clee, cleb, clbb, cleebin, clebbin, clbbbin

        nbin    = self.nbin
        pol_spec   = np.dot(mcm_inv, clpol)
        
        eebin, ebbin, bbbin = (pol_spec[:nbin], pol_spec[nbin:2*nbin], pol_spec[2*nbin:3*nbin])
        
        return (lbin, eebin, ebbin, bbbin)

class PITAS4FLIPPER(PITAS):
    def __init__(self, mcm_identifier, window_scalar, window_pol, bin_edges, lmax=None, transfer=None, overwrite=False):
        super(PITAS4FLIPPER, self).__init__(mcm_identifier, l2e(window_scalar), l2e(window_pol), bin_edges, lmax=lmax, transfer=transfer, overwrite=overwrite)
        
    #def get_power(self,lmap1, lmap2=None, polcomb='SS', pure_eb=True):
    #    emap2 = None if lmap2 is None else l2e(lmap2)
    #    return  super(PITAS4FLIPPER, self).get_power(l2e(lmap1), emap2, polcomb=polcomb, pure_eb=pure_eb)

    def get_power_scalarXscalar(self, lmap1, lmap2, ret_dl=False):
        emap2 = None if lmap2 is None else l2e(lmap2)
        return  super(PITAS4FLIPPER, self).get_power_scalarXscalar(l2e(lmap1), emap2, ret_dl=ret_dl)

    def get_power_scalarXvector(self, lmap1, lmap2, ret_dl=False):
        emap2 = None if lmap2 is None else l2e(lmap2)
        return  super(PITAS4FLIPPER, self).get_power_scalarXvector(l2e(lmap1), emap2, ret_dl=ret_dl)

    def get_power_pureeb(self, emap1, bmap1, emap2=None, bmap2=None, ret_dl=False):
        eemap2 = None if emap2 is None else l2e(emap2)
        bemap2 = None if bmap2 is None else l2e(bmap2)
        return super(PITAS4FLIPPER, self).get_power_pureeb(l2e(emap1), l2e(bmap1), eemap2, bemap2, ret_dl=ret_dl)

def l2e(lmap):
    ''' convert from flipper liteMap to enmap '''
    return enmap.from_flipper(lmap)


def get_bbl(mcm_identifier, window_scalar=None, window_pol=None, bin_edges=None, output_dir=None, lmax=None, transfer=None, overwrite=False): 
    if output_dir is None: output_dir = pitas.config.get_output_dir()
    if mcm_identifier is not None: mcm_dir = os.path.join(output_dir, mcm_identifier)
    if pitas.mpi.rank == 0: print("[get_bbl] mcm directory: %s" %mcm_dir)

    bbl_cltt, bbl_cltp, bbl_clpp = (None, None, None)
    def load_bbl(key):
        file_name = 'curved_full_BBL_%s.dat' %key
        file_name = os.path.join(mcm_dir, file_name)
        
        print("trying to load %s" %file_name)
        return np.loadtxt(file_name)

    try:
        assert(not overwrite) 
        bbl_cltt = load_bbl('CLTT'); bbl_cltp = load_bbl('CLTP')
        bbl_clpp = load_bbl('CLPP'); bbl_dltt = load_bbl('DLTT')
        bbl_dltp = load_bbl('DLTP'); bbl_dlpp = load_bbl('DLPP')
    except:
        if pitas.mpi.rank == 0: 
            print("failed to load mcm. calculating mcm")
            pitas.util.check_None(window_scalar, window_pol, bin_edges, mcm_dir)
            mcm.generate_mcm(window_scalar, window_pol, bin_edges, mcm_dir, lmax=lmax, transfer=transfer)
            print("finish calculating mcm")
        else: pass
        pitas.mpi.barrier()

        bbl_cltt = load_bbl('CLTT'); bbl_cltp = load_bbl('CLTP')
        bbl_clpp = load_bbl('CLPP'); bbl_dltt = load_bbl('DLTT')
        bbl_dltp = load_bbl('DLTP'); bbl_dlpp = load_bbl('DLPP')
    return (bbl_cltt, bbl_cltp, bbl_clpp, bbl_dltt, bbl_dltp, bbl_dlpp)

def get_mcm_inv(mcm_identifier, window_scalar=None, window_pol=None, bin_edges=None, output_dir=None, lmax=None, transfer=None, overwrite=False):
    if output_dir is None: output_dir = pitas.config.get_output_dir()
    if mcm_identifier is not None: mcm_dir = os.path.join(output_dir, mcm_identifier)
    if pitas.mpi.rank == 0: print("mcm directory: %s" %mcm_dir)

    mbb_cltt_inv, mbb_cltp_inv, mbb_clpp_inv = (None, None, None)
    def load_mbb_inv(key):
        file_name = 'curved_full_%s.dat' %key
        file_name = os.path.join(mcm_dir, file_name)
        
        print("trying to load %s" %file_name)
        return np.loadtxt(file_name)

    try:
        assert(not overwrite)
        mbb_cltt_inv = load_mbb_inv('CLTT_inv'); mbb_cltp_inv = load_mbb_inv('CLTP_inv')
        mbb_clpp_inv = load_mbb_inv('CLPP_inv')

        mbb_dltt_inv = load_mbb_inv('DLTT_inv'); mbb_dltp_inv = load_mbb_inv('DLTP_inv')
        mbb_dlpp_inv = load_mbb_inv('DLPP_inv')
    except:
        if pitas.mpi.rank == 0: 
            print("failed to load mcm. calculating mcm")
            pitas.util.check_None(window_scalar, window_pol, bin_edges, mcm_dir)
            mcm.generate_mcm(window_scalar, window_pol, bin_edges, mcm_dir, lmax=lmax, transfer=transfer)
            print("finish calculating mcm")
        else: pass
        pitas.mpi.barrier()

        mbb_cltt_inv, mbb_cltp_inv, mbb_clpp_inv, mbb_dltt_inv, mbb_dltp_inv, mbb_dlpp_inv = \
                get_mcm_inv(mcm_identifier, overwrite=False)

    return (mbb_cltt_inv, mbb_cltp_inv, mbb_clpp_inv, mbb_dltt_inv, mbb_dltp_inv, mbb_dlpp_inv)

def get_raw_power(emap1, emap2=None, lmax=None, normalize=True, ret_dl=False):
    l, cl = pitas.util.get_spectra(emap1, emap2, lmax=lmax)
    if ret_dl: cl = l*(l+1.)/(2*np.pi)*cl

    if normalize:
        fsky  = pitas.util.get_fsky(emap1)
        cl    /= fsky
    else: pass

    return (l, cl)
