#!/usr/bin/env python
from pixell import enmap,curvedsky
from pitas.util import get_spectra
from pitas.mcm_core import mcm_core
import pitas, numpy as np, os 
import warnings

def generate_mcm(window_temp, window_pol, bin_edges, mcm_dir=None, lmax=None, transfer=None, save_raw=False):
    # expect enmap
    warnings.warn('[pitas/modecoupling] generating mcm')

    if mcm_dir is None: mcm_dir = pitas.config.get_output_dir()
    pitas.pitas_io.create_dir(mcm_dir)

    modlmap   = window_temp.modlmap()
    binner    = PITAS_BINNER(bin_edges, lmax)
    lmax      = binner.lmax
    bin_sizes = binner.bin_sizes   
    nbin      = binner.nbin    
    bin_lower = binner.bin_lower
    bin_upper = binner.bin_upper

    def get_windowspec(window1, window2=None, lmax=lmax):
        l, cl = get_spectra(window1, window2, lmax=lmax)
        cl    *= (2.*l+1.)
        return l, cl
    
    l, cl_temp   = get_windowspec(window_temp, lmax=lmax)
    l, cl_cross  = get_windowspec(window_temp, window_pol, lmax=lmax)
    l, cl_pol    = get_windowspec(window_pol, lmax=lmax)

    # load transfer function
    f_tran = None
    if transfer is not None:
        l_tran, f_tran = transfer
        l_tran, f_tran = l_tran[:lmax],f_tran[:lmax]
        transfer       = f_tran **2.
    else: 
        transfer       = np.ones(len(l))
        

    # full sized mcm matrices
    mcm_tt, mcm_tp, mcm_pp_diag, mcm_pp_offdiag = np.zeros((4, lmax, lmax))
    # binned mcm matrices for cls
    mbb_cltt, mbb_cltp, mbb_clpp_diag, mbb_clpp_offdiag = np.zeros((4, nbin, nbin))
    bbl_cltt, bbl_cltp, bbl_clpp_diag, bbl_clpp_offdiag = np.zeros((4, nbin, lmax))

    # binned mcm matrices for dls
    mbb_dltt, mbb_dltp, mbb_dlpp_diag, mbb_dlpp_offdiag = np.zeros((4, nbin, nbin))
    bbl_dltt, bbl_dltp, bbl_dlpp_diag, bbl_dlpp_offdiag = np.zeros((4, nbin, lmax))
    
    mcm_core.calc_mcm(cl_temp, cl_cross, cl_pol, transfer, mcm_tt.T, mcm_tp.T, mcm_pp_diag.T, mcm_pp_offdiag.T)

    # bin mcm
    def bin_mcm(mcm_full, mcm_binned):
        mcm_core.bin_mcm(mcm_full.T, bin_lower, bin_upper, bin_sizes.T, mcm_binned.T)

    # combine pol data
    def assamble_mat_pp(mat_pp_diag, mat_pp_offdiag):
        nbiny, nbinx = mat_pp_diag.shape # pp matrix is always nxn matrix
        mat_pp = np.zeros((3*nbiny, 3*nbinx))
        mat_pp[:nbiny, :nbinx]          = mat_pp[2*nbiny:3*nbiny, 2*nbinx:3*nbinx] = mat_pp_diag
        mat_pp[2*nbiny:3*nbiny, :nbinx] = mat_pp[:nbiny, 2*nbinx:3*nbinx]          = mat_pp_offdiag
        mat_pp[nbiny:2*nbiny, nbinx:2*nbinx]                                       = mat_pp_diag-mat_pp_offdiag  
 
        return mat_pp

    def save_matrix(key, mbb):
        file_name = 'curved_full_%s.dat' %key
        file_name = os.path.join(mcm_dir, file_name)
        np.savetxt(file_name, mbb)
    
    def process_mcm_process(mcm_tt, mcm_tp, mcm_pp_diag, mcm_pp_offdiag, \
            mbb_tt, mbb_tp, mbb_pp_diag, mbb_pp_offdiag, \
            bbl_tt, bbl_tp, bbl_pp_diag, bbl_pp_offdiag):

        bin_mcm(mcm_tt, mbb_tt); bin_mcm(mcm_tp, mbb_tp)
        bin_mcm(mcm_pp_diag, mbb_pp_diag); bin_mcm(mcm_pp_offdiag, mbb_pp_offdiag)
        
        mbb_pp = assamble_mat_pp(mbb_pp_diag, mbb_pp_offdiag)
     
        mbb_tt_inv = np.linalg.inv(mbb_tt)
        mbb_tp_inv = np.linalg.inv(mbb_tp)
        mbb_pp_inv = np.linalg.inv(mbb_pp)

        mcm_core.binning_matrix(mcm_tt.T,bin_lower,bin_upper,bin_sizes, bbl_tt.T)
        mcm_core.binning_matrix(mcm_tp.T,bin_lower,bin_upper,bin_sizes, bbl_tp.T) 
        mcm_core.binning_matrix(mcm_pp_diag.T,bin_lower,bin_upper,bin_sizes, bbl_pp_diag.T)
        mcm_core.binning_matrix(mcm_pp_offdiag.T,bin_lower,bin_upper,bin_sizes, bbl_pp_offdiag.T)

        bbl_pp = assamble_mat_pp(bbl_pp_diag, bbl_pp_offdiag)

        bbl_tt = np.dot(mbb_tt_inv,bbl_tt)
        bbl_tp = np.dot(mbb_tp_inv,bbl_tp)
        bbl_pp = np.dot(mbb_pp_inv,bbl_pp)

        return (mbb_tt, mbb_tp, mbb_pp, mbb_tt_inv, mbb_tp_inv, mbb_pp_inv, bbl_tt, bbl_tp, bbl_pp)

    # process mcm (cls) -------- 
    mbb_cltt, mbb_cltp, mbb_clpp, mbb_cltt_inv, mbb_cltp_inv, mbb_clpp_inv, bbl_cltt, bbl_cltp, bbl_clpp = \
            process_mcm_process(mcm_tt, mcm_tp, mcm_pp_diag, mcm_pp_offdiag, \
            mbb_cltt, mbb_cltp, mbb_clpp_diag, mbb_clpp_offdiag, \
            bbl_cltt, bbl_cltp, bbl_clpp_diag, bbl_clpp_offdiag)

    #save_matrix("CLTT", mbb_cltt); save_matrix("CLTP", mbb_cltp); save_matrix("CLPP", mbb_clpp)
    save_matrix("CLTT_inv", mbb_cltt_inv) 
    save_matrix("CLTP_inv", mbb_cltp_inv)
    save_matrix("CLPP_inv", mbb_clpp_inv)
    save_matrix("BBL_CLTT", bbl_cltt) 
    save_matrix("BBL_CLTP", bbl_cltp) 
    save_matrix("BBL_CLPP", bbl_clpp) 

    del mbb_cltt, mbb_cltp, mbb_clpp, mbb_cltt_inv, mbb_cltp_inv, mbb_clpp_inv
    del bbl_cltt, bbl_cltp, bbl_clpp, bbl_clpp_diag, bbl_clpp_offdiag

    # calc bbl matrix (dls) --------
    l_mcm       = np.arange(mcm_tt.shape[0]) + 2.
    mat_2dl     = np.diag(l_mcm*(l_mcm+1.)/(2.*np.pi))
    mat_2dl_inv = np.diag(1./(l_mcm*(l_mcm+1.)/(2.*np.pi)))

    mcm_tt  = np.dot(mcm_tt, mat_2dl_inv); mcm_tp = np.dot(mcm_tp, mat_2dl_inv)
    mcm_pp_diag  = np.dot(mcm_pp_diag, mat_2dl_inv); mcm_pp_offdiag = np.dot(mcm_pp_offdiag, mat_2dl_inv)
    
    mcm_tt  = np.dot(mat_2dl, mcm_tt); mcm_tp = np.dot(mat_2dl, mcm_tp)
    mcm_pp_diag  = np.dot(mat_2dl, mcm_pp_diag); mcm_pp_offdiag = np.dot(mat_2dl, mcm_pp_offdiag)

    # process mcm (cls) -------- 
    mbb_dltt, mbb_dltp, mbb_dlpp, mbb_dltt_inv, mbb_dltp_inv, mbb_dlpp_inv, bbl_dltt, bbl_dltp, bbl_dlpp = \
            process_mcm_process(mcm_tt, mcm_tp, mcm_pp_diag, mcm_pp_offdiag, \
            mbb_dltt, mbb_dltp, mbb_dlpp_diag, mbb_dlpp_offdiag, \
            bbl_dltt, bbl_dltp, bbl_dlpp_diag, bbl_dlpp_offdiag)

    #save_matrix("DLTT", mbb_dltt); save_matrix("DLTP", mbb_dltp); save_matrix("DLPP", mbb_dlpp)
    save_matrix("DLTT_inv", mbb_dltt_inv) 
    save_matrix("DLTP_inv", mbb_dltp_inv)
    save_matrix("DLPP_inv", mbb_dlpp_inv)
    save_matrix("BBL_DLTT", bbl_dltt) 
    save_matrix("BBL_DLTP", bbl_dltp) 
    save_matrix("BBL_DLPP", bbl_dlpp) 

    del mbb_dltt, mbb_dltp, mbb_dlpp, mbb_dltt_inv, mbb_dltp_inv, mbb_dlpp_inv
    del bbl_dltt, bbl_dltp, bbl_dlpp, bbl_dlpp_diag, bbl_dlpp_offdiag

    del mcm_tt, mcm_tp, mcm_pp_diag, mcm_pp_offdiag

class PITAS_BINNER(object):
    def __init__(self, bin_edges, lmax=None): 

        lmax      = int(np.max(bin_edges) if lmax is None else lmax)
        assert(lmax >= 0.)
       
        bin_edges    = bin_edges.astype(np.int)
        bin_edges[0] = 2

        bin_lower      = bin_edges[:-1].copy()
        bin_lower[1:] += 1
        bin_upper      = bin_edges[1:].copy()
        
        bin_upper = bin_upper[np.where(bin_upper <= lmax)]
        bin_lower = bin_lower[:len(bin_upper)]
        bin_lower = bin_lower[np.where(bin_lower <= lmax)]

        self.lmax       = lmax
        self.bin_edges  = bin_edges
        self.bin_lower  = bin_lower
        self.bin_upper  = bin_upper
        self.bin_center = (bin_lower + bin_upper)/2.
        self.bin_sizes  = bin_upper - bin_lower + 1
        self.nbin       = len(bin_lower)

        assert((self.bin_sizes > 0)).all()


    def bin(self, l, cl):
        lbin  = self.bin_center
        clbin = np.zeros(self.nbin)
        for idx in range(0, self.nbin):
            low_lim , upp_lim = (self.bin_lower[idx], self.bin_upper[idx])
            
            loc   = np.where((l >= low_lim) & (l <= upp_lim))
            clbin[idx] = cl[loc].mean()

        return (lbin, clbin)
















