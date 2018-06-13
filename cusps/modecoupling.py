#!/usr/bin/env python
from enlib import enmap,curvedsky
from cusps.util import get_spectra
from cusps.mcm_core import mcm_core
import cusps, numpy as np, os 
from orphics import stats

def generate_mcm(window_temp, window_pol, bin_edges, mcm_dir=None, lmax=None, transfer=None):
    # expect enmap

    if mcm_dir is None: mcm_dir = cusps.config.get_output_dir()
    cusps.cusps_io.create_dir(mcm_dir)

    modlmap   = window_temp.modlmap()
    lmax      = int(np.max(bin_edges) if lmax is None else lmax)
    assert(lmax >= 0.)

    # make sure that bin is edges are interger
    bin_edges = bin_edges.astype(np.int)
    if bin_edges[0] < 2: bin_edges[0] = 2
    bin_edges = bin_edges[np.where(bin_edges <= lmax)] 
    bin_sizes = bin_edges[1:] - bin_edges[:-1]
    assert((bin_sizes > 0).all())
    binner    = stats.bin1D(bin_edges)

    nbin      = len(bin_edges) - 1
    assert(nbin > 0)

    def process_spectrum(window1, window2=None, lmax=lmax):
        l, cl = get_spectra(window1, window2, lmax=lmax)
        cl    *= (2.*l+1.)
        return l, cl
    

    l, cl_temp   = process_spectrum(window_temp, lmax=lmax)
    l, cl_cross  = process_spectrum(window_temp, window_pol, lmax=lmax)
    l, cl_pol    = process_spectrum(window_pol, lmax=lmax)


    f_tran = None
    if transfer is not None:
        l_tran, f_tran = transfer
        l_tran, f_tran = l_tran[:lmax],f_tran[:lmax]
        transfer       = f_tran **2.
    else: 
        transfer       = np.ones(len(l))
        #f_tran = np.ones(len(l))
        

    # full sized mcm matrices
    mcm_tt, mcm_tp, mcm_pp_diag, mcm_pp_offdiag = np.zeros((4, lmax, lmax))
    # binned mcm matrices
    mbb_tt, mbb_tp, mbb_pp_diag, mbb_pp_offdiag = np.zeros((4, nbin, nbin))
    bbl                                         = np.zeros((nbin, lmax))

    mcm_core.calc_mcm(cl_temp, cl_cross, cl_pol, transfer, mcm_tt.T, mcm_tp.T, mcm_pp_diag.T, mcm_pp_offdiag.T)

    # bin mcm
    def bin_mcm(mcm_full, mcm_binned):
        bin_lower, bin_upper = bin_edges[:-1], bin_edges[1:]
        mcm_core.bin_mcm(mcm_full.T, bin_lower, bin_upper, bin_sizes.T, mcm_binned.T)

    bin_mcm(mcm_tt, mbb_tt); bin_mcm(mcm_tp, mbb_tp)
    bin_mcm(mcm_pp_diag, mbb_pp_diag); bin_mcm(mcm_pp_offdiag, mbb_pp_offdiag)


    # implement binning
    #test binning matrix
    mcm_core.binning_matrix(mcm_tt.T,bin_edges[:-1],bin_edges[1:],bin_sizes, bbl.T)
    bbl = np.dot(np.linalg.inv(mbb_tt),bbl)
    # clean up

    del mcm_tt, mcm_tp, mcm_pp_diag, mcm_pp_offdiag

    # combine pol data
    def assamble_mat_pp(mat_pp_diag, mat_pp_offdiag):
        nbin = mat_pp_diag.shape[0] # pp matrix is always nxn matrix
        mat_pp = np.zeros((3*nbin, 3*nbin))
        mat_pp[:nbin, :nbin]         = mat_pp[2*nbin:3*nbin, 2*nbin:3*nbin] = mat_pp_diag
        mat_pp[2*nbin:3*nbin, :nbin] = mat_pp[:nbin, 2*nbin:3*nbin]         = mat_pp_offdiag
        mat_pp[nbin:2*nbin, nbin:2*nbin]                                    = mat_pp_diag-mat_pp_offdiag  

        return mat_pp

    mbb_pp = assamble_mat_pp(mbb_pp_diag, mbb_pp_offdiag)
    
    def generate_bbl(mcm_tt, mcm_tp, mcm_pp_diag, mcm_pp_offdiag, lmax, bin_sizes):
        nbin   = len(bin_sizes)
        #mcm_pp = assamble_mat_pp(mcm_pp_diag, mcm_pp_offdiag)
        #bbl_tt, bbl_tp = np.zeros((2, nbin, lmax-1)) # bbl starts at l = 2 
        #bbl_pp         = np.zeros((3*nbin, 3*(lmax-1)))
        bbl_size       = mcm_tt.shape[0]
        l_bbl          = np.arange(bbl_size) + 2 # starts from l=2
        #bbl_size       = len(l_bbl)
        guass_mat      = np.zeros(bbl_size, bbl_size)
        #guass_mat_pp   = np.zeros(3*bbl_size, bbl_size) 
        bbl_tt         = np.zeros((nbin, bbl_size))


        modlmap1d = np.ravel(modlmap) 
        sigma     = np.mean(modlmap1d[1:]-modlmap1d[:-1])
        del modlmap1d

        # build I(l,l') 
        for l_cur in l_bbl:
            l_idx = l_cur - 2 # currection for zero indexing
            guass_mat[l_idx,:] = (-(l_bbl - l_cur)**2./(2.*sigma)**2.)
            gauss_mat[l_idx,:] /= np.mean(guass_mat[l_idx,:])

        #gauss_mat_pp[:bbl_size,:] = gauss_mat_pp[bbl_size:2*bbl_size,:] \
        #        = gauss_mat_pp[2*bbl_size:3*bbl_size,:] = guass_mat[:,:]
        
        bbl_tt_raw = np.dot(mcm_tt, guass_mat)
        del gauss_mat#, guass_mat_pp
    
        # start binning
        for l_cur in l_bbl:
            l_idx = l_cur - 2
            _, bbl_tt[:,l_idx] = binner.binned(bbl_tt_raw[:,l_idx])
        
        del bbl_tt_raw
        return bbl_tt

    bbl_tt = np.dot(np.linalg.inv(mbb_tt), bbl_tt)

    
    def save_matrix(key, mbb):
        file_name = 'curved_full_%s.dat' %key
        file_name = os.path.join(mcm_dir, file_name)
        np.savetxt(file_name, mbb)

    save_matrix("TT", mbb_tt); save_matrix("TP", mbb_tp); save_matrix("PP", mbb_pp)
    save_matrix("TT_inv", np.linalg.inv(mbb_tt)) 
    save_matrix("TP_inv", np.linalg.inv(mbb_tp))
    save_matrix("PP_inv", np.linalg.inv(mbb_pp))
    save_matrix("BBL", bbl) 
    save_matrix("BBLNEW", bbl_tt)
