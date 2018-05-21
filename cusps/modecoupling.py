#!/usr/bin/env python
from enlib import enmap,curvedsky
from cusps.util import get_spectra
from cusps.mcm_core import mcm_core
import cusps, numpy as np, os 

def generate_mcm(window_temp, window_pol, bin_edges, mcm_dir=None, lmax=None, transfer=None):
    # expect enmap

    if mcm_dir is None: mcm_dir = cusps.config.get_output_dir()
    cusps.cusps_io.create_dir(mcm_dir)

    lmax      = int(np.max(bin_edges) if lmax is None else lmax)
    assert(lmax >= 0.)

    # make sure that bin is edges are interger
    bin_edges = bin_edges.astype(np.int)
    if bin_edges[0] < 2: bin_edges[0] = 2
    bin_edges = bin_edges[np.where(bin_edges <= lmax)] 
    bin_sizes = bin_edges[1:] - bin_edges[:-1]
    assert((bin_sizes > 0).all())

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
    else: 
        f_tran = np.ones(len(l))

    # full sized mcm matrices
    mcm_tt, mcm_tp, mcm_pp_diag, mcm_pp_offdiag = np.zeros((4, lmax, lmax))
    # binned mcm matrices
    mbb_tt, mbb_tp, mbb_pp_diag, mbb_pp_offdiag = np.zeros((4, nbin, nbin))
    bbl                                         = np.zeros((nbin, lmax))

    mcm_core.calc_mcm(cl_temp, cl_cross, cl_pol, f_tran, mcm_tt.T, mcm_tp.T, mcm_pp_diag.T, mcm_pp_offdiag.T)

    # bin mcm
    def bin_mcm(mcm_full, mcm_binned):
        bin_lower, bin_upper = bin_edges[:-1], bin_edges[1:]
        mcm_core.bin_mcm(mcm_full.T, bin_lower, bin_upper, bin_sizes.T, mcm_binned.T)

    bin_mcm(mcm_tt, mbb_tt); bin_mcm(mcm_tp, mbb_tp)
    bin_mcm(mcm_pp_diag, mbb_pp_diag); bin_mcm(mcm_pp_offdiag, mbb_pp_offdiag)

    # clean up
    del mcm_tt, mcm_tp, mcm_pp_diag, mcm_pp_offdiag

    # combine pol data
    mbb_pp = np.zeros((3*nbin, 3*nbin))
    mbb_pp[:nbin, :nbin]         = mbb_pp[2*nbin:3*nbin, 2*nbin:3*nbin] = mbb_pp_diag
    mbb_pp[2*nbin:3*nbin, :nbin] = mbb_pp[:nbin, 2*nbin:3*nbin]         = mbb_pp_offdiag
    mbb_pp[nbin:2*nbin, nbin:2*nbin]                                    = mbb_pp_diag-mbb_pp_offdiag 

    def save_mbb(key, mbb):
        file_name = 'curved_full_%s.dat' %key
        file_name = os.path.join(mcm_dir, file_name)
        np.savetxt(file_name, mbb)

    save_mbb("TT", mbb_tt); save_mbb("TP", mbb_tp); save_mbb("PP", mbb_pp)
    save_mbb("TT_inv", np.linalg.inv(mbb_tt)) 
    save_mbb("TP_inv", np.linalg.inv(mbb_tp))
    save_mbb("PP_inv", np.linalg.inv(mbb_pp))

    # implement binning
    #test binning matrix
    #mcm_code_full.binning_matrix(mcm.T,binLo,binHi,binsize, Bbl.T)

    #inv=np.linalg.inv(mbb)
    #Bbl=np.dot(inv,Bbl)
    #np.save('%sBbl.%s_curved_T_%s_%s.dat'%(mbbDir,mbbFilename,arrayTag,spTag),mcm)


