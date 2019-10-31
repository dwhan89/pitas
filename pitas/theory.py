#-
# theory.py
#-
#

import pitas
import numpy as np
import os
from scipy.interpolate import interp1d

def load_theory_cls(mode, unit='camb', l_interp = None, load_dls=True, fill_value=0.):
    '''
        mode: lensed or unlensed

        output
        l, cltt, clee, clbb, clte 
    '''

    file_dict = {'non-linear_lensed':'non-linear_lensedCls.dat',\
            'bode_almost_wmap5_lmax_1e4_lensed': 'bode_almost_wmap5_lmax_1e4_lensedCls.dat',\
            'bode_almost_wmap5_lmax_1e4_scal': 'bode_almost_wmap5_lmax_1e4_scalCls.dat',\
            'cosmo2017_10K_acc3_lensed': 'cosmo2017_10K_acc3_lensedCls.dat',\
            'cosmo2017_10K_acc3_lensed_extended': 'cosmo2017_10K_acc3_lensedCls_extended.dat',\
            'cosmo2017_10K_acc3_unlensed': 'cosmo2017_10K_acc3_scalCls.dat'
            }

    assert(unit in ['camb', 'uk'])

    if file_dict.has_key(mode):
        path2file = os.path.join(pitas.config.get_resource_dir(), file_dict[mode])
        print("loading %s" %path2file)
    else:
        raise ValueError("%s is not supported" %mode)

    
    from pitas.constants import TCMB
    
    theo = np.loadtxt(path2file)
    cls = {}
  
    if mode in  ['non-linear_lensed', 'bode_almost_wmap5_lmax_1e4_lensed','cosmo2017_10K_acc3_lensed',\
            'cosmo2017_10K_acc3_lensed_extended']:
        l    = theo[:,0]

        dltt = theo[:,1]
        dlee = theo[:,2]
        dlbb = theo[:,3]
        dlte = theo[:,4]

        cltt = dltt*(2*np.pi)/(l*(l+1.0))
        clee = dlee*(2*np.pi)/(l*(l+1.0))
        clbb = dlbb*(2*np.pi)/(l*(l+1.0))
        clte = dlte*(2*np.pi)/(l*(l+1.0))

        cls['l']    = l
        cls['cltt'] = cltt
        cls['clee'] = clee
        cls['clbb'] = clbb
        cls['clte'] = clte
        if load_dls:
            cls['dltt'] = dltt
            cls['dlee'] = dlee
            cls['dlbb'] = dlbb
            cls['dlte'] = dlte
        else: pass


    elif mode in ['bode_almost_wmap5_lmax_1e4_scal', 'cosmo2017_10K_acc3_unlensed']:
        l    = theo[:,0]

        dltt = theo[:,1]
        dlee = theo[:,2]
        dlbb = np.zeros(l.shape)
        dlte = theo[:,3]
        clpp = theo[:,4]/(l**4.)

        cltt = dltt*(2*np.pi)/(l*(l+1.0))
        clee = dlee*(2*np.pi)/(l*(l+1.0))
        clbb = np.zeros(l.shape)
        clte = dlte*(2*np.pi)/(l*(l+1.0))

        cls['l']    = l
        cls['cltt'] = cltt
        cls['clee'] = clee
        cls['clbb'] = clbb
        cls['clte'] = clte 
        cls['clpp'] = clpp
        cls['clkk'] = clpp*(l*(l+1.))**2./4.
        if load_dls:
            cls['dltt'] = dltt
            cls['dlee'] = dlee
            cls['dlbb'] = dlbb
            cls['dlte'] = dlte
        else: pass

    else:
        raise ValueError()
        
    if unit == 'uk':
        print("converting theory to %s")
        for key in list(cls.keys()):
            if key == 'l': continue
            else         : cls[key] /= TCMB**2
    else: pass

    if l_interp is not None:
        print("interpolating")
        for key in list(cls.keys()):
            if key == 'l': continue
            else         : cls[key] = interp1d(cls['l'], cls[key], bounds_error=False, fill_value=fill_value)(l_interp)
        cls['l'] = l_interp
    else: pass

    return cls

def load_theory_func(mode, unit='camb', fill_value=np.nan):
    cls = load_theory_cls(mode, unit)
    l   = cls['l']

    for key in list(cls.keys()):
        if key == ['l']: continue
        else           : cls[key] = interp1d(l, cls[key], bounds_error=False, fill_value=fill_value)

    return cls

#def load_theory_from_orphics('lensed'

def get_total_1d_power(ell, cl, beam_fwhm, noiselevel, deconvolve=False, get_dl=True, unit='camb'):
    cl_ret          = cl.copy()
    _, beam_factor  = get_gauss_beam(ell, beam_fwhm)
    cl_ret          *= beam_factor
    cl_ret          += get_white_noise_power(ell, noiselevel, unit=unit)
    if deconvolve: cl_ret /= beam_factor
    dl = None if not get_dl else ell*(ell+1.)/(2.*np.pi) * cl_ret

    return (ell, cl_ret, dl)


def get_white_noise_power(lbin, noiselevel, unit='camb'):
    '''
        input: noiselevel in uk.arcmin
    '''
    ps      = np.ones(lbin.size)
    ps      *= (np.pi / (180. * 60))**2.  * noiselevel **2.
    if unit=='uk':
        from pitas.constants import TCMB
        ps /= TCMB **2.

    return ps


def get_gauss_beam(lbin, beam_fwhm):
    '''
        beam_fwhm in arcmins # equivalent to f**2
    '''
    beam_fwhm = np.deg2rad(beam_fwhm/60.)
    sigma     = beam_fwhm/(2.*np.sqrt(2.*np.log(2)))
    beam      = np.exp(-(lbin)**2.*sigma**2.)
   
    #return interp1d(lbin, beam, bounds_error=False, fill_value=beam[-1])
    return (lbin, beam)

def get_gauss_beam_func(lbin, beam_fwhm):
    lbin, beam = get_gauss_beam(lbin, beam_fwhm)

    return interp1d(lbin, beam, bounds_error=False, fill_value=(beam[0], beam[-1]))




