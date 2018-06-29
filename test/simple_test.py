import pitas
from pitas import power
from pitas.util import cl2dl
from enlib import enmap, curvedsky, powspec
import numpy as np
import os, sys


# directory setup
postfix       = 'simple_test_output'
output_dir    = os.path.join('./', postfix)
output_path   = lambda x: os.path.join(output_dir, x)
resource_dir  = pitas.config.get_resource_dir() 
resource_path = lambda x: os.path.join(resource_dir, x) 

pitas.pitas_io.create_dir(output_dir)

# miscs 
lmax      = 6000
bin_edges = pitas.util.get_default_bin_edges(lmax)
taper     = enmap.read_fits(resource_path('test_taper.fits'))

# initialize pitas (if it is a first time, it will take 10min+ to compute mode coupling) 
overwrite      = False
mcm_identifier = "simple_test"
pitas_lib      = pitas.power.PITAS(mcm_identifier, taper, taper, bin_edges, lmax, None, overwrite)
binner         = pitas_lib.binner
lbin           = pitas_lib.bin_center

# load sims (unlensed 15x15 sq deg)
tmap, qmap, umap  = enmap.read_fits(resource_path('test_tqu2.fits'))
tmap -= np.mean(tmap)
qmap -= np.mean(qmap)
umap -= np.mean(umap)
tmap *= taper
qmap *= taper
umap *= taper

# load theory 
theo = powspec.read_spectrum(resource_path('cosmo2017_10K_acc3_scalCls.dat'))
cltt_th = theo[0,0,2:]
clee_th = theo[1,1,2:]
clte_th = theo[2,2,2:]
cleb_th = clbb_th = np.zeros(cltt_th.shape)
l_th    = np.arange(len(cltt_th))+2.

# bin theory
lbin, clttbin_th = pitas_lib.bin_theory_scalarxscalar(l_th, cltt_th)
lbin, cltebin_th = pitas_lib.bin_theory_scalarxvector(l_th, clte_th)
lbin, cleebin_th, clebbin_th, clbbbin_th = pitas_lib.bin_theory_pureeb(l_th, clee_th, cleb_th, clbb_th)

# tqu 2 teb
tmap, emap, bmap = pitas.util.tqu2teb(tmap, qmap, umap, lmax)

# taking spectra here
lbin, cltt = pitas_lib.get_power(tmap, polcomb='TT')
lbin, clte = pitas_lib.get_power(tmap, emap, polcomb='TE')

lbin, clee, cleb, clbb = pitas_lib.get_power_pureeb(emap, bmap)

spectra     = {'tt':cltt, 'ee':clee, 'bb':clbb, 'te':clte}
theory      = {'tt':cltt_th, 'ee':clee_th, 'bb':clbb_th, 'te':clte_th}
theory_bin  = {'tt':clttbin_th, 'ee':cleebin_th, 'bb':clbbbin_th, 'te':cltebin_th}

for spec_idx in spectra.keys():
    dlbin      = cl2dl(lbin, spectra[spec_idx])
    dlth       = cl2dl(lbin, theory_bin[spec_idx])
    plotter = pitas.visualize.plotter(yscale='linear')
    plotter.add_data(lbin, dlbin, label='sim %s' %spec_idx.upper()) 
    plotter.add_data(lbin, dlth, label='binned theory %s' %spec_idx.upper(), color='k', ls='--')
    plotter.set_title('Curved Sky Dl_%s ' %spec_idx.upper())
    plotter.set_xlabel(r'$l$') 
    plotter.set_ylabel(r'$Dl(l)$')
    plotter.set_xlim([200,5000])
    plotter.show_legends()
    plotter.save(output_path("%s_spec.png"%spec_idx))


