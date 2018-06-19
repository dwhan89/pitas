import cusps
from cusps import power
from cusps.util import cl2dl
from actsims import simTools as act_sim
from enlib import enmap, curvedsky
from orphics import stats, io, maps
import numpy as np
import os, flipper, sys


# directory setup
postfix     = 'simple_test_output'
output_dir  = os.path.join('./', postfix)
cmb_dir     = '/global/cscratch1/sd/engelen/simsS1516_v0.3/data/' # alex's sim on NERSC
output_path = lambda x: os.path.join(output_dir, x)

cusps.cusps_io.create_dir(output_dir)

# miscs 
lmax      = 6000
deg       = 15
coords    = np.array([[-deg/2.,-deg/2.],[deg/2.,deg/2.]])
polcombs  = ['tt','ee','bb','te', 'pp']
bin_edges = np.linspace(0, lmax, 60)

def get_sim(sim_idx):
    ret = act_sim.getActpolCmbSim(None, coords, sim_idx, cmb_dir, doBeam=False, pixelFac= 2)
     
    for i in range(len(ret)):
        ret[i]  = enmap.from_flipper(ret[i])
        ret[i] -= np.mean(ret[i])
    return (ret[0], ret[1], ret[2]) # TQU

# initialize window
temp_map, _, _  = get_sim(0)
shape, wcs      = temp_map.shape, temp_map.wcs
taper, _        = maps.get_taper(shape)
taper           = enmap.enmap(taper, wcs=wcs)

# initialize cusps (if it is a first time, it will take 10min+ to compute mode coupling) 
overwrite      = False
mcm_identifier = "simple_test"
cusps_fc       = cusps.power.CUSPS(mcm_identifier, taper, taper, bin_edges, lmax, None, overwrite)
binner         = cusps_fc.binner
lbin           = cusps_fc.bin_center

# helper function
def qu2eb(qmap, umap):
    # replace it with enlib QU2EB
    import polTools

    qtemp = qmap.to_flipper()
    utemp = umap.to_flipper()
    emap, bmap = polTools.convertToEB(qtemp, utemp, True, False) 

    emap = enmap.from_flipper(emap)
    bmap = enmap.from_flipper(bmap)

    return (emap, bmap)

# load sims
tmap, qmap, umap  = get_sim(0) 
tmap -= np.mean(tmap)
qmap -= np.mean(qmap)
umap -= np.mean(umap)
qmap *= -1. 
tmap *= taper
qmap *= taper
umap *= taper

emap, bmap = qu2eb(qmap, umap)

# taking spectra here
lbin, cltt = cusps_fc.get_power(tmap, polcomb='TT')
lbin, clte = cusps_fc.get_power(tmap, emap, polcomb='TE')

lbin, clee, cleb, clbb = cusps_fc.get_power_pureeb(emap, bmap)

spectra = {'tt':cltt, 'ee':clee, 'bb':clbb, 'te':clte,}

for spec_idx in spectra.keys():
    dlbin      = cl2dl(lbin, spectra[spec_idx])
    plotter = cusps.visualize.plotter(yscale='linear')
    plotter.add_data(lbin, dlbin, label=spec_idx.upper())
    plotter.set_title('Curved Sky Dl_%s ' %spec_idx.upper())
    plotter.set_xlabel(r'$l$') 
    plotter.set_ylabel(r'$Dl(l)$')
    plotter.set_xlim([0,5000])
    plotter.show_legends()
    plotter.save(output_path("%s_spec.png"%spec_idx))


