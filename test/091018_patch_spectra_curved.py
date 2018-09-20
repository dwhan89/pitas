import cmblens
import pitas 
import pitas.power as power
from actsims import simTools as act_sim
from enlib import enmap, curvedsky, lensing, powspec, utils
from orphics import stats, io, maps
import numpy as np
import os, flipper, sys
import healpy as hp
from cmblens.util import cl2dl
import seaborn as sns
from actsims.simTools import resample_fft_withbeam

log = cmblens.logger.getLogger()

cmblens.config.argparser.add_argument('-s', '--start',
                default=0, type=int,
                        help='lower bound for the sims to be used')

cmblens.config.argparser.add_argument('-e', '--end',
                default=319, type=int,
                        help='upper bound for the sims to be used')


cmblens.config.argparser.add_argument('-mp', '--mpi',
                default='t',
                help='switch for mpi')

args = cmblens.config.argparser.parse_args()

# initialize mpi 
cmblens.mpi.init(cmblens.util.str2bool(args.mpi))

start  = args.start
end    = args.end
nsims  = end - start + 1

assert(end >= start)

# evenly divide work over ranks
subtasks = cmblens.mpi.taskrange(imin=start, imax=end)

# directory setup
theory_dir  = '/global/homes/d/dwhan89/cori/workspace/cmblens/inputParams'
postfix     = '091018_fullsky_0.5res_oldbin'
output_dir  = os.path.join('/global/homes/d/dwhan89/shared/outbox/cori/test_cusps', postfix)
output_path = lambda x: os.path.join(output_dir, x)

if cmblens.mpi.rank == 0:
    cmblens.util.create_dir(output_dir)
else: pass
cmblens.mpi.barrier()


# load theory
sim_lmax = 8000
lmax     = 6000
input_theo = powspec.read_camb_full_lens(os.path.join(theory_dir, 'cosmo2017_10K_acc3_lenspotentialCls.dat')) 
theo       = cmblens.theory.load_theory_cls('cosmo2017_10K_acc3_lensed_cmb', unit='camb', l_interp=None)
l_th       = theo['l']
bin_edges = np.linspace(0, lmax, 60)
binner     = pitas.modecoupling.PITAS_BINNER(bin_edges, lmax=lmax)


theo_bin = {}
for key in theo.keys():
    if key == 'l': continue
    lbin_th, clbin_th = binner.bin(l_th, theo[key])

    theo_bin['l'] = lbin_th
    theo_bin[key] = clbin_th

lbin = binner.bin_center

# miscs 
coords_str   = 'fullsky'

proj      = 'car'
polcombs  = ['tt','ee','bb']
#bin_edges = pitas.util.get_default_bin_edges(lmax)
res          = 0.5 * utils.arcmin


def get_sim(sim_idx):
    oshape, owcs = enmap.fullsky_geometry(res=res, proj='car')
    nshape       = (3,) + oshape

    ret          = lensing.rand_map(nshape, owcs, input_theo, lmax=sim_lmax, maplmax=sim_lmax, seed=sim_idx)

    return ret[0][0], ret[0][1], ret[0][2]


# helper function
def tqu2teb(tmap, qmap, umap):
    log.info("coverting tqu2tqb")
    tqu     = np.zeros((3,)+tmap.shape)
    tqu[0], tqu[1], tqu[2] = (tmap, qmap, umap)
    tqu     = enmap.enmap(tqu, tmap.wcs)
    alm     = curvedsky.map2alm(tqu, lmax=sim_lmax)

    teb     = curvedsky.alm2map(alm[:,None], tqu[:,None], spin=0)[:,0]
    del tqu

    return (teb[0], teb[1], teb[2])

def add_spectra(tmap, emap, bmap, sim_idx):
    log.info("taking spectra %d" %sim_idx)
    cmb_dict = {'t': tmap, 'e': emap, 'b': bmap}
    polcombs = ['tt', 'ee', 'bb']#, 'pp']
    #polcombs = ['ee']
    #polcombs=['tt']

    global lbin
    for polcomb in polcombs:
        if polcomb in ['pp']: continue
        emap1 = cmb_dict[polcomb[0]]
        emap2 = cmb_dict[polcomb[1]]
        print '[add_spectra]:', polcomb

        l, cl = power.get_raw_power(emap1, emap2=emap2, lmax=lmax, normalize=False)
        lbin, clbin = binner.bin(l,cl)
        st.add_data('dl%s_raw'%polcomb, sim_idx, cl2dl(lbin, clbin))

        theo_idx  = 'cl%s'% polcomb
        frac_diff =  (clbin - theo_bin[theo_idx])/theo_bin[theo_idx]
        st.add_data('frac%s_raw'%polcomb, sim_idx, frac_diff)

plot_only       = False
stat_override   = False
stat_identifier = '0.5arcmin_fullsky_cusps_ps_lmax%d_oldbin' %sim_lmax
st = cmblens.stats.STATS(stat_identifier=stat_identifier, overwrite=stat_override)

for sim_idx in subtasks:
    log.info("processing %d" %sim_idx)
    if st.has_data('fracbb_raw', sim_idx) or plot_only: continue
    tmap, qmap, umap  = get_sim(sim_idx) 
    tmap, emap, bmap = tqu2teb(tmap, qmap, umap)
    del qmap, umap


    if sim_idx == 0 and False:
        io.high_res_plot_img(tmap, output_path('tmap_len_%d_%s.png'%(sim_idx,coords_str)), down=3)
        io.high_res_plot_img(emap, output_path('emap_len_%d_%s.png'%(sim_idx,coords_str)), down=3)
        io.high_res_plot_img(bmap, output_path('bmap_len_%d_%s.png'%(sim_idx,coords_str)), down=3)

    if plot_only: continue
    cmb_dict = {'t': tmap, 'e': emap, 'b': bmap}
   
    add_spectra(tmap, emap, bmap, sim_idx) 
st.get_stats()

def add_with_err(plotter, st, l, key, **kwargs): 
    mean =  st.stats[key]['mean']
    err  =  st.stats[key]['std_mean']
    if not np.isnan(err).any():
        plotter.add_err(l, mean, err, ls='--', alpha=0.5, marker='o', **kwargs)
    else:
        plotter.add_data(l, mean, ls='--', alpha=0.5, **kwargs)
        

if cmblens.mpi.rank == 0:
    log.info("plotting")
   
    for polcomb in polcombs:
        if polcomb == 'pp': continue
        prefix  = 'dl%s' % polcomb
        plotter = pitas.visualize.plotter(yscale='linear')
        plotter.add_data(theo_bin['l'], theo_bin[prefix], label='Dl%s Binned Theory'%polcomb.upper()) 
        add_with_err(plotter, st, lbin, '%s_raw'%prefix, label='Dl%s (PITAS)'%polcomb.upper()) 
        plotter.set_title('Curved Sky Dl_%s ' %polcomb.upper())
        plotter.set_xlabel(r'$l$') 
        plotter.set_ylabel(r'$Dl(l)$')
        plotter.set_xlim([-50,5000])
        plotter.show_legends()
        plotter.save(output_path("%s_spec_%s.png"%(prefix, coords_str)))

   
    for polcomb in polcombs:
        if polcomb == 'pp': continue
        prefix  = 'dl%s' % polcomb
        plotter = pitas.visualize.plotter(figsize=(10, 8), yscale='linear') 
        add_with_err(plotter, st, lbin, 'frac%s_raw'%polcomb, label='Dl%s (PITAS)'%polcomb.upper())
        plotter.set_title('Fractional Difference Dl_%s ' %polcomb.upper(), fontsize=22)
        plotter.set_xlabel(r'$l$', fontsize=22) 
        plotter.set_ylabel(r'$(sim - theo)/theo$', fontsize=22)
        plotter.set_xlim([-50,5000])
        plotter.set_ylim([-0.05,0.05])
        plotter.hline(y=0, color='k')
        plotter.show_legends(fontsize=18)
        plotter.save(output_path("frac_diff%s_%s.png"%(prefix, coords_str)))

