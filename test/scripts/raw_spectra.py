import cmblens
from cusps import power
from actsims import simTools as act_sim
from enlib import enmap, curvedsky
from orphics import stats, io, maps
import numpy as np
import os, flipper
from cmblens.util import cl2dl

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

output_dir  = '/global/homes/d/dwhan89/shared/outbox/cori/test_cusps'
output_path = lambda x: os.path.join(output_dir, x)

cmblens.util.create_dir(output_dir)

# sim directory
cmb_dir        = '/global/cscratch1/sd/engelen/simsS1516_v0.2/data/'

# load theory
l_th = np.arange(10000)
theo = cmblens.theory.load_theory_cls('cosmo2017_10K_acc3_unlensed_cmb', unit='camb', l_interp=l_th)
l_th, cltt_th, dltt_th = theo['l'], theo['cltt'], theo['dltt']

deg      = 20
bin_file ='DELTA50_0_2000'
ps       = cltt_th.reshape((1,1,cltt_th.size))
coords  = np.array([[-deg/2.,-deg/2.],[deg/2.,deg/2.]])
proj     = 'car'
res      = 1
lmax     = 5000

bin_edges     = np.linspace(0, lmax, 50)
quick_binner  = stats.bin1D(bin_edges)

lbin  = None

# for flat periodic sims
shape, wcs = maps.rect_geometry(width_deg = deg, px_res_arcmin=0.5)
mg         = maps.MapGen(shape,wcs,cltt_th.reshape((1,1,l_th.size)))

fc         = maps.FourierCalc(shape,wcs)

#
def get_sim(sim_idx, mode='flat'):
    assert(mode in ['flat', 'actsim'])

    ret = None
    if mode == 'flat':
        ret =  mg.get_map(seed=sim_idx)
    elif mode == 'actsim':
        ret = enmap.from_flipper(act_sim.getActpolCmbSim(None, coords, sim_idx, cmb_dir, doBeam=False)[0])
    else:
        assert(1)

    ret -= np.mean(ret)
    return ret

def get_flat_power(map1, map2=None):
    map2 = map1 if map2 is None else map2
    power2d, _, _ = fc.power2d(emap=map1,emap2=map2)
    binner  = stats.bin2D(map1.modlmap(), bin_edges) 
    return binner.bin(power2d)



st = stats.Stats(cmblens.mpi.comm)
for sim_idx in subtasks:
    log.info("processing %d" %sim_idx)
    imap = get_sim(sim_idx, mode='actsim') 
    if sim_idx == 0:
        io.high_res_plot_img(imap, output_path('tmap_unlen_%d.png'%sim_idx), down=3)
    
    l, cl = power.get_raw_power(imap, lmax=lmax)
    lbin, cl = quick_binner.binned(l, cl)

    st.add_to_stats("dltt", cl2dl(lbin, cl)) 

    l, cl = get_flat_power(imap)
    st.add_to_stats('dltt_flat', cl2dl(lbin, cl))   
 

st.get_stats()

def add_with_err(plotter, st, l, key, **kwargs): 
    mean =  st.stats[key]['mean']
    err  =  st.stats[key]['errmean']
    if not np.isnan(err).any():
        plotter.add_err(l, mean, err, **kwargs)
    else:
        plotter.add_data(l, mean, **kwargs)
        

if cmblens.mpi.rank == 0:
    log.info("plotting")

    plotter = cmblens.visualize.plotter(yscale='log')
    plotter.add_data(l_th, dltt_th, label='DlTT (Theo)') 
    add_with_err(plotter, st, lbin, 'dltt', label='DlTT (CUSPS)') 
    add_with_err(plotter, st, lbin, 'dltt_flat', label='DlTT (FC)')
    plotter.set_xlim([0,5000])
    plotter.show_legends()
    plotter.save(output_path("raw_spec.png"))
