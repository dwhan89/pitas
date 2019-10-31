import cmblens, cusps
from cusps import power
from actsims import simTools as act_sim
from enlib import enmap, curvedsky
from orphics import stats, io, maps
import numpy as np
import os, flipper, sys
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

cmblens.config.argparser.add_argument('-c', '--coords',
                default='[[-6.,-25.],[6.,25.]]')

cmblens.config.argparser.add_argument('-dec', '--declination',
                default='0.0', type=float)

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
postfix     = '062218'
output_dir  = os.path.join('/global/homes/d/dwhan89/shared/outbox/cori/test_cusps', postfix)
cmb_dir     = '/global/cscratch1/sd/engelen/simsS1516_v0.3/data/'
output_path = lambda x: os.path.join(output_dir, x)

if cmblens.mpi.rank == 0:
    cmblens.util.create_dir(output_dir)
else: pass
cmblens.mpi.barrier()


# load theory
lmax = 6000
theo = cmblens.theory.load_theory_cls('cosmo2017_10K_acc3_unlensed_cmb', unit='camb', l_interp=None)
l_th = theo['l']

# miscs 
coords       = np.array(eval(args.coords))
coords[:,0] += args.declination 
coords_str   = str(coords).replace('\n', '').replace(' ', '')

proj      = 'car'
polcombs  = ['tt','ee','bb','te', 'pp']
bin_edges = np.linspace(0, lmax, 60)

def get_sim(sim_idx):
    ret = act_sim.getActpolCmbSim(None, coords, sim_idx, cmb_dir, doBeam=False, pixelFac= 2)
     
    for i in range(len(ret)):
        ret[i]  = enmap.from_flipper(ret[i])
        ret[i] -= np.mean(ret[i])
    return (ret[0], ret[1], ret[2]) # TQU

temp_map, _, _  = get_sim(0)
shape, wcs      = temp_map.shape, temp_map.wcs
taper, _        = maps.get_taper(shape)
taper           = enmap.enmap(taper, wcs=wcs)

# initialize cusps
overwrite      = False
mcm_identifier = "%lsd_le%d_nb%d_lm%d_%s_%s" %(0, lmax, 60, lmax, coords_str, postfix)
cusps_fc       = cusps.power.CUSPS(mcm_identifier, taper, taper, bin_edges, lmax, None, overwrite)
binner         = cusps_fc.binner


# bin the theory
theo_bin = {}
for key in list(theo.keys()):
    if key == 'l': continue
    lbin_th, clbin_th = cusps_fc.bin_theory(l_th, theo[key]) 
    theo_bin['l']     = lbin_th
    theo_bin[key]     = clbin_th

# helper function
def tqu2teb(tmap, qmap, umap):
    tqu     = np.zeros((3,)+tmap.shape)
    tqu[0], tqu[1], tqu[2] = (tmap, qmap, umap)
    tqu     = enmap.enmap(tqu, tmap.wcs)
    alm     = curvedsky.map2alm(tqu, lmax=lmax)

    teb     = curvedsky.alm2map(alm[:,None], tqu.copy()[:,None], spin=0)[:,0]
    del tqu

    return (teb[0], teb[1], teb[2])

plot_only       = False
stat_override   = False
lbin            = cusps_fc.bin_center
stat_identifier = mcm_identifier + 'cusps_ps'
st = cmblens.stats.STATS(stat_identifier=stat_identifier, overwrite=stat_override)

for sim_idx in subtasks:
    log.info("processing %d" %sim_idx)
    tmap, qmap, umap  = get_sim(sim_idx) 
    tmap -= np.mean(tmap)
    qmap -= np.mean(qmap)
    umap -= np.mean(umap)
    tmap *= taper
    qmap *= taper
    umap *= taper

    tmap, emap, bmap = tqu2teb(tmap, qmap, umap)

    if sim_idx == 0:
        io.high_res_plot_img(tmap, output_path('tmap_unlen_%d_%s.png'%(sim_idx,coords_str)), down=3)
        io.high_res_plot_img(qmap, output_path('qmap_unlen_%d_%s.png'%(sim_idx,coords_str)), down=3)
        io.high_res_plot_img(umap, output_path('umap_unlen_%d_%s.png'%(sim_idx,coords_str)), down=3)
        io.high_res_plot_img(emap, output_path('emap_unlen_%d_%s.png'%(sim_idx,coords_str)), down=3)
        io.high_res_plot_img(bmap, output_path('bmap_unlen_%d_%s.png'%(sim_idx,coords_str)), down=3)

    if plot_only: continue
    cmb_dict = {'t': tmap, 'e': emap, 'b': bmap}
    
    # take take tt and te spectra
    for polcomb in polcombs:
        if polcomb in ['ee', 'eb', 'bb', 'pp']: continue
        if st.has_data('frac%s_deconv'%polcomb, sim_idx): continue 
        emap1 = cmb_dict[polcomb[0]]
        emap2 = cmb_dict[polcomb[1]] 
        log.info('[add_spectra]: ' +  polcomb)
        l, cl       = cusps_fc.get_power(emap1, emap2=emap2, polcomb=polcomb.upper())
        lbin, clbin = binner.bin(l,cl)
        st.add_data('dl%s_deconv'%polcomb, sim_idx, cl2dl(lbin, clbin))

        theo_idx  = 'cl%s'% polcomb
        frac_diff =  (clbin - theo_bin[theo_idx])/theo_bin[theo_idx]
        st.add_data('frac%s_deconv'%polcomb, sim_idx, frac_diff)

    # take pol spectra
    if 'pp' in polcombs:
        if st.has_data('fracbb_deconv', sim_idx): continue
        log.info('[add_spectra]:' + ' pp')
        
        lbin, clee, cleb, clbb = cusps_fc.get_power_pureeb(emap, bmap) 
        st.add_data('dlee_deconv', sim_idx, cl2dl(lbin, clee))
        st.add_data('dleb_deconv', sim_idx, cl2dl(lbin, cleb))
        st.add_data('dlbb_deconv', sim_idx, cl2dl(lbin, clbb))

        theo_idx  = 'clee'
        frac_diff =  (clee - theo_bin[theo_idx])/theo_bin[theo_idx]
        st.add_data('frac%s_deconv'%'ee', sim_idx, frac_diff)

        theo_idx  = 'clbb'
        frac_diff =  (clbb - theo_bin[theo_idx])/theo_bin[theo_idx]
        st.add_data('frac%s_deconv'%'bb', sim_idx, frac_diff)

cmblens.mpi.barrier()
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
        plotter = cusps.visualize.plotter(yscale='linear')
        plotter.add_data(theo_bin['l'], theo_bin[prefix], label='Dl%s Binned Theory'%polcomb.upper()) 
        add_with_err(plotter, st, lbin, '%s_deconv'%prefix, label='Dl%s (CUSPS_DECOV)'%polcomb.upper())
        plotter.set_title('Curved Sky Dl_%s ' %polcomb.upper())
        plotter.set_xlabel(r'$l$') 
        plotter.set_ylabel(r'$Dl(l)$')
        plotter.set_xlim([0,5000])
        plotter.show_legends()
        plotter.save(output_path("%s_spec_%s.png"%(prefix, coords_str)))

   
    for polcomb in polcombs:
        if polcomb == 'pp': continue
        prefix  = 'dl%s' % polcomb
        plotter = cusps.visualize.plotter(figsize=(10, 8), yscale='linear') 
        add_with_err(plotter, st, lbin, 'frac%s_deconv'%polcomb, label='Dl%s (curved-sky ps)'%polcomb.upper())
        plotter.set_title('Fractional Difference Dl_%s ' %polcomb.upper(), fontsize=22)
        plotter.set_xlabel(r'$l$', fontsize=22) 
        plotter.set_ylabel(r'$(sim - theo)/theo$', fontsize=22)
        plotter.set_xlim([0,5000])
        plotter.set_ylim([-0.05,0.05])
        plotter.hline(y=0, color='k')
        plotter.show_legends(fontsize=18)
        plotter.save(output_path("frac_diff%s_%s.png"%(prefix, coords_str)))

