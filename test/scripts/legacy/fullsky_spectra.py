import cmblens, cusps
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

proj     = 'car'
res      = 1
lmax     = 6000


overwrite = False

bin_edges = np.linspace(0, lmax, 60)
binner    = stats.bin1D(bin_edges)

mcm_identifier = "%lsd_le%d_nb%d_lm%d_a%d" %(0, lmax, 60, lmax, deg**2)

theo_bin = {}
for key in list(theo.keys()):
    if key == 'l': continue
    lbin_th, clbin_th = binner.binned(l_th, theo[key])
    
    theo_bin['l'] = lbin_th
    theo_bin[key] = clbin_th

lbin = None
def add_spectra(tmap, emap, bmap, deconv):
    cmb_dict = {'t': tmap, 'e': emap, 'b': bmap}
    #polcombs = ['tt', 'te', 'ee', 'bb', 'pp']
    #polcombs = ['ee']
    polcombs=['tt']
    
    global lbin 
    if not deconv:
        for polcomb in polcombs:
            if polcomb in ['pp']: continue
            emap1 = cmb_dict[polcomb[0]]
            emap2 = cmb_dict[polcomb[1]]
            print('[add_spectra]:', polcomb)

            l, cl = power.get_raw_power(emap1, emap2=emap2, lmax=lmax, normalize=False)   
            lbin, clbin = binner.binned(l,cl)
            st.add_to_stats('dl%s_raw'%polcomb, cl2dl(lbin, clbin))

            theo_idx  = 'cl%s'% polcomb
            frac_diff =  (clbin - theo_bin[theo_idx])/theo_bin[theo_idx]
            st.add_to_stats('frac%s_raw'%polcomb, frac_diff)

    else: 
        fsky = np.sum(taper)/taper.size
        print(fsky)
        for polcomb in polcombs:
            if polcomb in ['ee', 'eb', 'bb', 'pp']: continue
            emap1 = cmb_dict[polcomb[0]]
            emap2 = cmb_dict[polcomb[1]] 
            print('[add_spectra]:', polcomb)
            l, cl = cusps_fc.get_power(emap1, emap2=emap2, polcomb=polcomb.upper())
            #cl /= 0.669
            lbin, clbin = binner.binned(l,cl)
            clbin /= fsky
            st.add_to_stats('dl%s_deconv'%polcomb, cl2dl(lbin, clbin))

            theo_idx  = 'cl%s'% polcomb
            frac_diff =  (clbin - theo_bin[theo_idx])/theo_bin[theo_idx]
            st.add_to_stats('frac%s_deconv'%polcomb, frac_diff)
    
        if 'pp' in polcombs:
            print('[add_spectra]:', 'pp')
            lbin, clee, cleb, clbb = cusps_fc.get_pureeb_power(emap, bmap)
            st.add_to_stats('dlee_deconv', cl2dl(lbin, clee))
            st.add_to_stats('dleb_deconv', cl2dl(lbin, cleb))
            st.add_to_stats('dlbb_deconv', cl2dl(lbin, clbb))

            theo_idx  = 'clee'
            frac_diff =  (clbin - theo_bin[theo_idx])/theo_bin[theo_idx]
            st.add_to_stats('frac%s_deconv'%'ee', frac_diff)

            theo_idx  = 'clbb'
            frac_diff =  (clbin - theo_bin[theo_idx])/theo_bin[theo_idx]
            st.add_to_stats('frac%s_deconv'%'bb', frac_diff)

st = stats.Stats(cmblens.mpi.comm)
for sim_idx in subtasks:
    log.info("processing %d" %sim_idx)
    sim_idx_str = str(sim_idx)
    sim_idx_str = sim_idx_str.zfill(5)

    tmap = enmap.read_fits("/global/cscratch1/sd/engelen/simsS1516_v0.2/data/cmb_set00_%s/fullskyUnlensedMap_T_%s.fits" %(sim_idx_str, sim_idx_str))
    tmap -= np.mean(tmap)
    #tmap, qmap, umap  = get_sim(sim_idx, mode='actsim') 
    #tmap *= taper
    #qmap *= taper
    #umap *= taper

    #emap, bmap = cusps.util.qu2eb(qmap, umap)
    #emap, bmap = qu2eb(qmap, umap)

    #if sim_idx == 0:
    #    io.high_res_plot_img(tmap, output_path('tmap_unlen_%d.png'%sim_idx), down=3)
    #    io.high_res_plot_img(qmap, output_path('qmap_unlen_%d.png'%sim_idx), down=3)
    #    io.high_res_plot_img(umap, output_path('umap_unlen_%d.png'%sim_idx), down=3)
    #    io.high_res_plot_img(emap, output_path('emap_unlen_%d.png'%sim_idx), down=3)
    #    io.high_res_plot_img(bmap, output_path('bmap_unlen_%d.png'%sim_idx), down=3)

    add_spectra(tmap, None, None, deconv=False)
    #add_spectra(tmap, emap, bmap, deconv=True)

    #l, cl = power.get_raw_power(tmap, lmax=lmax)
    #lbin, cl = quick_binner.binned(l, cl)
    #st.add_to_stats("dltt_raw", cl2dl(lbin, cl)) 




    #lbin, cl = power.get_power(tmap, mcm_identifier, taper, taper, bin_edges, polcomb='TT', lmax=lmax, overwrite=overwrite)
    #lbin, cl = cusps_fc.get_power(tmap, polcomb='TT')
    #cl *= cusps.util.get_fsky(imap)
    #st.add_to_stats("dltt_decov", cl2dl(lbin, cl)) 

    #l, cl = get_flat_power(tmap)
    #st.add_to_stats('dltt_flat', cl2dl(lbin, cl))   
 

st.get_stats()

def add_with_err(plotter, st, l, key, **kwargs): 
    mean =  st.stats[key]['mean']
    err  =  st.stats[key]['errmean']
    if not np.isnan(err).any():
        plotter.add_err(l, mean, err, ls='--', alpha=0.5, marker='o', **kwargs)
    else:
        plotter.add_data(l, mean, ls='--', alpha=0.5, **kwargs)
        

if cmblens.mpi.rank == 0:
    log.info("plotting")
    polcombs = ['tt']#, 'te', 'ee', 'bb']
   
    for polcomb in polcombs:
        prefix  = 'dl%s' % polcomb
        #yscale = 'log' if polcomb is not 'te' else 'linear'
        plotter = cusps.visualize.plotter(yscale='linear')
        plotter.add_data(theo_bin['l'], theo_bin[prefix], label='Dl%s Binned Theory'%polcomb.upper()) 
        add_with_err(plotter, st, lbin, '%s_raw'%prefix, label='DlTT (CUSPS RAW)') 
        #add_with_err(plotter, st, lbin, 'dltt_flat', label='DlTT (FC)')
        #add_with_err(plotter, st, lbin, '%s_deconv'%prefix, label='Dl%s (CUSPS_DECOV)'%polcomb.upper())
        plotter.set_title('Curved Sky Dl_%s ' %polcomb.upper())
        plotter.set_xlabel(r'$l$') 
        plotter.set_ylabel(r'$Dl(l)$')
        plotter.set_xlim([0,5000])
        plotter.show_legends()
        plotter.save(output_path("%s_spec.png"%prefix))
 
    polcombs = ['tt']#, 'te', 'ee', 'bb']
    for polcomb in polcombs:
        prefix  = 'dl%s' % polcomb
        plotter = cusps.visualize.plotter(yscale='linear')
        add_with_err(plotter, st, lbin, 'frac%s_raw'%polcomb, label='Dl%s (CUSPS_raw)'%polcomb.upper())
        plotter.set_title('Fractional Difference Dl_%s ' %polcomb.upper())
        plotter.set_xlabel(r'$l$') 
        plotter.set_ylabel(r'$(sim - theo)/theo$')
        plotter.set_xlim([0,5000])
        plotter.set_ylim([-0.05,0.05])
        plotter.hline(y=0, color='k')
        #add_with_err(plotter, st, lbin, 'dltt_flat', label='DlTT (FC)')
        plotter.show_legends()
        plotter.save(output_path("frac_diff%s.png"%prefix))
