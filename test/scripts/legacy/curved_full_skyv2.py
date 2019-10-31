#-
# cmblens_lens_map.py
#-
# test making flat periodic lensed maps
#

import cmblens, cusps
import cmblens.flipper as flipper
from orphics import maps, io, stats
from itertools import product
import numpy as np, pickle as pickle
import os, mapTools, sys
import fftPol
from actsims import simTools as act_sim
from enlib import enmap, curvedsky
from cusps import power

log = cmblens.logger.getLogger()

# input from command line
cmblens.config.argparser.add_argument('-mp', '--mpi',
    default='t',
    help='switch for mpi')

cmblens.config.argparser.add_argument('-s', '--start',
    default=0, type=int,
    help='lower bound for the sims to be used')

cmblens.config.argparser.add_argument('-e', '--end',
    default=319, type=int,
    help='upper bound for the sims to be used')

args = cmblens.config.argparser.parse_args()

start = args.start
end   = args.end

assert(end>=start)

# trigger mpi
cmblens.mpi.init(cmblens.util.str2bool(args.mpi))
subtasks = cmblens.mpi.taskrange(imin=start, imax=end)

# directory
post_fix        = 'fullsky'
output_dir      = os.path.join('/global/homes/d/dwhan89/shared/outbox/cori/curved_sim_test', post_fix)
output_path     = lambda x: os.path.join(output_dir, x)
sim_dir         = '/global/cscratch1/sd/engelen/simsS1516_v0.2/data'
sim_file_temp = os.path.join(sim_dir, 'cmb_set00_{}/fullsky{}Map_{}_{}.fits')

if cmblens.mpi.rank == 0:
    cmblens.util.create_dir(output_dir)
else: pass
cmblens.mpi.barrier()

# sim variable

# misc
lmin      = 0
lmax      = 6000
map_types = ['Lensed', 'Unlensed']
cmb_types = ['T', 'E', 'B']
bin_edges = np.linspace(0, lmax, 60)
bin_centers = (bin_edges[:-1]+bin_edges[1:])/2.
binner    = stats.bin1D(bin_edges)
TCMB      = cmblens.constants.TCMB

mcm_identifier = "%lsd_le%d_nb%d_lm%d" %(0, lmax, 60, lmax)

# input theory to compare
theo     = cmblens.delensing.theory.load_theory_cls('cosmo2017_10K_acc3_cmb', unit='camb') 
l_th = theo['l']

ltheo_bin = {}
for key in list(theo['lensed'].keys()):
    if key == 'l': continue
    llbin, lclbin = binner.binned(l_th, theo['lensed'][key])
    ltheo_bin['l']  = llbin
    ltheo_bin[key]  = lclbin

utheo_bin = {}
for key in list(theo['unlensed'].keys()):
    if key == 'l': continue
    ulbin, uclbin = binner.binned(l_th, theo['unlensed'][key])
    utheo_bin['l']  = ulbin
    utheo_bin[key] = uclbin

theo_bin = {'lensed': ltheo_bin, 'unlensed': utheo_bin}

### helper functions
def add_raw_spec(st, key, idx, map1, map2=None):
    cl   = None
    lbin = bin_centers
    if not st.has_data(key, idx):
        l, cl = power.get_raw_power(map1, emap2=map2, lmax=lmax, normalize=False)
        cl  = np.nan_to_num(cl)
        cl  = np.nan_to_num(cl)
        lbin, cl = binner.binned(l, cl)

        st.add_data(key, idx, cl)
    else:
        cl = st.storage[key][idx]

    return (lbin, cl) 

def add_frac_diff(st, key, idx, clbin, clbin_th):
    frac_diff = cmblens.cmblens_math.frac_diff(clbin, clbin_th)

    if not st.has_data(key, idx):
        st.add_data(key, idx, frac_diff)
    else:
        pass

def add_diff(st, key, idx, cl_len, cl_delen): 
    if not st.has_data(key, idx):
        st.add_data(key, idx, cl_len-cl_delen)
    else:
        pass
    
    return None

####### flags 
overwrite       = False
plot_lensed     = True
plot_unlensed   = True

#########################33

st = cmblens.stats.STATS(stat_identifier=mcm_identifier)
lbin = None
for sim_idx in subtasks: 
    log.info("processing: %d" %sim_idx)

    if st.has_data('clbb_lensed_frac', sim_idx): continue

    zpsim_idx = '%05d' %sim_idx
    for map_type, cmb_type in product(map_types, cmb_types):
        log.info("processing %s" %(((map_type,cmb_type),))) 

        file_name = sim_file_temp.format(zpsim_idx, map_type, cmb_type, zpsim_idx)
        log.info("loading %s" %file_name)

        emap = enmap.read_fits(file_name)
         
        theo_key = 'cl'+(cmb_type.lower()*2)
        stat_key = theo_key + '_' + map_type.lower()
        frac_key = stat_key + '_frac'

        lbin, clbin = add_raw_spec(st, stat_key, sim_idx, emap)
        add_frac_diff(st, frac_key, sim_idx, clbin, theo_bin[map_type.lower()][theo_key])

        del emap
        # load unlensed
            
    cmblens.mpi.barrier()
    ret = st.get_stats()


lbin = bin_centers
if cmblens.mpi.rank == 0:
    log.info("plotting")

    panel2ps    = {0: 'cltt', 1: 'clte', 2: 'clee', 3: 'clbb'} 
    spec2color  = {'lensed': 'k', 'unlensed': 'b', 'delensed': 'g', 'perfect': 'k', 'camb': 'k', 'camb_filt': 'r'}

    def cl2dl(cl, l=l_th, conv=TCMB**2.):
        return cl*(l*(l+1.))/(2*np.pi) * conv

    def add_err(plotter, lbin, st, idx, st_idx, **kwargs):
        mean =  cl2dl(st.stats[st_idx]['mean'], lbin)
        err  =  cl2dl(st.stats[st_idx]['std_mean'], lbin)
        plotter.add_err(idx, lbin, mean, err, **kwargs)

    def add_data(plotter, lbin, st, idx, st_idx, **kwargs):
        mean =  cl2dl(st.stats[st_idx]['mean'], lbin)
        plotter.add_data(idx, lbin, mean, **kwargs)

    def add_with_err(plotter, st, l, key, **kwargs):
        mean =  st.stats[key]['mean']
        err  =  st.stats[key]['std_mean']
        if not np.isnan(err).any():
            plotter.add_err(l, mean, err, ls='--', alpha=0.5, marker='o', **kwargs)
        else:
            plotter.add_data(l, mean, ls='--', alpha=0.5, **kwargs)

    colors  = cmblens.util.create_dict(['TT', 'EE', 'TE', 'EB', 'BB'])
    colors['TT'] = 'b' 
    colors['EE'] = 'g' 
    colors['TE'] = 'c' 
    colors['EB'] = 'y' 
    colors['BB'] = 'm' 

    if plot_lensed:
        
        for polcom in ['tt', 'ee', 'bb']:
            plotter = cmblens.visualize.plotter(yscale='linear') 
            color_idx = polcom.upper()
            add_with_err(plotter, st, lbin, 'cl%s_lensed' %polcom, color=colors[color_idx], label='%s sim'%polcom.upper()) 
            plotter.add_data(lbin, ltheo_bin['cl%s'%polcom], label='theory', color='k', alpha=0.5) 
            plotter.show_legends()
            plotter.set_ylabel(r'$Cl(l)$')
            plotter.set_xlabel(r'$l$')
            plotter.set_xlim([0, 5000])
            plotter.save(output_path("lraw_spec_%s_%d.png" %(polcom, lmin)))

        for polcom in ['tt',  'ee', 'bb']:
            plotter = cmblens.visualize.plotter(yscale='linear') 
            color_idx = polcom.upper()
            add_with_err(plotter, st, lbin, 'cl%s_lensed_frac'%polcom, label='%s' %polcom.upper())
            plotter.show_legends()
            plotter.set_ylabel(r'$(sim-theo)/theo$')
            plotter.set_xlabel(r'$l$')
            plotter.set_ylim([-0.05, 0.05])
            plotter.set_xlim([0, 5000])
            plotter.hline(y=0, color='k', ls='--')
            plotter.save(output_path("lraw_spec_frac_%s_%d.png" %(polcom, lmin)))


    if plot_unlensed:
        for polcom in ['tt', 'ee', 'bb']:
            plotter = cmblens.visualize.plotter(yscale='linear') 
            
            color_idx = polcom.upper()
            add_with_err(plotter, st, lbin, 'cl%s_unlensed' %polcom, color=colors[color_idx], label='%s sim'%polcom.upper()) 
            plotter.add_data(lbin, utheo_bin['cl%s'%polcom], label='theory', color='k', alpha=0.5) 
            plotter.show_legends()
            plotter.set_ylabel(r'$Cl(l)$')
            plotter.set_xlabel(r'$l$')
            plotter.set_xlim([0, 5000])
            plotter.save(output_path("uraw_spec_%s_%d.png" %(polcom, lmin)))

        for polcom in ['tt', 'ee', 'bb']:
            plotter = cmblens.visualize.plotter(yscale='linear') 
            color_idx = polcom.upper()
            add_with_err(plotter, st, lbin, 'cl%s_unlensed_frac'%polcom, label='%s' %polcom.upper())
            plotter.show_legends()
            plotter.set_ylabel(r'$(sim-theo)/theo$')
            plotter.set_xlabel(r'$l$')
            plotter.set_ylim([-0.05, 0.05])
            plotter.set_xlim([0, 5000])
            plotter.hline(y=0, color='k', ls='--')
            plotter.save(output_path("uraw_spec_frac_%s_%d.png" %(polcom, lmin)))

