import pitas
from enlib import enmap, resample, curvedsky
import healpy as hp
import os, numpy as np
from itertools import product
from orphics import io, maps
import sys
from actsims.simTools import resample_fft_withbeam
from scipy import ndimage
import seaborn as sns

# input from command line
pitas.config.argparser.add_argument('-mp', '--mpi',
    default='t',
    help='switch for mpi')

pitas.config.argparser.add_argument('-so', '--stat_overwrite',
    default='f',
    help='stats')

pitas.config.argparser.add_argument('-mo', '--mcm_overwrite',
    default='f',
    help='stats')

pitas.config.argparser.add_argument('-s', '--start',
    default=2, type=int,
    help='lower bound for the sims to be used')

pitas.config.argparser.add_argument('-e', '--end',
    default=200, type=int,
    help='upper bound for the sims to be used')

pitas.config.argparser.add_argument('-lm', '--lmin',
    default=0, type=int,
    help='lmin')

pitas.config.argparser.add_argument('-LM', '--lmax',
    default=5300, type=int,
    help='lmax')

pitas.config.argparser.add_argument('-w', '--weight',
    default='f',
    help='add weight to mask')

pitas.config.argparser.add_argument('-pt', '--ptsc',
    default='f',
    help='add ptsc mask')

pitas.config.argparser.add_argument('-sm', '--smooth',
    default=5, type=int,
    help='# sigma for guassian filter')

pitas.config.argparser.add_argument('-hf', '--highpass',
    default='f',
    help='use high pass')

pitas.config.argparser.add_argument('-kxky', '--kxky_filter',
    default='f',
    help='use kxky filter')

pitas.config.argparser.add_argument('-tf', '--transfer',
    default='f',
    help='transfer')

pitas.config.argparser.add_argument('-rs', '--resample',
    default='t',
    help='resample')

pitas.config.argparser.add_argument('-up', '--upsample',
    default='t',
    help='upsampling')

pitas.config.argparser.add_argument('-be', '--beam',
    default='f',
    help='beam')

pitas.config.argparser.add_argument('-px', '--pixwind',
    default='f',
    help='pixel window')

pitas.config.argparser.add_argument('-b', '--bin',
    default='UNIFORM',
    help='binfile')

args = pitas.config.argparser.parse_args()


start = args.start
end   = args.end

assert(end>=start)

# initialize mpi
pitas.mpi.init(pitas.util.str2bool(args.mpi))
subtasks = pitas.mpi.taskrange(imin=start, imax=end)

mcm_overwrite  = pitas.util.str2bool(args.mcm_overwrite)
plot_maps      = True
plot_only      = False
stat_overwrite = pitas.util.str2bool(args.stat_overwrite)

postfix      = pitas.util.parse_argparse(args, exclude=['mpi', 'end', 'mcm_overwrite', 'stat_overwrite']) 
postfix      += '_louis_taper_wider_binning_fix2_upsampling'
output_dir   = os.path.join('./', postfix)
output_path  = lambda x: os.path.join(output_dir, x)
if pitas.mpi.rank == 0:
    pitas.pitas_io.create_dir(output_dir)
pitas.mpi.barrier()

beam_dir     = os.path.join(pitas.config.get_resource_dir(), 'beams_pa12_2013_2014_160201')
beam_path    = lambda x : os.path.join(beam_dir, 'beam_tform_160201_2014_pa{}_jitter_CMB_deep56.txt'.format(x))
weight_dir   = '/global/project/projectdirs/act/data/synced_maps/mr3' 
sim_dir      = '/global/cscratch1/sd/engelen/simsS1516_v0.3/data/'
taper_dir    = '/global/cscratch1/sd/dwhan89/temp/testWindows'
ptsc_dir     = '/global/project/projectdirs/act/data/synced_maps/mr2_20170608_20170816/masks/c7v5_LAMBDA'
array_idxes  = [2]
cmb_idxes    = ['I', 'Q', 'U']
polcombs     = ['tt', 'te', 'ee', 'bb']
coords       = np.array([[-7,-8],[4,40.0]])
coords       *= np.pi/180.
lmin         = args.lmin
lmax         = args.lmax
unit         = 'camb'
TCMB         = pitas.constants.TCMB
bin_file     = args.bin
kspace_cuts  = (90, 50, 0, None) #(kx, ky, lmin, lmax)
smooth       = args.smooth
upsample     = pitas.util.str2bool(args.upsample)
resample     = pitas.util.str2bool(args.resample)
highpass     = pitas.util.str2bool(args.highpass)
kxky_filter  = pitas.util.str2bool(args.kxky_filter)
apply_weight = pitas.util.str2bool(args.weight)
apply_ptsc   = pitas.util.str2bool(args.ptsc)
apply_beam = pitas.util.str2bool(args.beam)
apply_pixwind = pitas.util.str2bool(args.pixwind)
apply_transfer = pitas.util.str2bool(args.transfer)

# frankenstein it
if bin_file == 'PITAS': 
    bin_edges = pitas.util.get_default_bin_edges(lmax)
elif bin_file == 'UNIFORM':
    nbin      = lmax/100
    bin_edges = np.linspace(2, lmax, nbin) 
else:
    bin_file    = os.path.join(pitas.config.get_resource_dir(), bin_file)
    bin_edges    = pitas.flipper_tools.read_bin_edges(bin_file)
    temp         = bin_edges[1:]-bin_edges[:-1]
    temp_len     = len(np.where(temp<=135)[0])
    bin_edges    = np.unique(np.concatenate((bin_edges[:temp_len], np.arange(max(bin_edges[:temp_len]),lmax+100,100))))
    del temp, temp_len

highpass1D   = None
lbin         = None
taper        = None
template_map = None
ptsc_mask    = None
pixwindow    = None
masks        = {1:None, 2:None}
masks_pol    = {1:None, 2:None}
weights      = {1:None, 2:None}
theos        = {1:None, 2:None}
modlmap      = None
wcs, shape   = None, None
theo         = pitas.theory.load_theory_cls('cosmo2017_10K_acc3_lensed', 'camb')

########## helper functions #############
class MapHandler(object):
    def __init__(self, array_idx, sim_dir=sim_dir, weight_dir=weight_dir, taper_dir=taper_dir, ptsc_dir=ptsc_dir, coords=coords, lmin=lmin, lmax=lmax, kspace_cuts=kspace_cuts):
        self.array_idx   = array_idx
        self.sim_dir     = sim_dir
        self.weight_dir  = weight_dir
        self.taper_dir   = taper_dir
        self.ptsc_dir    = ptsc_dir
        self.weight_temp = os.path.join(weight_dir, 's14_deep56_pa{}_f150_nohwp_night_4way_coadd_hits.fits')
        self.coords      = coords
        self.lmin        = lmin
        self.lmax        = lmax
        self.kspace_cuts = kspace_cuts

    def get_iqu_sims(self, sim_idx, unit='uk'):
        oshape, owcs = enmap.fullsky_geometry(res=1.0*np.pi/180./60.)
        zpsim_idx    = '%05d' %sim_idx

        input_file_temp = os.path.join(self.sim_dir, 'cmb_set00_{}/fullskyLensedMapUnaberrated_{}_{}.fits')

        transfer = np.array(pitas.act_analysis.get_beam_transfer(beam_path(ar_idx))).T
        ret = {}
        for cmb_type in ['T', 'Q', 'U']:
            input_file = input_file_temp.format(zpsim_idx, cmb_type, zpsim_idx)
            key = 'I' if cmb_type == 'T' else cmb_type
            ret[key] = enmap.read_fits(input_file, box=coords)
            
            
            nshape, nwcs = enmap.scale_geometry(ret[key].shape, ret[key].wcs, 2) if upsample else (oshape, owcs)                

            if nshape != ret[key].shape:
                print 'resampling ...'
                if resample:
                    ndata    = resample_fft_withbeam(ret[key], nshape, None, apply_beam, transfer, apply_pixwind)
                    ret[key] = enmap.enmap(ndata, nwcs)
                else:
                    ret[key] = pitas.util.interp_enmap(ret[key], nshape, nwcs)
        if unit == 'none':
            TCMB = cmblens.constants.TCMB
            for key in ret.keys():
                ret[key].data /= TCMB
        else: pass

        return (ret['I'], ret['Q'], ret['U'])
    
    def get_taper(self):
        global taper
        if taper is None:
            #taper = enmap.read_fits(taper_file, box=self.coords)
            temp    = self.get_template_map()
            ndata,_ = maps.get_taper(temp.shape)
            wcs   = temp.wcs.copy()
            taper = enmap.enmap(ndata, wcs)
        else: pass
        return taper.copy()

    def get_template_map(self):
        global template_map
        if template_map is None:
            #temp = self.get_taper().copy()
            temp,_,_ = self.get_iqu_sims(0)
            template_map = enmap.empty(temp.shape, temp.wcs)
        else: pass
        return template_map.copy()
    
    def get_mapinfo(self):
        global wcs, shape
        if wcs is None or shape is None:
            temp = self.get_template_map()
            wcs, shape = temp.wcs, temp.shape
        else: pass
        return (wcs.copy(), shape)
    
    def get_prepared_iqu(self, sim_idx): 
        imap, qmap, umap = self.get_iqu_sims(sim_idx)
        #imap, qmap, umap = self.beam2tqu(imap,qmap,umap,mode='deconv',lmax=self.lmax)       
        if highpass == True: imap, qmap, umap = self.hp2tqu(imap, qmap, umap) 

        ret  = {'imap':imap, 'qmap':qmap, 'umap': umap}    
        mask_temp = self.get_mask_temp()
        mask_pol  = self.get_mask_pol()
        for key in ret.keys():
            kfilter   = self.get_filter(key)
            if kfilter is not None: 
                print 'filtering..'
                ret[key]  = maps.filter_map(ret[key], kfilter) 
        
        ret['imap'] = ret['imap']*mask_temp 
        ret['qmap'] = ret['qmap']*mask_pol
        ret['umap'] = ret['umap']*mask_pol

        return (ret['imap'], ret['qmap'], ret['umap'])

    def hp2tqu(self, tmap, qmap, umap):
        filt   = self.get_highpass1D().copy()
        l_filt = np.arange(self.lmax+1)
        filt   = filt[:self.lmax+1]

        ret = [tmap, qmap, umap]

        owcs,shape = (ret[0].wcs.copy(), ret[0].shape)
        tqu        = np.zeros((3,)+ret[0].shape)
        tqu[0], tqu[1], tqu[2] = (ret[0], ret[1], ret[2])

        tqu     = enmap.enmap(tqu, owcs)
        alm     = curvedsky.map2alm(tqu, lmax=self.lmax)
        del ret

        for idx in range(alm.shape[0]):
            alm[idx] = hp.sphtfunc.almxfl(alm[idx], filt)

        tqu= curvedsky.alm2map(alm, tqu)

        return (tqu[0], tqu[1], tqu[2]) # tqu 


    def get_filter(self, cmb_type): 
        temp = self.get_template_map()
        shape, wcs = temp.shape, temp.wcs

        kfilter = None
        if kxky_filter:
            kxcut, kycut, lmin, lmax = self.kspace_cuts
            kmask                    = maps.mask_kspace(shape, wcs, kxcut, kycut, lmin, lmax)
            kfilter                  = kmask
        else: pass
        
        if apply_pixwind:
            pixwind = self.get_pixwindow(mode='deconv')
            if kfilter is None: kfilter=pixwind
            else: kfilter *= pixwind

        return kfilter

    def get_pixwindow(self, mode='deconv'):
        assert(mode in ['conv', 'deconv'])
        global pixwindow
        
        wfact = 1. if mode == 'conv' else -1.
        if pixwindow is None:
            shape  = self.get_template_map().shape
            wy, wx = enmap.calc_window(shape)
            pixwindow = wy[:,None]*wx[None,:]
        else: pass

        return pixwindow.copy()**wfact

    def beam2tqu(self, tmap, qmap, umap, mode='deconv', lmax=lmax):
        assert(mode in ['conv', 'deconv'])

        l_beam, f_beam = pitas.act_analysis.get_beam_transfer(beam_path(self.array_idx))

        loc = np.where(l_beam <= lmax)
        beam_fact = f_beam
        if mode is 'deconv': beam_fact = 1./f_beam

        ret = [tmap, qmap, umap]

        owcs,shape = (ret[0].wcs.copy(), ret[0].shape)
        tqu        = np.zeros((3,)+ret[0].shape)
        tqu[0], tqu[1], tqu[2] = (ret[0], ret[1], ret[2])

        tqu     = enmap.enmap(tqu, owcs)
        alm     = curvedsky.map2alm(tqu, lmax=lmax)
        del ret

        for idx in range(alm.shape[0]):
            alm[idx] = hp.sphtfunc.almxfl(alm[idx], beam_fact)

        tqu= curvedsky.alm2map(alm, tqu)

        return (tqu[0], tqu[1], tqu[2]) # tqu 


    def get_modlmap(self):
        global modlmap
        if modlmap is None:
            modlmap = self.get_template_map().modlmap()
        else: pass
        return modlmap.copy()

    def get_prepared_teb(self, sim_idx):
        imap, qmap, umap = self.get_prepared_iqu(sim_idx)
        return pitas.util.tqu2teb(imap, qmap, umap, lmax=self.lmax)

    def get_ptsc_mask(self):
        global ptsc_mask 
        if ptsc_mask is None:
            mask_file = os.path.join(self.ptsc_dir, 'mask_05.00arcmin_0.015Jy_new_car.fits')
            print '[MH] loading %s' %mask_file
            
            nwcs, nshape = self.get_mapinfo()
            ptsc_mask = enmap.read_fits(mask_file, box=enmap.box(nshape, nwcs))

            ptsc_mask = pitas.util.interp_enmap(ptsc_mask, nshape, nwcs)
        return ptsc_mask.copy()

    def get_weight(self):
        global weights
        if weights[self.array_idx] is None:
            weight_file     = self.weight_temp.format(self.array_idx)
            print '[MH] loading %s' %weight_file 
            nwcs, nshape = self.get_mapinfo()
            weight          = enmap.read_fits(weight_file, box=enmap.box(nshape, nwcs))

            weight       = pitas.util.interp_enmap(weight, nshape, nwcs)
            weights[self.array_idx] = weight
        else: pass
        return weights[self.array_idx]

    def get_mask_temp(self, blur=5):
        global masks
        if masks[self.array_idx] is None:
            mask     = self.get_taper()
            owcs     = mask.wcs
           
            if apply_ptsc:   mask *= self.get_ptsc_mask()
            if apply_weight: mask *= self.get_weight()

            owcs = mask.wcs
            if blur>0: mask = enmap.enmap(ndimage.gaussian_filter(np.array(mask), sigma=blur), owcs)
            masks[self.array_idx] = mask
        else: pass
        return masks[self.array_idx].copy()

    def get_mask_pol(self, blur=5):
        global masks_pol
        if masks_pol[self.array_idx] is None:
            mask     = self.get_taper()
            owcs     = mask.wcs
           
            if apply_weight: mask *= self.get_weight()

            owcs = mask.wcs
            if blur>0: mask = enmap.enmap(ndimage.gaussian_filter(np.array(mask), sigma=blur), owcs)
            masks_pol[self.array_idx] = mask
        else: pass
        return masks_pol[self.array_idx].copy()
    
    def get_highpass1D(self, lmin=lmin, lmax=500):
        global highpass1D
        if highpass1D is None:
            l_trim = np.ceil(np.max(self.get_modlmap()))
            _, highpass1D = pitas.util.cos_highpass(500, lmin, l_trim)
        else: pass
        return highpass1D.copy()

    def get_transfer(self):
        l_tran   = np.arange(self.lmax+1)
        transfer = np.ones(len(l_tran)) 

        if apply_transfer:
            if kxky_filter: 
                _, trans_kxky = pitas.util.get_transfer_kxky_filter(kspace_cuts[0], kspace_cuts[1], lmax)
                transfer      *= trans_kxky
            else: pass

            if highpass:
                trans_hf      = MH.get_highpass1D().copy()
                transfer      *=trans_hf
            else: pass
        else: pass

        return (l_tran, transfer)

ident_postfix = postfix
st = pitas.stats.STATS(stat_identifier='d56_realmask%s' %ident_postfix, output_dir=output_dir, overwrite=stat_overwrite)
for ar_idx in array_idxes:
    MH             = MapHandler(ar_idx)
    mask_temp      = MH.get_mask_temp()
    mask_pol       = MH.get_mask_pol()

    if apply_beam: transfer = pitas.act_analysis.get_beam_transfer(beam_path(ar_idx))
    else: transfer = MH.get_transfer()

    mcm_identifier = "092018_d56_sim_ar{}_ar{}{}".format(ar_idx, ar_idx, ident_postfix)
    pitas_lib      = pitas.power.PITAS(mcm_identifier, mask_temp, mask_pol, bin_edges, lmax=lmax, transfer=transfer, overwrite=mcm_overwrite)
    lbin           = pitas_lib.bin_center

    theo_bin, l_th = {}, theo['l']
    lbin_th, theo_bin['dltt']    = pitas_lib.bin_theory_scalarxscalar(l_th, theo['dltt'])
    lbin_th, theo_bin['dlte']    = pitas_lib.bin_theory_scalarxvector(l_th, theo['dlte'])
    _, dlee_th, dleb_th, dlbb_th = pitas_lib.bin_theory_pureeb(l_th, theo['dlee'], np.zeros(len(l_th)), theo['dlbb'])

    lbin_th, theo_bin['cltt']    = pitas_lib.bin_theory_scalarxscalar(l_th, theo['cltt'])
    lbin_th, theo_bin['clte']    = pitas_lib.bin_theory_scalarxvector(l_th, theo['clte'])
    _, clee_th, cleb_th, clbb_th = pitas_lib.bin_theory_pureeb(l_th, theo['clee'], np.zeros(len(l_th)), theo['clbb'])

    theo_bin['l'] = lbin_th
    theo_bin['dlee'], theo_bin['dleb'], theo_bin['dlbb'] = (dlee_th, dleb_th, dlbb_th)
    theo_bin['clee'], theo_bin['cleb'], theo_bin['clbb'] = (clee_th, cleb_th, clbb_th)

    cl2dl_facts = {}
    cl2dl_facts['tt'] = pitas.util.get_cl2dl_factor(theo_bin['cltt'], theo_bin['dltt'])
    cl2dl_facts['te'] = pitas.util.get_cl2dl_factor(theo_bin['clte'], theo_bin['dlte'])
    cl2dl_facts['ee'] = pitas.util.get_cl2dl_factor(theo_bin['clee'], theo_bin['dlee'])
    cl2dl_facts['bb'] = pitas.util.get_cl2dl_factor(theo_bin['clbb'], theo_bin['dlbb'])

    theos[ar_idx] = theo_bin.copy()

    for sim_idx in subtasks:
        
        idx_temp = "dl{}_ar%d" %ar_idx

        if st.has_data(idx_temp.format('bb'), sim_idx): continue
        
        print "processing ar{}| sim_idx{}".format(ar_idx, sim_idx) 
        tmap, emap, bmap = MH.get_prepared_teb(sim_idx)
 
        lbin, cltt  = pitas_lib.get_power(tmap, polcomb='TT')
        lbin, clte = pitas_lib.get_power(tmap, emap, polcomb='TE') 
        lbin, clee, cleb, clbb = pitas_lib.get_power_pureeb(emap, bmap)


        st.add_data(idx_temp.format('tt'), sim_idx, cl2dl_facts['tt']*cltt)
        st.add_data(idx_temp.format('te'), sim_idx, cl2dl_facts['te']*clte)
        st.add_data(idx_temp.format('ee'), sim_idx, cl2dl_facts['ee']*clee)
        st.add_data(idx_temp.format('bb'), sim_idx, cl2dl_facts['bb']*clbb)
        
        if plot_maps and pitas.mpi.rank == 0: 
            _, qmap, umap = MH.get_prepared_iqu(sim_idx)
            io.high_res_plot_img(tmap, output_path('ptmap1.png'), down=3)
            io.high_res_plot_img(emap, output_path('pemap1.png'), down=3)
            io.high_res_plot_img(bmap, output_path('pbmap1.png'), down=3) 
            io.high_res_plot_img(qmap, output_path('pqmap1.png'), down=3)
            io.high_res_plot_img(umap, output_path('pumap1.png'), down=3) 
            io.high_res_plot_img(MH.get_mask_temp(), output_path('mask.png'), down=3)
            io.high_res_plot_img(MH.get_mask_pol(), output_path('mask_pol.png'), down=3)
            plot_maps = False
        else: pass

for key in st.storage.keys():
    if 'fracdiff' in key: continue
    for sim_idx in st.storage[key].keys():
        new_key = key + "_fracdiff"
        frac_diff = pitas.pitas_math.frac_diff(st.storage[key][sim_idx], theos[2][key[:4]])
        st.add_data(new_key, sim_idx, frac_diff)
 
st.get_stats(save_data=True)

lmin = -50
lmax = 5000

if pitas.mpi.rank == 0:
    for ar_idx in array_idxes:
        sns.set_style('white')
        idx_temp = "dl{}_ar%d" %(ar_idx)
        for polcomb in polcombs:
            st_idx  = idx_temp.format(polcomb)
            prefix  = 'dl%s' % polcomb
            yscale  = 'linear'
            plotter = pitas.visualize.plotter(xscale='linear', yscale=yscale)
            plotter.add_data(lbin, theos[ar_idx][prefix], label='Dl%s Theory'%polcomb.upper())

            mean, err = st.stats[st_idx]['mean'], st.stats[st_idx]['std_mean']
            plotter.add_err(lbin, mean, err, marker='o', alpha=0.8, label="Dl_%s PITAS"%polcomb.upper())

            plotter.vline(100, ls='--', color='k')
            plotter.set_title('Curved Sky Dl_%s ' %polcomb.upper())
            plotter.set_xlabel(r'$l$')
            plotter.set_ylabel(r'$Dl(l)$')
            plotter.set_xlim([lmin,lmax])
            plotter.show_legends()
            plotter.save(output_path("%s_spec.png"%(st_idx)))


        idx_temp = "dl{}_ar%d_fracdiff" %(ar_idx)
        for polcomb in polcombs:
            st_idx  = idx_temp.format(polcomb)
            plotter = pitas.visualize.plotter()
            mean, err = st.stats[st_idx]['mean'], st.stats[st_idx]['std_mean']
            plotter.add_err(lbin, mean, err, marker='o', alpha=0.8, label="Dl_%s PITAS"%polcomb.upper())
            plotter.set_title('Fractional Difference Dl_%s ' %polcomb.upper())
            plotter.set_xlabel(r'$l$', fontsize=22)
            plotter.set_ylabel(r'$(sim - theo)/theo$', fontsize=22)
            plotter.set_xlim([lmin,lmax])
            plotter.hline(y=0, color='k')
            plotter.show_legends(fontsize=18)
            plotter.set_ylim([-0.05,0.05])
            plotter.hline(0.01, ls='--', color='k')
            plotter.hline(-0.01, ls='--', color='k')
            plotter.vline(100, ls='--', color='k')
            plotter.show_legends()
            plotter.save(output_path("%s_spec.png"%(st_idx)))
