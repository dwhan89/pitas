#-
# util.py
#-
#
from enlib import enmap, curvedsky
import healpy as hp, numpy as np

############ fsky ###################
def get_fsky(emap):
    area = emap.area() #in rad sq#
    fsky = area / (4.*np.pi)
    return fsky


############ spectra ################
def get_spectra(emap1, emap2=None, lmax=5000):
    alm1 = curvedsky.map2alm(emap1, lmax=lmax).astype(np.complex128)
    alm2 = alm1 if emap2 is None else curvedsky.map2alm(emap2, lmax=lmax).astype(np.complex128)

    cl  = hp.alm2cl(alm1, alm2, lmax=lmax)
    l   = np.arange(len(cl))

    return (l, cl)

def check_None(*args):
    for arg in args:
        assert(arg is not None)
    return True

def cl2dl(l, cl):
    return cl*(l*(l+1.))/(2*np.pi)


############# maps #################
def tqu2teb(tmap, qmap, umap, lmax):
    tqu     = np.zeros((3,)+tmap.shape)
    tqu[0], tqu[1], tqu[2] = (tmap, qmap, umap)
    tqu     = enmap.enmap(tqu, tmap.wcs)
    alm     = curvedsky.map2alm(tqu, lmax=lmax)

    teb     = curvedsky.alm2map(alm[:,None], tqu.copy()[:,None], spin=0)[:,0]
    del tqu

    return (teb[0], teb[1], teb[2]) #tmap, emap, bmap

############# misc #################

def get_from_dict(nested_dict, keys, safe=True):
    if not type(keys) is tuple: keys = (keys,)
    if safe and not has_key(nested_dict, keys): return None

    if(len(keys) > 1):
        return get_from_nested_dict(nested_dict[keys[0]], keys[1:], False)
    else:
        return nested_dict[keys[0]]


def has_key(nested_dict, keys):
    ''' search through nested dictionary to fine the elements '''
    if not type(keys) is tuple: keys = (keys,)
    if not type(nested_dict) == dict: return False

    if(len(keys) > 1):
        has_it = nested_dict.has_key(keys[0])
        return has_key(nested_dict[keys[0]], keys[1:]) if has_it else False
    else:
        return nested_dict.has_key(keys[0])

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise TypeException("Can't convert 'str' object to 'boolean'")

def merge_dict(a, b, path=None, clean = True):
    '''
    merges b into a

    ref: https://stackoverflow.com/questions/7204805/dictionaries-of-dictionaries-merge"
    '''
    if path is None: path = []

    for key in b:
        if key in a:
            if isinstance(a[key], dict):
                if isinstance(b[key], dict):
                    merge_dict(a[key], b[key], path + [str(key)])
                else:
                    a[key] = b[key]
            elif isinstance(a[key], np.ndarray) and np.array_equal(a[key], b[key]):
                pass
            elif a[key] == b[key]:
                pass # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]

    if clean: del b

    return a

def get_default_bin_edges(lmax):
    bin_edges = None
    if lmax <= 400:
        bin_edges = np.linspace(0,400,20)
    else:
        bin_num   = (lmax-400)/100
        if bin_num == 0: bin_num = 1
        bin_edges = np.concatenate((np.linspace(0,400,20), np.linspace(400, lmax, bin_num)))
        bin_edges = np.unique(bin_edges)
    return bin_edges




