#-
# stats.py
#-
#
import pitas
import numpy as np, os, pickle as pickle

class STATS(object):

    def __init__(self, stat_identifier=None, output_dir=None, overwrite=False):
        self.output_dir  = output_dir if output_dir else os.path.join(pitas.config.get_default_output_dir(), 'stats')
        if pitas.mpi.rank == 0 : 
            print("[STATS] output_dir is %s" %self.output_dir)
            pitas.pitas_io.create_dir(self.output_dir)
        pitas.mpi.barrier()

        file_name        = "stats.pkl" if not stat_identifier else "stats_%s.pkl" %stat_identifier
        
        self.output_file = os.path.join(self.output_dir, file_name)
        self.storage     = {}
        self.stats       = {}
        self.tag         = 3235 # can this be randomly assigned 
        try:
            assert(not overwrite)
            self.storage = pickle.load(open(self.output_file, 'r'))
            if pitas.mpi.rank == 0: print("[STATS] loaded %s" %self.output_file)
        except:
            if pitas.mpi.rank == 0: print("[STATS] starting from scratch")

    def add_data(self, data_key, data_idx, data, safe=False):
        if not self.storage.has_key(data_key): self.storage[data_key] = {}
    
        if self.storage[data_key].has_key(data_idx) and safe:
            raise ValueError("[STATS] already have %s" %((data_key,data_idx)))
        else:
            self.storage[data_key][data_idx] = data

    def collect_data(self, dest=0):
        print("[STATS] collecting data")
        
        if pitas.mpi.is_mpion():
            pitas.mpi.transfer_data(self.storage, self.tag, dest=dest, mode='merge')
        self.tag += 1

    def has_data(self, data_key, data_idx):
        has_data = self.storage.has_key(data_key)
        return has_data if not has_data else self.storage[data_key].has_key(data_idx)
    
    def reload_data(self):
        ### passing all data through mpi is through. save it and reload it
        try: 
            self.storage = pickle.load(open(self.output_file, 'r'))
            if pitas.mpi.rank == 0: print("[STATS] loaded %s" %self.output_file)
        except:
            if pitas.mpi.rank == 0: print("[STATS] failed to reload data")

    def save_data(self, root=0, reload_st=True):
        self.collect_data()
       
        if pitas.mpi.rank == root:
            print("[STATS] saving %s from root %d" %(self.output_file, root))
            with open(self.output_file, 'w') as handle:
                pickle.dump(self.storage, handle)
        else: pass
        pitas.mpi.barrier()
        if reload_st: self.reload_data()

    def get_stats(self, subset_idxes=None, save_data=True):
        if save_data: self.save_data(reload_st=True)
    
        print("calculating stats")
        ret = {}
        for key in list(self.storage.keys()):
            if subset_idxes is None:
                ret[key] = stats(np.array(list(self.storage[key].values())))
            else: 
                ret[key] = stats(np.array(list(self.get_subset(subset_idxes, key, False).values())))

        self.stats = ret
        return ret

    def purge_data(self, data_idx, data_key=None):
        self.collect_data()
        
        def _purge(data_key, data_idx):
            print("[STATS] purging %s %d" %(data_idx, data_key)) 
            del self.storage[data_key][data_idx] 

        if pitas.mpi.rank == 0:
            if data_key is not None:
                _purge(data_key, data_idx)
            else:
                for data_key in list(self.storage.keys()):
                    _purge(data_key, data_idx) 

        self.reload_data()

    def get_subset(self, data_idxes, data_key=None, collect_data=False):
        if collect_data: self.collect_data()
        
        def _collect_subset(data_key, data_idxes):
            return dict((k, v) for k, v in iter(self.storage[data_key].items()) if k in data_idxes)
        
        ret = {}
        if data_key is not None:
            ret = _collect_subset(data_key, data_idxes)
        else:
            for data_key in list(self.storage.keys()):
                ret[data_key] = _collect_subset(data_key, data_idxes)

        return ret

def stats(data, axis=0, ddof=0):
    datasize = data.shape
    mean     = np.mean(data, axis=axis)
    cov      = np.cov(data.transpose(), ddof=ddof)
    cov_mean = cov/ float(datasize[0])
    corrcoef = np.corrcoef(data.transpose())
    std      = np.std(data, axis=axis, ddof=ddof) # use the N-1 normalization
    std_mean = std/ np.sqrt(datasize[0])

    return {'mean': mean, 'cov': cov, 'corrcoef': corrcoef, 'std': std, 'datasize': datasize\
            ,'std_mean': std_mean, 'cov_mean': cov_mean}

def chisq_pval(chisq, dof):
    from scipy.stats import chi2
    return 1 - chi2.cdf(chisq, dof)

def chisq(obs, exp, cov_input, ddof=None):
    ''' compute chisq 
        
        input
        obs  : observation
        exp  : expected value
        cov  : covariance 
        ddof : degree of freedom 

        oupput:
        chisq: computed chisq
        p    : p value
    '''
    from scipy.stats import chi2

    diff  = obs-exp if not (exp == 0.).all() else obs.copy()
   
    #print cov, diff
    cov  = cov_input.copy()
    norm = np.mean(np.abs(cov))
    cov  /= norm
    diff /= np.sqrt(norm) 
    #print cov, diff


    chisq = np.dot(np.linalg.pinv(cov), diff)
    chisq = np.dot(diff.T, chisq)

    if ddof is None: ddof = len(obs) - 1
    p     = chi2.sf(chisq, ddof)

    return chisq, p

def reduced_chisq(obs, exp, cov, ddof_cor=0.0):
    ''' calculate reduced chisq
        The default degree of freedom is k - 1 where k = # of observation.

        obs      : observation data
        fid      : expected value
        cov      : covariance
        ddof_cor : correction to the default degree of the freedom s.t. ddof = k-1-ddof_cor. Defaults to 0
    '''
    ddof     = len(obs)       # ddof
    ddof_cor = float(ddof_cor)
    ddof     = ddof - ddof_cor

    _chisq, p = chisq(obs,exp,cov,ddof)
    rchisq    = _chisq / ddof

    return rchisq, p




