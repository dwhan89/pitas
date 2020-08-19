# -
# /math.py
# -

import pitas
import numpy as np
import warnings


def frac_diff(data, expected, mode='raw', suppress_nan=True):
    frac_diff = (data - expected) / expected

    if suppress_nan and np.isnan(frac_diff).any():
        num_nan = np.sum(np.isnan(frac_diff))
        warnings.warn("#%d NaNs are suppressed" % num_nan)
        warnings.warn("%f perc is nan" % (float(num_nan) / frac_diff.size * 100))
        frac_diff = np.nan_to_num(frac_diff)
    else:
        pass

    if mode == 'raw':
        pass
    elif mode == 'avg':
        frac_diff = np.mean(frac_diff)
    elif mode == 'perc':
        frac_diff = frac_diff * 100
    elif mode == 'avgperc':
        frac_diff = np.abs(np.mean(frac_diff) * 100.0)
    else:
        pass

    return frac_diff
