import numpy as np

def calc_cross_corr (serie1, serie2):
    s0 = (serie1 - np.mean(serie1))/(np.std(serie1)*len(serie1))
    s1 = (serie2 - np.mean(serie2))/(np.std(serie2))
    corr = np.correlate(s0, s1, mode='full')
    # storing only the maximum coeficient - when series are in phase
    return max(corr)