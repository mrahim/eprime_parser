"""
    Check .mat conditions and report empty ones
"""

import os, glob
import numpy as np
from scipy import io
import pandas as pd

BASE_DIR = os.path.join('eprime_files', 'mat')

file_list = glob.glob(os.path.join(BASE_DIR, 'S*_cond.mat'))
#con_list = pd.read_csv('contrasts.csv', names=['index', 'contrast', 'conditions'])

empty_cond = dict()
cpt = -1
for f in file_list:
    cpt += 1
    subj = os.path.split(f)[1].rsplit('_',1)[0]
    cond = io.loadmat(f)
    for i in np.arange(cond['onsets'].shape[1]):
        cond_name = str(cond['names'][0,i][0])
        if cond['onsets'][0,i][0,0] == 3600:
            if subj in empty_cond.keys():
                empty_cond[subj] += ', ' + cond_name
            else:
                empty_cond[subj] = cond_name