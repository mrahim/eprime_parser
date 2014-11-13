# -*- coding: utf-8 -*-
"""
Functions to copy psydat files

@author: Mehdi Rahim
"""

import os, glob, shutil

SRC_BASE_DIR = os.path.join('/', 'home', 'Ppsypim', 'PSYDAT', 'Subjects')
DST_BASE_DIR = os.path.join('/', 'media', 'CORSAIR', 'data',
                            'PSYDAT', 'Subjects')
                            

subject_list = os.listdir(SRC_BASE_DIR)


for subj in subject_list:
    image_list = glob.glob(os.path.join(SRC_BASE_DIR, subj,
                                        'MRI', 'MID', '*.nii'))
    print len(image_list)