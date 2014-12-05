# -*- coding: utf-8 -*-
"""
Functions to copy psydat files

@author: Mehdi Rahim
"""

import os, glob, shutil


def copy_nii(subj, regex='S*.nii', dst_dir='raw'):
    """ Copy regex nii files to dst_dir
    """
    image_list = glob.glob(os.path.join(SRC_BASE_DIR, subj,
                                        'MRI', 'MID', regex))
                                        
    if len(image_list) > 0:
        print subj, len(image_list), image_list[0]
        dst_subj_dir = os.path.join(DST_BASE_DIR, subj)
        if not os.path.exists(dst_subj_dir):
            os.mkdir(dst_subj_dir)
                                        
        dst_file_dir = os.path.join(dst_subj_dir, dst_dir)
        if not os.path.exists(dst_file_dir):
            os.mkdir(dst_file_dir)
        for img in image_list:
            shutil.copy(img, dst_file_dir)


SRC_BASE_DIR = os.path.join('/', 'shfj', 'Ppsypim', 'PSYDAT', 'Subjects')
DST_BASE_DIR = os.path.join('/', 'media', 'CORSAIR', 'data',
                            'PSYDAT', 'Subjects')

subject_list = os.listdir(SRC_BASE_DIR)

for subj in subject_list:
    copy_nii(subj, regex='eseweaS*.nii', dst_dir='processed')
    