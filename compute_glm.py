"""
    * Compute GLM for psydat subjects.
    Inputs are :
    - design matrix CSV 
    - fMRI filelist
    
    * Compute contrasts

"""
import os, glob
from collections import OrderedDict
import numpy as np
import pandas as pd
import nibabel as nib
from nipy.modalities.fmri import design_matrix
from nipy.modalities.fmri.glm import FMRILinearModel
from nilearn.plotting import plot_stat_map
import pypreprocess.reporting.glm_reporter as glm_reporter
import pypreprocess.reporting.base_reporter as base_reporter

DM_DIR = os.path.join('design_matrices')
"""
FMRI_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data',
                        'psydat', 'preprocess_output')
"""
FMRI_DIR = os.path.join('/', 'media', 'CORSAIR', 'data', 'psydat')


subject_path_list = sorted(glob.glob(os.path.join(FMRI_DIR, 'S*')))


for subject_path in subject_path_list[26:]:
    
    _, subj_id = os.path.split(subject_path)
    subj_id = subj_id[1:]
    print subj_id
    
    # load design matrix
    dm_file = os.path.join(DM_DIR, subj_id+'.csv')
    design_mat = design_matrix.dmtx_from_csv(dm_file)
    
    # load anat
    anat_file = glob.glob(os.path.join(FMRI_DIR, 'S' + subj_id,
                                       'anat',
                                       'w*.img'))[0]
    anat_img = nib.load(anat_file)
    anat = anat_img.get_data()
    anat_affine = anat_img.get_affine()
    
    
    # create 4D fmri
    fmri_files = sorted(glob.glob(os.path.join(FMRI_DIR,
                                               'S' + subj_id,
                                               'func',
                                               'wr*.nii')))
    img4d = []
    img_data = []
    for fmri_file in fmri_files:
        img = nib.load(fmri_file)
        img_data.append(img.get_data())
    img_data = np.array(img_data)
    img_data = img_data.transpose([1,2,3,0])
    img4d = nib.Nifti1Image(img_data, img.get_affine())
    
    # computer glm
    glm = FMRILinearModel(img4d, design_mat.matrix, mask='compute')
    glm.fit(do_scaling=True, model='ar1')
    
    # specify contrasts
    contrasts = {}
    n_columns = len(design_mat.names)
    
    # Specify and estimate all contrasts
    # 1- load contrasts into pandas dataframe
    contrast_df = pd.read_csv('contrasts.csv')
    
    # 2- insert zero-columns of derivatives and regressors
    # regressors
    for i in np.arange(28, n_columns):
        contrast_df.insert(len(contrast_df.keys()),
                           design_mat.names[i],
                           np.zeros(50))
    # derivatives
    for i in np.arange(15, 1, -1):
        contrast_df.insert(i, design_mat.names[2*i-3],
                           np.zeros(50))
    
    # 3- extract matrix
    contrast_mat = []
    for c in contrast_df.keys():
        if c != 'contrast':
            contrast_mat.append(contrast_df[c].values)
    contrast_mat = np.array(contrast_mat).T
    
    # 4- construct contrast dict
    contrasts = OrderedDict()
    for i in range(len(contrast_mat)):
        contrasts[contrast_df['contrast'][i]] = contrast_mat[i,:]
    
    stat_map = {}
    z_maps = {}
    # Compute all contrasts
    for contrast_id in contrasts.keys():
        z_map, t_map, eff_map, var_map = glm.contrast(contrasts[contrast_id],
                                                      con_id=contrast_id,
                                                      contrast_type='t',
                                                      output_z=True,
                                                      output_stat=True,
                                                      output_effects=True,
                                                      output_variance=True)
        stat_map['z_map'] = z_map
        stat_map['t_map'] = t_map
        stat_map['eff_map'] = eff_map
        stat_map['var_map'] = var_map
        
        #plot_stat_map(stat_map['z_map'], anat_file, title=contrast_id, threshold=3)
        
        c_id = contrast_df[contrast_df['contrast']==contrast_id].index[0]+1
        for map_type in stat_map.keys():
            map_dir = os.path.join(FMRI_DIR, 'S' + subj_id, 'maps')
            if not os.path.isdir(map_dir):
                os.mkdir(map_dir)
            map_type_dir = os.path.join(map_dir, map_type)
            if not os.path.isdir(map_type_dir):
                os.mkdir(map_type_dir)
            map_path = os.path.join(map_type_dir,
                                    'c' + "%03d" % (c_id) + '_' + contrast_id +'.nii.gz')
            
            if map_type == 'z_map':
                z_maps[contrast_id] = map_path
            nib.save(stat_map[map_type], map_path)
    
    
    # do stats report
    if not os.path.isdir(os.path.join(subject_path, 'reports')):
        os.mkdir(os.path.join(subject_path, 'reports'))
    stats_report_filename = os.path.join(subject_path, 'reports',
                                         "report_stats.html")
    glm_reporter.generate_subject_stats_report(
        stats_report_filename,
        contrasts,
        z_maps,
        glm.mask,
        design_matrices=[design_mat],
        subject_id='S'+subj_id,
        anat=anat,
        anat_affine=anat_affine,
        slicer='z',
        cut_coords=5,
        cluster_th=10  # we're only interested in this 'large' clusters
        )
