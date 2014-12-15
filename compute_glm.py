"""
    Compute GLM for psydat subjects.
    Inputs are :
    - design matrix CSV 
    - fMRI filelist

"""
import os, glob
import numpy as np
import nibabel as nib
from nipy.modalities.fmri import design_matrix
from nipy.modalities.fmri.glm import FMRILinearModel

DM_DIR = os.path.join('design_matrices')
FMRI_DIR = os.path.join('/', 'disk4t', 'mehdi', 'data',
                        'psydat', 'preprocess_output')

subj_id = '15143'

# load design matrix
dm_file = os.path.join(DM_DIR, subj_id+'.csv')
design_mat = design_matrix.dmtx_from_csv(dm_file)

# load anat
anat_file = glob.glob(os.path.join(FMRI_DIR, 'S' + subj_id,
                                   'anat',
                                   'w*.img'))[0]

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
for i in range(14):
    contrasts['%s' % design_mat.names[2 * i]] = np.eye(n_columns)[2 * i]


# TODO : add all contrasts
contrasts['press R-L'] = contrasts['press_right'] - contrasts['press_left']

contrast_id = 'press R-L'
z_map, t_map, eff_map, var_map = glm.contrast(contrasts[contrast_id],
                                              con_id=contrast_id,
                                              output_z=True,
                                              output_stat=True,
                                              output_effects=True,
                                              output_variance=True)
