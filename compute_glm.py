"""
    * Compute GLM for psydat subjects.
    Inputs are :
    - design matrix CSV 
    - fMRI filelist
    
    * Compute contrasts

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

# TODO : specify and estimate all contrasts
contrasts['press R-L'] = contrasts['press_right'] - contrasts['press_left']

contrast_list = ['anticip',
                 'anticip_hit',
                 'anticip_missed',
                 'anticip_noresp',
                 'anticip_hit-missed',
                 'anticip_missed-hit',
                 'anticip_hit-noresp',
                 'anticip_noresp-hit',
                 'anticip_hit_largewin-smallwin',
                 'anticip_hit_largewin-nowin',
                 'anticip_hit_smallwin-nowin',
                 'anticip_missed_largewin-smallwin',
                 'anticip_missed_largewin-nowin',
                 'anticip_missed_smallwin-nowin',
                 'anticip-anticip_noresp',
                 'feedback',
                 'feedback_hit',
                 'feedback_missed',
                 'feedback_hit-missed',
                 'feedback_missed-hit',
                 'feedback_hit_largewin-smallwin',
                 'feedback_hit_largewin-nowin',
                 'feedback_hit_smallwin-nowin',
                 'feedback_missed_largewin-smallwin',
                 'feedback_missed_largewin-nowin',
                 'feedback_missed_smallwin-nowin',
                 'press L+R',
                 'press L-R',
                 'press R-L',
                 'anticip_hit_somewin-nowin',
                 'anticip_missed_somewin-nowin',
                 'feedback_hit_somewin-nowin',
                 'feedback_missed_somewin-nowin',
                 'feedback_somewin_hit-missed',
                 'feedback_somewin_missed-hit',
                 'feedback_somewin-nowin',
                 'anticip_hit_largewin',
                 '-anticip_hit_largewin',
                 'feedback_hit_largewin',
                 '-feedback_hit_largewin',
                 'anticip_hit_largewin-feedback_hit_largewin',
                 'feedback_hit_largewin-anticip_hit_largewin',
                 'anticip_hit_nowin-feedback_hit_nowin',
                 'feedback_hit_nowin-anticip_hit_nowin',
                 'anticip_largewin-smallwin (hit+missed)',
                 'anticip_largewin-nowin (hit+missed)',
                 'anticip_smallwin-nowin (hit+missed)',
                 'feedback_largewin-smallwin (hit+missed)',
                 'feedback_largewin-nowin (hit+missed)',
                 'feedback_smallwin-nowin (hit+missed)']


contrast_id = 'press R-L'
z_map, t_map, eff_map, var_map = glm.contrast(contrasts[contrast_id],
                                              con_id=contrast_id,
                                              output_z=True,
                                              output_stat=True,
                                              output_effects=True,
                                              output_variance=True)

# TODO : save maps
# TODO : save reports
# 
