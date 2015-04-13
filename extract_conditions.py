# -*- coding: utf-8 -*-
"""
function for the 
A script that extracts the conditions from eprime csv file,
and saves them into an SPM multiconditions file (.mat).
It saves the corresponding movement regressors from movement_files/
into a (.mat) file in movement_files
The conditions are :
    *
    - Anticip_hit_largewin
    - Anticip_hit_smallwin
    - Anticip_hit_nowin
    - Anticip_missed_largewin
    - Anticip_missed_smallwin
    - Anticip_missed_nowin
    - Anticip_noresp
    *
    - Feedback_hit_largewin
    - Feedback_hit_smallwin
    - Feedback_hit_nowin
    - Feedback_missed_largewin
    - Feedback_missed_smallwin
    - Feedback_missed_nowin
    - Feedback_noresp
    *
    - Press_left
    - Press_right
"""

import os, glob
from collections import OrderedDict
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter
from nipy.modalities.fmri import design_matrix
from nipy.modalities.fmri.experimental_paradigm import BlockParadigm


BASE_DIR = os.path.join('eprime_files_caiman', 'csv')
DST_BASE_DIR = os.path.join('eprime_files_caiman', 'mat')

N_SCANS = 289
TR = 2400.
START_DELAY = 6000.
TASK_DURATION = {'anticip': 4., 'feedback': 1.45}


ANTICIP_DESIGN = True

def check_subject_eprime(eprime_file, mapping):
    """A temporary function that checks if the eprime id 
    has an existing corresponding subject
    """
    eprime_nb = eprime_file.split('/')[-1].split('.')[0].rsplit('-')[-2]
    res = mapping[mapping['eprime'] == int(eprime_nb)]['subject'].values
    return res, eprime_nb



def generate_multiconditions_excel(output_file, conditions, onset, condition):
    """Generate a *.xlsx file which contains the conditions
    """
    work = xlsxwriter.Workbook(output_file + '.xlsx')
    wsheet = work.add_worksheet('Conditions')
    col = 0
    for key in conditions.keys():
        wsheet.write(0, col, key)
        row = 1
        for item in conditions[key]:
            wsheet.write(row, col, item)
            row += 1
        col += 1
        
    wsheet2 = work.add_worksheet('Timeline')
    order = onset.argsort()
    wsheet2.write(0, 0, 'onset')
    wsheet2.write(0, 1, 'condition')
    row = 1
    for i in order:
        wsheet2.write(row, 0, onset[i])
        wsheet2.write(row, 1, condition[i])
        row += 1
    work.close()


def generate_multiregressors_mat(output_file, regressors):
    """Generate a *.mat file that contains the regressors
    """
    io.savemat(output_file, {'R': regressors})

def generate_multiconditions_mat(output_file, conditions, ddurations):
    """Generate a *.mat file that contains the names, the onsets
        and the durations, according to SPM's mutiple coditions file
    """
    names = np.zeros((len(conditions),), dtype=np.object)
    onsets = np.zeros((len(conditions),), dtype=np.object)
    durations = np.zeros((len(conditions),), dtype=np.object)
    for i in np.arange(0, len(conditions)):
            names[i] = conditions.keys()[i]
            onsets[i] = conditions[names[i]]
            durations[i] = ddurations[names[i]]
            if len(onsets[i])==0:
                durations[i] = 0.
                onsets[i] = [3600.]
                print output_file
    io.savemat(output_file, {'names' : names,
                             'onsets' : onsets,
                             'durations' : durations})
    return names, onsets, durations

def compute_mid_conditions(filename):
    df = pd.read_csv(filename)
    # Extract hits, misses and noresps
    # hits
    hit = np.zeros(len(df))
    h_idx = df[df['PictureTarget.RESP'].notnull()]['TrialList']
    hit[h_idx.values - 1] = 1
    
    # noresps
    noresp = np.zeros(len(df))
    n_idx = df[df['PictureTarget.RESP'].isnull()]['TrialList']
    noresp[n_idx.values - 1] = 1
    
    # misses
    miss = np.zeros(len(df))
    m_idx = df[df['PictureTarget.RESP'].isnull()]['TrialList']
    miss[m_idx.values - 1] = 1
    
    # Extract bigwins, smallwins and nowins
    # big wins
    largewin = np.zeros(len(df))
    lw_idx = df[df['prize']==10]['TrialList']
    largewin[lw_idx.values - 1] = 1
    
    # small wins
    smallwin = np.zeros(len(df))
    sw_idx = df[df['prize']==2]['TrialList']
    smallwin[sw_idx.values - 1] = 1
    
    # no wins
    nowin = np.zeros(len(df))
    nw_idx = df[df['prize']==0]['TrialList']
    nowin[nw_idx.values - 1] = 1
    

    # Extract press left (5), press right (4)
    # press left
    pleft = np.zeros(len(df))
    pl_idx = df[df['PictureTarget.RESP'] == 5]['TrialList']
    pleft[pl_idx.values - 1] = 1
    
    # press right
    pright = np.zeros(len(df))
    pr_idx = df[df['PictureTarget.RESP'] == 4]['TrialList']
    pright[pr_idx.values - 1] = 1
    
    # Extract times
    first_onset = df['PicturePrime.OnsetTime'][0]
    anticip_start_time = (df['PicturePrime.OnsetTime'] - first_onset + START_DELAY - 2 * TR)/1000.
    response_time = (df['PictureTarget.RTTime'] - first_onset + START_DELAY - 2 * TR)/1000.
    feedback_start_time = (df['PictureTarget.OnsetTime'] + df['Target_time'] - first_onset + START_DELAY - 2 * TR)/1000.
    
    # Compute conditions
    cond = pd.DataFrame({'response_time': response_time,
                         'anticip_start_time': anticip_start_time,
                         'feedback_start_time': feedback_start_time})
    
    # Anticipation
    anticip_hit_largewin = cond[(hit==1) & (largewin==1)]['anticip_start_time'].values
    anticip_hit_smallwin = cond[(hit==1) & (smallwin==1)]['anticip_start_time'].values
    anticip_hit_nowin = cond[(hit==1) & (nowin==1)]['anticip_start_time'].values
    anticip_hit = np.hstack((anticip_hit_largewin,
                             anticip_hit_smallwin, anticip_hit_nowin))
    anticip_hit_modgain = np.hstack([[3.]*len(anticip_hit_largewin),
                                     [2.]*len(anticip_hit_smallwin),
                                     [1.]*len(anticip_hit_nowin)])
    
    anticip_missed_largewin = cond[(miss==1) & (largewin==1)]['anticip_start_time'].values
    anticip_missed_smallwin = cond[(miss==1) & (smallwin==1)]['anticip_start_time'].values
    anticip_missed_nowin = cond[(miss==1) & (nowin==1)]['anticip_start_time'].values
    anticip_missed = np.hstack((anticip_missed_largewin,
                                anticip_missed_smallwin, anticip_missed_nowin))
    anticip_missed_modgain = np.hstack([[3.]*len(anticip_missed_largewin),
                                        [2.]*len(anticip_missed_smallwin),
                                        [1.]*len(anticip_missed_nowin)])
    
    anticip_noresp = cond[(noresp==1)]['anticip_start_time'].values
    
    # Feedback
    feedback_hit_largewin = cond[(hit==1) & (largewin==1)]['feedback_start_time'].values
    feedback_hit_smallwin = cond[(hit==1) & (smallwin==1)]['feedback_start_time'].values
    feedback_hit_nowin = cond[(hit==1) & (nowin==1)]['feedback_start_time'].values
    feedback_hit = np.hstack((feedback_hit_largewin,
                              feedback_hit_smallwin, feedback_hit_nowin))
    feedback_hit_modgain = np.hstack([[3.]*len(feedback_hit_largewin),
                                     [2.]*len(feedback_hit_smallwin),
                                     [1.]*len(feedback_hit_nowin)])
    
    feedback_missed_largewin = cond[(miss==1) & (largewin==1)]['feedback_start_time'].values
    feedback_missed_smallwin = cond[(miss==1) & (smallwin==1)]['feedback_start_time'].values
    feedback_missed_nowin = cond[(miss==1) & (nowin==1)]['feedback_start_time'].values
    feedback_missed = np.hstack((feedback_missed_largewin,
                                 feedback_missed_smallwin, feedback_missed_nowin))
    feedback_missed_modgain = np.hstack([[3.]*len(feedback_missed_largewin),
                                        [2.]*len(feedback_missed_smallwin),
                                        [1.]*len(feedback_missed_nowin)])
    
    feedback_noresp = cond[(noresp==1)]['feedback_start_time'].values
    
    # Response
    press_left = cond[(pleft==1)]['response_time'].values
    press_right = cond[(pright==1)]['response_time'].values
    
    # namelist
    namelist = ['anticip_hit', 'anticip_missed', 'anticip_noresp',
                'feedback_hit', 'feedback_missed', 'feedback_noresp',
                'press_left', 'press_right']
    
    modulationnamelist = ['anticip_hit_modgain', 'anticip_missed_modgain', 
                          'feedback_hit_modgain', 'feedback_missed_modgain']

    conditions =  OrderedDict()
    """
    
    #XXX As the missed case is often missing, we won't use it at this time
    #XXX missed case has been corrected
    
    conditions = {'anticip_hit_largewin' : anticip_hit_largewin,
                  'anticip_hit_smallwin' : anticip_hit_smallwin,
                  'anticip_hit_nowin' : anticip_hit_nowin,
                  'anticip_missed_largewin' : anticip_missed_largewin,
                  'anticip_missed_smallwin' : anticip_missed_smallwin,
                  'anticip_missed_nowin' : anticip_missed_nowin,
                  'anticip_noresp' : anticip_noresp,
                  'feedback_hit_largewin' : feedback_hit_largewin,
                  'feedback_hit_smallwin' : feedback_hit_smallwin,
                  'feedback_hit_nowin' : feedback_hit_nowin,
                  'feedback_missed_largewin' : feedback_missed_largewin,
                  'feedback_missed_smallwin' : feedback_missed_smallwin,
                  'feedback_missed_nowin' : feedback_missed_nowin,
                  'feedback_noresp' : feedback_noresp,
                  'press_left' : press_left,
                  'press_right' : press_right}
   """

    if ANTICIP_DESIGN:
        conditions['anticip_hit_largewin'] = anticip_hit_largewin
        conditions['anticip_hit_smallwin'] = anticip_hit_smallwin
        conditions['anticip_hit_nowin'] = anticip_hit_nowin
        conditions['anticip_missed_largewin'] = anticip_missed_largewin
        conditions['anticip_missed_smallwin'] = anticip_missed_smallwin
        conditions['anticip_missed_nowin'] = anticip_missed_nowin
        #conditions['anticip_noresp'] = anticip_noresp
    else:
        conditions['feedback_hit_largewin'] = feedback_hit_largewin
        conditions['feedback_hit_smallwin'] = feedback_hit_smallwin
        conditions['feedback_hit_nowin'] = feedback_hit_nowin
        conditions['feedback_missed_largewin'] = feedback_missed_largewin
        conditions['feedback_missed_smallwin'] = feedback_missed_smallwin
        conditions['feedback_missed_nowin'] = feedback_missed_nowin
        #conditions['feedback_noresp'] = feedback_noresp
    conditions['press_left'] = press_left
    conditions['press_right'] = press_right
   

    durations =  OrderedDict()
    
    for k in conditions.keys():
        if 'feedback' in k:
            durations[k] = TASK_DURATION['feedback']
        elif 'anticip' in k:
            durations[k] = TASK_DURATION['anticip']
        else:
            durations[k] = 0.
    """
    durations = {'anticip_hit_largewin' : 4.,
                  'anticip_hit_smallwin' : 4.,
                  'anticip_hit_nowin' : 4.,
                  'anticip_missed_largewin' : 4.,
                  'anticip_missed_smallwin' : 4.,
                  'anticip_missed_nowin' : 4.,
                  'anticip_noresp' : 4.,
                  'feedback_hit_largewin' : 1.45,
                  'feedback_hit_smallwin' : 1.45,
                  'feedback_hit_nowin' : 1.45,
                  'feedback_missed_largewin' : 1.45,
                  'feedback_missed_smallwin' : 1.45,
                  'feedback_missed_nowin' : 1.45,
                  'feedback_noresp' : 1.45,
                  'press_left' : 0.,
                  'press_right' : 0.}
    """
    
    return conditions, durations

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

# Load eprime csv file
file_list = glob.glob(os.path.join(BASE_DIR, 'c_*.csv'))

for f in file_list:
    print f

    # Compute conditions    
    conditions, durations = compute_mid_conditions(f)

    # Load regressors if they exist
    mapping = pd.read_csv(os.path.join('eprime_files_caiman', 'mapping.csv'),
                          names=['eprime','subject'])
    
    subject_id, eprime_id = check_subject_eprime(f, mapping)
    subject_id = subject_id.astype(np.int)
    print subject_id, eprime_id
    
    if len(subject_id)>0:
        #print subject_id[0]
        filepath = os.path.join('movement_files_caiman', 
                                ''.join(['S',str(subject_id[0]), '_reg.csv']))
        if os.path.isfile(filepath):
            reg = pd.read_csv(filepath)
            regressors = reg.values[:,1:]
            output_file = os.path.join(DST_BASE_DIR, 
                                       ''.join(['S',str(subject_id[0]),'_reg']))
            generate_multiregressors_mat(output_file, regressors)
    

        # Create paradigms   
        condition = []
        onset = []
        duration = []
        for c in conditions:
            condition += [c]*len(conditions[c])
            onset = np.hstack([onset, conditions[c]])
            duration += [durations[c]]*len(conditions[c])
    
        paradigm = BlockParadigm(con_id=condition,
                                 onset=onset,
                                 duration=duration)
                                   
        frametimes = np.linspace(0, (N_SCANS-1)*TR/1000., num=N_SCANS)
    
        design_mat = design_matrix.make_dmtx(frametimes, paradigm,
                                         hrf_model='Canonical',
                                         drift_model='Cosine',
                                         hfcut=128)
                                         
        output_file = os.path.join(DST_BASE_DIR,
                                   f.split('/')[-1].split('.')[0])
    
        generate_multiconditions_mat(output_file, conditions, durations)
        generate_multiconditions_excel(output_file, conditions, onset, condition)
    
        fig_title = f.split('/')[-1].split('.')[0]
        if len(subject_id)>0:
            fig_title += '-S'+ str(subject_id[0])
            output_file_s = os.path.join(DST_BASE_DIR,
                                         ''.join(['S',str(subject_id[0]),
                                         '_', str(eprime_id), '_cond']))        
            generate_multiconditions_mat(output_file_s, conditions, durations)
            generate_multiconditions_excel(output_file_s, conditions, onset, condition)
        #design_mat.show()
        #plt.title(fig_title)
        #print [len(conditions[k]) for k in conditions.keys()]
