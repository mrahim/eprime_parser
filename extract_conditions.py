# -*- coding: utf-8 -*-
"""
functions for the
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

import os
import glob
from collections import OrderedDict
import numpy as np
from scipy import io
import pandas as pd
import xlsxwriter
from configobj import ConfigObj

if os.path.isfile('io_paths.ini'):
    paths = ConfigObj(infile='io_paths.ini')
    BASE_DIR = paths['csv_output_dir']
    DST_BASE_DIR = paths['mat_output_dir']
    MAPPING_FILE = paths['subject_eprime_mapping']
    MVT_CSV_DIR = paths['movement_csv_dir']
    ANTICIP_DESIGN = False
    FEEDBACK_DESIGN = False
    USE_NO_RESP = False
    if paths['anticip_conditions'] == 'True':
        ANTICIP_DESIGN = True
    if paths['feedback_conditions'] == 'True':
        FEEDBACK_DESIGN = True
    if paths['no_resp_condition'] == 'True':
        USE_NO_RESP = True
else:
    BASE_DIR = 'data/csv'
    DST_BASE_DIR = 'data/mat'
    ANTICIP_DESIGN = True
    FEEDBACK_DESIGN = False
    USE_NO_RESP = False
    MAPPING_FILE = 'data/mapping.csv'
    MVT_CSV_DIR = 'data/mvt'

N_SCANS = 289
TR = 2400.
START_DELAY = 6000.
TASK_DURATION = {'anticip': 4., 'feedback': 1.45}


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
            if len(onsets[i]) == 0:
                durations[i] = 0.
                onsets[i] = [3600.]
                print output_file
    io.savemat(output_file, {'names': names,
                             'onsets': onsets,
                             'durations': durations})
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
    lw_idx = df[df['prize'] == 10]['TrialList']
    largewin[lw_idx.values - 1] = 1

    # small wins
    smallwin = np.zeros(len(df))
    sw_idx = df[df['prize'] == 2]['TrialList']
    smallwin[sw_idx.values - 1] = 1

    # no wins
    nowin = np.zeros(len(df))
    nw_idx = df[df['prize'] == 0]['TrialList']
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
    anticip_start_time = (df['PicturePrime.OnsetTime'] - first_onset +
                          START_DELAY - 2 * TR)/1000.
    response_time = (df['PictureTarget.RTTime'] - first_onset +
                     START_DELAY - 2 * TR)/1000.
    feedback_start_time = (df['PictureTarget.OnsetTime'] + df['Target_time'] -
                           first_onset + START_DELAY - 2 * TR)/1000.

    # Compute conditions
    cond = pd.DataFrame({'response_time': response_time,
                         'anticip_start_time': anticip_start_time,
                         'feedback_start_time': feedback_start_time})

    # Anticipation
    anticip_hit_largewin = \
    cond[(hit == 1) & (largewin == 1)]['anticip_start_time'].values
    anticip_hit_smallwin = \
    cond[(hit == 1) & (smallwin == 1)]['anticip_start_time'].values
    anticip_hit_nowin = \
    cond[(hit == 1) & (nowin == 1)]['anticip_start_time'].values
    anticip_hit = np.hstack((anticip_hit_largewin,
                             anticip_hit_smallwin,
                             anticip_hit_nowin))
    anticip_missed_largewin = \
    cond[(miss == 1) & (largewin == 1)]['anticip_start_time'].values
    anticip_missed_smallwin = \
    cond[(miss == 1) & (smallwin == 1)]['anticip_start_time'].values
    anticip_missed_nowin = \
    cond[(miss == 1) & (nowin == 1)]['anticip_start_time'].values
    anticip_missed = np.hstack((anticip_missed_largewin,
                                anticip_missed_smallwin,
                                anticip_missed_nowin))
    anticip_noresp = cond[(noresp==1)]['anticip_start_time'].values

    # Feedback
    feedback_hit_largewin = \
    cond[(hit == 1) & (largewin == 1)]['feedback_start_time'].values

    feedback_hit_smallwin = \
    cond[(hit==1) & (smallwin==1)]['feedback_start_time'].values
    feedback_hit_nowin = \
    cond[(hit==1) & (nowin==1)]['feedback_start_time'].values
    feedback_hit = np.hstack((feedback_hit_largewin,
                              feedback_hit_smallwin,
                              feedback_hit_nowin))

    feedback_missed_largewin = \
    cond[(miss==1) & (largewin==1)]['feedback_start_time'].values
    feedback_missed_smallwin = \
    cond[(miss==1) & (smallwin==1)]['feedback_start_time'].values
    feedback_missed_nowin = \
    cond[(miss==1) & (nowin==1)]['feedback_start_time'].values
    feedback_missed = np.hstack((feedback_missed_largewin,
                                 feedback_missed_smallwin,
                                 feedback_missed_nowin))
    feedback_noresp = cond[(noresp == 1)]['feedback_start_time'].values

    # Response
    press_left = cond[(pleft == 1)]['response_time'].values
    press_right = cond[(pright == 1)]['response_time'].values

    conditions = OrderedDict()
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
        if USE_NO_RESP:
            conditions['anticip_noresp'] = anticip_noresp

    if FEEDBACK_DESIGN:
        conditions['feedback_hit_largewin'] = feedback_hit_largewin
        conditions['feedback_hit_smallwin'] = feedback_hit_smallwin
        conditions['feedback_hit_nowin'] = feedback_hit_nowin
        conditions['feedback_missed_largewin'] = feedback_missed_largewin
        conditions['feedback_missed_smallwin'] = feedback_missed_smallwin
        conditions['feedback_missed_nowin'] = feedback_missed_nowin
        if USE_NO_RESP:
            conditions['feedback_noresp'] = feedback_noresp

    conditions['press_left'] = press_left
    conditions['press_right'] = press_right

    durations = OrderedDict()
    for k in conditions.keys():
        if 'feedback' in k:
            durations[k] = TASK_DURATION['feedback']
        elif 'anticip' in k:
            durations[k] = TASK_DURATION['anticip']
        else:
            durations[k] = 0.

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
    mapping = pd.read_csv(MAPPING_FILE, names=['eprime', 'subject'])

    subject_id, eprime_id = check_subject_eprime(f, mapping)
    subject_id = subject_id.astype(np.int)

    if len(subject_id) > 0:
        filepath = os.path.join(MVT_CSV_DIR,
                                ''.join(['S', str(subject_id[0]), '_reg.csv']))
        if os.path.isfile(filepath):
            reg = pd.read_csv(filepath)
            regressors = reg.values[:, 1:]
            output_file = os.path.join(DST_BASE_DIR,
                                       ''.join(['S', str(subject_id[0]),
                                                '_reg']))
            generate_multiregressors_mat(output_file, regressors)

        # Create paradigms
        condition = []
        onset = []
        duration = []
        for c in conditions:
            condition += [c] * len(conditions[c])
            onset = np.hstack([onset, conditions[c]])
            duration += [durations[c]] * len(conditions[c])

        output_file = os.path.join(DST_BASE_DIR,
                                   f.split('/')[-1].split('.')[0])

        generate_multiconditions_mat(output_file, conditions, durations)
        generate_multiconditions_excel(output_file, conditions, onset,
                                       condition)

        fig_title = f.split('/')[-1].split('.')[0]
        if len(subject_id) > 0:
            fig_title += '-S' + str(subject_id[0])
            output_file_s = os.path.join(DST_BASE_DIR,
                                         ''.join(['S', str(subject_id[0]),
                                                  '_', str(eprime_id),
                                                  '_cond']))
            generate_multiconditions_mat(output_file_s, conditions, durations)
            generate_multiconditions_excel(output_file_s, conditions, onset,
                                           condition)
