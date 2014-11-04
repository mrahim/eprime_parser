# -*- coding: utf-8 -*-
"""
Functions to extract informations from e-prime files

@author: Mehdi Rahim
"""

import os, glob
import numpy as np
import pandas as pd

BASE_DIR = os.path.join('/', 'shfj', 'Ppsypim', 'PSYDAT', 'Stats', 'eprime')
DST_BASE_DIR = os.path.join('eprime_files', 'csv')


# File id and date Parser
def parse_and_correct_file_id_eprime(filename):
    """ returns the file_id and the corrected date from the filename
    """
    file_id = filename.split('-')[1].strip()
    c_date = ''
    if len(file_id) == 5:
        c_date = file_id[2:4] + '-' + file_id[0:2]
        if file_id[4] < '9':
            c_date += '-201' + file_id[4]
        else:
            c_date += '-200' + file_id[4]
    return file_id, c_date

# Quick and Dirty Parser
def parse_data_eprime(filename):
    """ returns a dict of header informations and a DataFrame of values
    at the 3rd level of filename
    """
    edf = pd.DataFrame() # Subject dataframe
    lvl = {}    # E-Prime level 3 dict
    hdr = {} # Header dict
    level_flag = -1 # Flag on the current header/level
    with open(filename, 'rU') as f:
        lines = [x.strip() for x in f.read().split('\n')]
        for line in lines:
            if line in ["*** Header Start ***", "*** Header End ***"]:
                # set the flag on header section
                level_flag = 0
                continue
            if line == "*** LogFrame Start ***":
                # reset the level 3 dict
                lvl = {}
                continue
            if line == "*** LogFrame End ***":
                # append dict according to the level
                if lvl:
                    edf = edf.append(lvl, ignore_index=True)
                level_flag = -1
                continue
            fields = line.split(": ")
            fields[0] = fields[0].replace(':', '')
            fields[0] = fields[0].replace(' ', '')
            if fields[0] == "Level":
                level_flag = int(fields[1])
                continue
            if level_flag == 3:
                lvl[fields[0]] = ''
                if len(fields) == 2:
                    lvl[fields[0]] = fields[1]
            elif level_flag in [0, 1, 2]:
                hdr[fields[0]] = fields[1]
    return edf, hdr


##############################################################################
""" Parsing all subjects and saving :
    - a session csv per subject 
    - a whole subject header csv 
"""

header_selected_cols = ['c_Subject', 'Subject', 'c_SessionDate',
                        'SessionDate', 'SessionTime', 'nbTrials', 'PP.Onset']
eprime_selected_cols = ['TrialList',
                        'PicturePrime.OnsetTime',
                        'PictureTarget.OnsetTime',
                        'PictureTarget.RTTime',
                        'Target_time',
                        'Fix_time',
                        'Ant_time',
                        'J_time',
                        'PictureTarget.ACC',
                        'PictureTarget.CRESP',
                        'PictureTarget.RESP',
                        'prize',
                        'SumPrize',
                        'CorrectAnswer',
                        'PictureTarget.OnsetDelay',
                        'PictureTarget.RT',
                        'TargetPosition',
                        'PicturePrime.OnsetDelay']

# header will contain "meta-data" of all the subjects
header = pd.DataFrame()

# list of the eprime files
file_list = glob.glob(os.path.join(BASE_DIR, '*.txt'))

for fn in file_list:
    h, fname = os.path.split(fn)
    print fname
    
    # Parse data (df) and header informations (hd)
    df, hd = parse_data_eprime(fn)
    
    # Add informations about the first onset (if available)
    hd['PP.Onset'] = ''
    if 'PicturePrime.OnsetTime' in df.keys():
        hd['PP.Onset'] = df['PicturePrime.OnsetTime'][0]

    # Add informations about the corrected subject id
    hd['c_Subject'], hd['c_SessionDate'] = parse_and_correct_file_id_eprime(fn)
    
    # Number of trials in order to check the experimentation integrity
    hd['nbTrials'] = np.str(df['TrialList'].count())

    # Append each subject    
    header = header.append(hd, ignore_index=True)
    
    
    # Save the raw exprimentation data of the current subject
    df.to_csv(os.path.join(DST_BASE_DIR, fname + '.csv'), sep=',')
                           
    # Save the selected data if the current subject
    if hd['nbTrials'] == '66':
        df.to_csv(os.path.join(DST_BASE_DIR, 'c_' + fname + '.csv'),
                  sep=',', columns=eprime_selected_cols)

# Save all subjects meta-data
header.to_csv(os.path.join(DST_BASE_DIR, 'all_subjects.csv'), sep=',')
header.to_csv(os.path.join(DST_BASE_DIR, 'all_subjects_c.csv'), sep=',',
              columns=header_selected_cols)
