# -*- coding: utf-8 -*-
"""
Functions to extract the headers of all the subjects

@author: Mehdi Rahim
"""

import os
import glob
from configobj import ConfigObj

if os.path.isfile('io_paths.ini'):
    paths = ConfigObj(infile='io_paths.ini')
    BASE_DIR = paths['eprime_files_dir']
else:
    BASE_DIR = 'data/eprime'


# Header Parsing Function
def parse_header_eprime(filename):
    """ returns a dictionary of values in the header of the filename
    """
    header = {}
    with open(filename, 'rU') as f:
        lines = [x.strip() for x in f.read().split('\n')]
        for line in lines:
            if line == "*** Header Start ***":
                continue
            if line == "*** Header End ***":
                return header
            fields = line.split(": ")
            if len(fields) == 2:
                header[fields[0]] = fields[1]

# File ID Parsing Function
def parse_file_id_eprime(filename):
    """ returns the file_id of from the filename
    """
    return filename.split('-')[1]


##############################################################################

# parse header for all the subjects
file_list = glob.glob(os.path.join(BASE_DIR, "*.txt"))

for fn in file_list:
    h, fname = os.path.split(fn)
    head = parse_header_eprime(fn)
    print head['SessionDate'], '-', head['SessionTime']
    print head['Subject'].strip(), '-', parse_file_id_eprime(fn), '-', fname
