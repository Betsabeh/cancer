#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 09:33:28 2025

@author: betsa
"""

#https://github.com/luisvalesilva/multisurv/blob/master/data/preprocess_omics.ipynb

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import os
import requests
from io import StringIO


def request_file_info(data_type):
    fields = [
        'file_name',
        'cases.submitter_id',
        'cases.samples.sample_type',
        'cases.project.project_id',
        'cases.project.primary_site',
        ]

    fields = ','.join(fields)

    files_endpt = 'https://api.gdc.cancer.gov/files'

    filters = {
        'op': 'and',
        'content':[
            {
            'op': 'in',
            'content':{
                'field': 'files.experimental_strategy',
                'value': [data_type]
                }
            }
        ]
    }

    params = {
        'filters': filters,
        'fields': fields,
        'format': 'TSV',
        'size': '200000'
        }

    response = requests.post(files_endpt, headers = {'Content-Type': 'application/json'}, json = params)

    return pd.read_csv(StringIO(response.content.decode('utf-8')), sep='\t')





#------------------------------------------------------------------------------
labels = pd.read_csv('labels.tsv', sep='\t')
print(labels.head())
id_groups = {
    'train': list(labels.loc[labels['group'] == 'train', 'submitter_id']),
    'val': list(labels.loc[labels['group'] == 'val', 'submitter_id']),
    'test': list(labels.loc[labels['group'] == 'test', 'submitter_id'])}

# 1- load MiRNA
miRNA_files = request_file_info(data_type='miRNA-Seq')
miRNA_files = miRNA_files[miRNA_files['cases.0.project.project_id'].str.startswith('TCGA')]
miRNA_files = miRNA_files[miRNA_files['file_name'].str.endswith('mirbase21.mirnas.quantification.txt')]
miRNA_files = miRNA_files[miRNA_files['cases.0.samples.0.sample_type'] == 'Primary Tumor']

print('All rows:       ', miRNA_files.shape[0])
print('Unique patients:', miRNA_files['cases.0.submitter_id'].unique().shape[0])






