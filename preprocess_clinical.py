#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 08:47:09 2025

@author: betsa
"""

#preprocessing clincal data:
#https://github.com/luisvalesilva/multisurv/blob/master/data/preprocess_clinical.ipynb

import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import LabelEncoder
from lifelines import KaplanMeierFitter


clinical_data = pd.read_csv('clinical_data.tsv', na_values=['not reported', 'Not Reported'],
                            sep='\t',low_memory= False)
Data_shape = np.shape(clinical_data)
print("Data shape:", Data_shape)
print(clinical_data.head(6))

label_cols = ['submitter_id', 'days_to_last_follow_up', 'vital_status', 'days_to_death']

# check the missing values
print("----------Missing value------------")
not_miss_col =[]
for col in clinical_data.columns:
   n_missing = sum(clinical_data[col].isnull())
   if n_missing > 0:
      if n_missing == Data_shape[0]:
          clinical_data = clinical_data.drop(columns=[col])
      else:
          print(f'{col}: {n_missing} ( {round(n_missing/Data_shape[0]*100, 3)}%)')
    
            
print("New Data shape:", Data_shape)   
clinical_data['days_to_death'].plot(kind = 'box')  
clinical_data['days_to_death'].sort_values(ascending=False).plot(use_index=False, kind='line')
print(clinical_data.describe())
