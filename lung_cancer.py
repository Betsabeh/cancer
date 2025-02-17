#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 08:53:52 2025

@author: betsa
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sksurv.datasets import load_veterans_lung_cancer
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder


data_x, data_y = load_veterans_lung_cancer()
print(data_x.head())
print("label=", data_y)

prob_36 = sum( 1  for item in data_y['Survival_in_days'] if item>36)
prob_36 = prob_36 / len(data_y['Survival_in_days'])

time, survival_prob, conf_int = kaplan_meier_estimator(
    data_y["Status"], data_y["Survival_in_days"], conf_type="log-log"
)

plt.step(time, survival_prob)
plt.fill_between(time, conf_int[0], conf_int[1], alpha=0.25, step="post")
plt.ylim(0, 1)
plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")

# one hot of categrical labels
data_x_numeric = OneHotEncoder().fit_transform(data_x)
print(data_x_numeric.head())

