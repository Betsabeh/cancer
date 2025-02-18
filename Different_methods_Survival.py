import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from lifelines.utils import concordance_index  

from sksurv.datasets import load_veterans_lung_cancer
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder
from sklearn import set_config
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.svm import FastSurvivalSVM
from sksurv.column import encode_categorical
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis, GradientBoostingSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
#-------------------------------------------------------------------------------
## 1- Load Data

data_x, data_y = load_veterans_lung_cancer()
#data_x is a data frame containing the features,
print("----------------------------------Data Features---------------------------------")
print(data_x.head())
#y is a structured array containing the event indicator , as boolean, and the survival/censoring time yi
print("------------------Y has boolean and sutvival time---------------------------------")
print("label=", data_y[0])

## 2- plot the survival/censoring times
print("-------------------Statistic Details----------------------------------")
print("min survival/censoring time: ", np.min(data_y["Survival_in_days"]))
print("max survival/censoring time: ", np.max(data_y["Survival_in_days"]))
n_censored = data_y.shape[0] - data_y["Status"].sum()
print(f"{n_censored / data_y.shape[0] * 100:.1f}% of records are censored")
plt.figure(figsize=(9, 6))
val, bins, patches = plt.hist(
    (data_y["Survival_in_days"][data_y["Status"]], data_y["Survival_in_days"][~data_y["Status"]]), bins=30, stacked=True
)
_ = plt.legend(patches, ["Time of Death", "Time of Censoring"])

##  3-probability of survaival days and kaplan_meier_estimator
prob_36 = sum( 1  for item in data_y['Survival_in_days'] if item>36)
prob_36 = prob_36 / len(data_y['Survival_in_days'])

time, survival_prob, conf_int = kaplan_meier_estimator(
    data_y["Status"], data_y["Survival_in_days"], conf_type="log-log"
)

print("------------------------------COX estimator--------------------------------------")
## 4- Cox esimator
data_x_numeric = OneHotEncoder().fit_transform(data_x)
#data_x_numeric.head()
#print(data_x_numeric.loc[0])
set_config(display="text")  # displays text representation of estimators
estimator = CoxPHSurvivalAnalysis()
estimator.fit(data_x_numeric, data_y)

## Test Data
x_new = pd.DataFrame.from_dict(
    {
        1: [65, 0, 0, 1, 60, 1, 0, 1],
        2: [65, 0, 0, 1, 60, 1, 0, 0],
        3: [65, 0, 1, 0, 60, 1, 0, 0],
        4: [65, 0, 1, 0, 60, 1, 0, 1],
    },
    columns=data_x_numeric.columns,
    orient="index",
)
###print("Test Data:")
###print(x_new)
# Predict Cox
pred_surv = estimator.predict_survival_function(x_new)
print("COX Results:")
print("prediction=", pred_surv)
time_points = np.arange(1, 1000)

for i, surv_func in enumerate(pred_surv):
    plt.step(time_points, surv_func(time_points), where="post", label=f"Sample {i + 1}")
plt.ylabel(r"est. probability of survival $\hat{S}(t)$")
plt.xlabel("time $t$")
plt.legend(loc="best")

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
## 4- SVM
print("---------------------------------SVM--------------------------------------")
estimator = FastSurvivalSVM(max_iter=1000, tol=1e-5, random_state=0)
x = encode_categorical(data_x)
###print("Data features in SVM:")
###print(x.loc[0])
estimator.fit(x, data_y)
pred = estimator.predict(x.iloc[:2])
print("SVM Results:")
print(np.round(pred, 3))
print(data_y[:2])
print("SVM Test Data resuls")
print(estimator.predict(x_new))

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
## 5-GB
print("---------------------------------GB---------------------------------------")
#separate train and test:
#second_column = np.array([item[1] for item in data_y])  # List comprehension to gather the second items
###print("labels is GB", data_y[0])


X_train, X_test, y_train, y_test = train_test_split(x, data_y, test_size=0.25, random_state=0)
###print("data features in GB")
###print(X_train.loc[0])


#train
est_cph_tree = GradientBoostingSurvivalAnalysis(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)
est_cph_tree.fit(X_train, y_train)
result_test = est_cph_tree.predict(X_test)
print("Result GB:")
print(result_test)
print("Test real labels",y_test)
print("Cindex GB:",est_cph_tree.score(X_test, y_test))
#--------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
## 6-RF
print("--------------------------------RF---------------------------------------")
rsf = RandomSurvivalForest(n_estimators=200, min_samples_split=5, min_samples_leaf=15, n_jobs=-1, random_state=23)
rsf.fit(X_train, y_train)
print("Cindex RF:",rsf.score(X_test, y_test))
###print(X_test)
print("predicted values:",rsf.predict(X_test))
print("real labels:",y_test)
#selecting test Data
X_test_sorted = X_test.sort_values(by=["Age_in_years"])
X_test_sel = pd.concat((X_test_sorted.head(3), X_test_sorted.tail(3)))
###print("selected Test:")
###print(X_test_sel)

surv = rsf.predict_survival_function(X_test_sel, return_array=True)
plt.figure()
for i, s in enumerate(surv):
    plt.step(rsf.unique_times_, s, where="post", label=str(i))
plt.ylabel("Survival probability")
plt.xlabel("Time in days")
plt.legend()
plt.grid(True)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
## 7-Dtree
print('-----------------------------Dtree---------------------------------------')
dtree = DecisionTreeRegressor(max_depth =4)
X_train, X_test, y_train, y_test = train_test_split(x, data_y, test_size=0.25, random_state=0)

#separate survival month
sur_train = np.array([item[1] for item in y_train])  # survival time
event_train = np.array([item[0] for item in y_train])  # event observed
sur_test = np.array([item[1] for item in y_test])  # survival time
event_test = np.array([item[0] for item in y_test])  # event observed

dtree.fit(X_train, sur_train)
print("Dtree results:")
pred_y_test = dtree.predict(X_test)
print("real values:")
print(sur_test)
print(pred_y_test)
c_index = concordance_index(sur_test, pred_y_test,event_test)  
print("Concordance Index:", c_index)  



