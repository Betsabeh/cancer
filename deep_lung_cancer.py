#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 20:29:05 2025

@author: betsa
"""

# standard libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_selector as selector
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# other libraries
import tensorflow as tf
from tensorflow.keras import layers, models

# Data library
from sksurv.datasets import load_veterans_lung_cancer


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
## 1- Load Data
data_x, data_y = load_veterans_lung_cancer()
y_event =np.array([item[0] for item in data_y])
y_surv_time = np.array([item[1] for item in data_y])
print("Features:", data_x.head())
print("event labels:",y_event)
print("survival labels:",y_surv_time)


## 2-encoding
categorical_features_Scl = selector(dtype_exclude = np.number)
numerical_features_Scl = selector(dtype_include= np.number)

categorical_features_name = categorical_features_Scl(data_x)
numerical_features_name = numerical_features_Scl(data_x)

numerical_features = data_x[numerical_features_name]
categorical_features= data_x[categorical_features_name]

Onehot_encoder = OneHotEncoder(sparse_output=False)
categorical_transformed_features = Onehot_encoder.fit_transform(categorical_features)
#categorical_transformed_features = np.array(categorical_transformed_features)
map_categorical_name=Onehot_encoder.get_feature_names_out()

All_features = np.concatenate((categorical_transformed_features,numerical_features), axis= 1)
features_name= map_categorical_name.tolist() + numerical_features_name 
X_Features = pd.DataFrame(data = All_features, 
                        index = range(np.shape(All_features)[0]),
                        columns = features_name)

## 3-Train-test split  
X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(  
    X_Features, y_event, y_surv_time, test_size=0.2, random_state=42  )  


## 4-Feature Sclaing
Scaler  = StandardScaler()
X_train = Scaler.fit_transform(X_train)
X_test  = Scaler.transform(X_test)


## 5-Create NN
def create_NN_Model(input_shape):
     input_layer =layers.Input(shape = input_shape)
     #hidden layers
     Dense1 = layers.Dense(units =16, activation='relu') (input_layer)
     Dense2 = layers.Dense(units =8, activation='relu') (Dense1)
     
     #classifcation layer
     class_output = layers.Dense(units =1, activation = 'sigmoid', name='class_output')(Dense2)
     
     #Regression layer
     reg_output = layers.Dense(1, name='reg_output')(Dense2)  
     
     #Create Model
     model = tf.keras.Model(inputs = input_layer, outputs = [class_output, reg_output])
     
     #compile
     model.compile(optimizer = 'adam',  loss={'class_output': 'binary_crossentropy', 
                                              'reg_output': 'mse'},  
                   metrics={'class_output': 'accuracy', 'reg_output': 'mae'}  )
     model.summary()
     return model
     
     
     
     
model = create_NN_Model(input_shape=(X_train.shape[1],))

## 6-Train Model
history= model.fit(X_train, {'class_output': y_class_train, 'reg_output': y_reg_train},  
    epochs=30,   batch_size=32,  validation_split=0.2)  

train_mae = history.history['reg_output_mae']   
val_mae = history.history['val_reg_output_mae']  
train_accuracy = history.history['class_output_accuracy']  
val_accuracy = history.history['val_class_output_accuracy']  

# Create a figure with subplots  
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))  
epochs = range(1, len(train_mae) + 1) 

# Plot Mean Absolute Error  
ax1.plot(epochs, train_mae, 'bo-', label='Train MAE')  
ax1.plot(epochs, val_mae, 'ro-', label='Validation MAE')  
ax1.set_title('Training and Validation Mean Absolute Error')  
ax1.set_xlabel('Epochs')  
ax1.set_ylabel('Mean Absolute Error')  
ax1.legend()  
ax1.grid()  

# Plot Accuracy  
ax2.plot(epochs, train_accuracy, 'bo-', label='Train Accuracy')  
ax2.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')  
ax2.set_title('Training and Validation Accuracy')  
ax2.set_xlabel('Epochs')  
ax2.set_ylabel('Accuracy')  
ax2.legend()  
ax2.grid()  

# Show the plots  
plt.tight_layout()  
plt.show()  


## &-validation
results = model.evaluate(X_test, {'class_output': y_class_test, 'reg_output': y_reg_test})  

print("--------------------Results:--------------------------------")
print(f"Test Loss: {results[0]}")  
print(f"Test Accuracy: {results[1]}")  
print(f"Test MSE: {results[2]}")  
print(f"Test MAE: {results[3]}")  



