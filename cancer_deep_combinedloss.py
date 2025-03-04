#standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector as selector
from sklearn.model_selection import train_test_split
from lifelines.utils import concordance_index

#tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

# Data library
from sksurv.datasets import load_veterans_lung_cancer

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
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

## 5- build model
def build_model(input_dim):
  inputs = layers.Input(shape =(input_dim,))
  #hidden layers
  x = layers.Dense(16, activation = 'relu') (inputs)
  x = layers.Dense(8,  activation = 'relu') (x)
  x = layers.Dense(4,  activation = 'relu') (x)
  #Regression layer
  reg_output = layers.Dense(units =1, name='reg_output')(x)
  #classifcation layer   
  class_output = layers.Dense(1, activation = 'sigmoid',name='class_output')(x)  
     
  #Create Model
  model = tf.keras.Model(inputs = inputs, outputs = [reg_output, class_output])

  model.summary()
  return model

## 6-my loss function
def my_loss(y_true, y_pred):
  Alpha =0.4
  True_Sur_Time = y_true[0]
  True_Event_Time = y_true[1]

  pred_Sur_Time = y_pred[0]
  pred_Event_Time = y_pred[1]


  loss1 = tf.reduce_mean(tf.square(True_Sur_Time-pred_Sur_Time))
  loss2 = tf.reduce_mean(tf.keras.losses.binary_crossentropy(True_Event_Time, pred_Event_Time))  
  total_loss =Alpha *loss1 + (1-Alpha)*loss2
  return total_loss



## 7-Prepare data for training
y_train =  (y_reg_train, y_class_train)#np.column_stack((y_reg_train, y_class_train))  # Combine survival times and event indicators
print (np.shape(y_train))
## 8-Build the model
input_dim = X_train.shape[1]
model = build_model(input_dim)

## 9-Compile model
model.compile(optimizer='adam', loss=my_loss)

## 10-train
model.fit(X_train, y_train, epochs = 100, batch_size= 32, verbose= 1)

## 11-validation
# Evaluate the model on the test set
def evaluate_model(model, X_test, y_test):
    # Predict hazard outputs
    predictions = model.predict(X_test)

    pred_sur_time = predictions[0].flatten() 
   

    # Calculate C-index
    c_index = concordance_index(y_reg_test, -pred_sur_time, y_class_test)
    print(f"C-Index: {c_index:.4f}")

    return predictions

y_test = (y_reg_test, y_class_test)
predictions = evaluate_model(model, X_test, y_test)
#The predictions can be interpreted in terms of hazard ratios
print("Predicted hazard outputs:", predictions)



