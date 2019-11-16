# -*- coding: utf-8 -*-
"""

"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error

df = pd.read_csv('training_data_wImpute_threshold_0.1.csv')

labels = df['meter_reading']
data = df.drop(columns=['meter_reading'])

#print(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state=42)

skf = KFold(n_splits=10,shuffle=True,random_state=42)

rmsle_dt = []
rmsle_linear = []
rmsle_knn = []
rmsle_nn = []

for train_index, test_index in skf.split(X_train):
    
    X_sub_train, X_valid = X_train.iloc[train_index], X_train.iloc[test_index]
    y_sub_train, y_valid = y_train.iloc[train_index], y_train.iloc[test_index]
    
    reg_tree = tree.DecisionTreeRegressor(criterion='mse')
    reg_tree = reg_tree.fit(X_sub_train, y_sub_train)
    predictions = reg_tree.predict(X_valid)

    rmsle_dt.append(np.sqrt(mean_squared_log_error(y_valid,predictions)))    
    #print(predictions)
    #print(np.sqrt(mean_squared_log_error(y_valid,predictions)))
    
    # Note: returns negative values which causes error when calculating RSMLE
    linear_reg = LinearRegression(fit_intercept=True).fit(X_sub_train,y_sub_train)
    predictions = linear_reg.predict(X_valid)
    
    #rmsle_linear.append(np.sqrt(mean_squared_log_error(y_valid,predictions)))
    #print(predictions)
    #print(np.sqrt(mean_squared_log_error(y_valid,predictions)))
    
    knn_reg = KNeighborsRegressor(n_neighbors=2,metric='manhattan').fit(X_sub_train,y_sub_train)
    predictions = knn_reg.predict(X_valid)
    
    rmsle_knn.append(np.sqrt(mean_squared_log_error(y_valid,predictions)))
    #print(predictions)
    #print(np.sqrt(mean_squared_log_error(y_test,predictions)))
    
    nn = MLPRegressor(activation='logistic').fit(X_sub_train,y_sub_train)
    predictions = nn.predict(X_valid)
    
    rmsle_nn.append(np.sqrt(mean_squared_log_error(y_valid,predictions)))
    #print(predictions)
    #print(np.sqrt(mean_squared_log_error(y_valid,predictions)))


""" Model Predictions of Validation Set"""


print('mean RMSLE = ',np.mean(rmsle_dt))
print('mean RMSLE = ',np.mean(rmsle_knn))
print('mean RMSLE = ',np.mean(rmsle_nn))

""" Model Predictions of Test Set """

# Regression Tree Results
predictions = reg_tree.predict(X_test)  
print(np.sqrt(mean_squared_log_error(y_test,predictions)))

# linear Regression Results
predictions = linear_reg.predict(X_test)
print(np.sqrt(mean_squared_log_error(y_test,predictions)))

# KNN Results
predictions = knn_reg.predict(X_test)
print(np.sqrt(mean_squared_log_error(y_test,predictions)))

# Multi Layer Perceptron Results
predictions = nn.predict(X_test)
print(np.sqrt(mean_squared_log_error(y_test,predictions)))
