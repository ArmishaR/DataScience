# -*- coding: utf-8 -*-
"""

"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error

df = pd.read_csv('training_data_wImpute_threshold_0.1.csv')

labels = df['meter_reading']
data = df.drop(columns=['meter_reading'])

#print(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state=42)

activation = ['logistic','relu','tanh']

for i in range(len(activation)):
    print('Activation Function: ',activation[i])
    
    skf = KFold(n_splits=10,shuffle=True,random_state=42)
    rmsle = []
    
    for train_index, test_index in skf.split(X_train, y_train):
        
        X_sub_train, X_valid = X_train.iloc[train_index], X_train.iloc[test_index]
        y_sub_train, y_valid = y_train.iloc[train_index], y_train.iloc[test_index]
        
        
        nn = MLPRegressor(hidden_layer_sizes=(25,),activation=activation[i]).fit(X_sub_train,y_sub_train)
        predictions = nn.predict(X_valid)
        
        #print(predictions)
        rmsle.append(np.sqrt(mean_squared_log_error(y_valid,predictions)))
        print(np.sqrt(mean_squared_log_error(y_valid,predictions)))

    print('mean RMSLE = ',np.mean(rmsle))