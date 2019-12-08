# -*- coding: utf-8 -*-
"""

"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
from sklearn import preprocessing

df = pd.read_csv('training_data_series5/training_data_woEncoding_threshold_0.8.csv')
df = df.drop(columns=df.columns[0], axis=1)

print(df)

le = preprocessing.LabelEncoder()
le.fit(df['primary_use'])
df['primary_use'] = le.transform(df['primary_use'])

labels = df['meter_reading']
data = df.drop(columns=['meter_reading'])

#print(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state=42)


criterion = ['mse','friedman_mse']

f = open('rf_para_opt.txt','a+')

for i in range(len(criterion)):
    
    print('Criterion: ',criterion[i])
    
    f.write('Criterion: ' + criterion[i] + '\n')
    
    skf = KFold(n_splits=10,shuffle=True,random_state=42)
    
    rmsle = []
    mae = []
    r2 = []
    adj_r2 = []  
    
    for train_index, test_index in skf.split(X_train):
        
        X_sub_train, X_valid = X_train.iloc[train_index], X_train.iloc[test_index]
        y_sub_train, y_valid = y_train.iloc[train_index], y_train.iloc[test_index]
                
        
        rf = RandomForestRegressor(n_estimators=25, criterion=criterion[i])
        rf = rf.fit(X_sub_train, y_sub_train)
        
        predictions = rf.predict(X_valid)
        
        rmsle.append(np.sqrt(mean_squared_log_error(y_valid,predictions)))
        mae.append(mean_absolute_error(y_valid, predictions))
        
        r2_value = r2_score(y_valid, predictions, multioutput='variance_weighted')
        
        r2.append(r2_score(y_valid, predictions, multioutput='variance_weighted'))
        adj_r2.append( 1 - ((1 - r2_value)*((len(y_valid)-1)/(len(y_valid)-len(X_valid.columns) - 1))))
        
    print('mean RMSLE for RF = ',np.mean(rmsle))
    print('mean R^2 = ',np.mean(r2))
    print('mean Adj R^2 = ',np.mean(adj_r2))
    print('mean MAE = ',np.mean(mae))
    
    
    f.write('mean RMSLE = ' + str(np.mean(rmsle)) + '\n')
    f.write('mean R^2 = ' + str(np.mean(r2)) + '\n')
    f.write('mean Adj R^2 = ' + str(np.mean(adj_r2)) + '\n')
    f.write('mean MAE = ' + str(np.mean(mae)) + '\n')

f.close()

