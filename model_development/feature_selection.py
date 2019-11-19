# -*- coding: utf-8 -*-
"""

"""

import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
from sklearn import preprocessing

df = pd.read_csv('threshold_0.8/training_data_woEncoding_threshold_0.8.csv')

le = preprocessing.LabelEncoder()
le.fit(df['primary_use'])
df['primary_use'] = le.transform(df['primary_use'])

labels = df['meter_reading']
data = df.drop(columns=['meter_reading'])

#print(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state=42)

        
rf = RandomForestRegressor(n_estimators=25, criterion='mse')
rf = rf.fit(X_train, y_train)

f = open('rf_var_importance.txt','w+')  
f. write(str(dict(zip(X_train.columns, rf.feature_importances_))))
f.close()

features = dict(zip(X_train.columns, rf.feature_importances_))
features = pd.DataFrame(list(features.items()))
features = features.sort_values(by=1,ascending=False)

print(features)

f = open('feature_selection_results.txt','a+')
for i in range(len(features)):
    
    if i != 0:
        X_train = X_train.drop(features.iloc[-i,0])
        print(X_train.columns)
    
    f.write('Features trained with: ', X_train.columns)

    rmsle_dt = []
    rmsle_rf = []
    
       
    # Decision Tree
    reg_tree = tree.DecisionTreeRegressor(criterion='friedman_mse')
    reg_tree = reg_tree.fit(X_train, y_train)
    predictions = reg_tree.predict(X_test)
    
    rmsle_dt = np.sqrt(mean_squared_log_error(y_test,predictions))
    mae_dt = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions, multioutput='variance_weighted')
    adj_r2 = 1 - ((1 - r2)*((len(y_test)-1)/(len(y_test)-len(X_test.columns) - 1)))
    r2_dt = r2
    adj_r2_dt = adj_r2
    
    f.write('Decision Tree' + '\n')
    f.write('test RMSLE = ' + str(rmsle_dt))
    f.write('test R^2 = ' + str(r2_dt))
    f.write('test Adj R^2 = '+ str(adj_r2_dt))
    f.write('test MAE = ' + str(mae_dt))
    
    # Random Forest 
    rf = RandomForestRegressor(n_estimators=25, criterion='mae')
    rf = rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    
    rmsle_rf = np.sqrt(mean_squared_log_error(y_test,predictions))
    mae_rf = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions, multioutput='variance_weighted')
    adj_r2 = 1 - ((1 - r2)*((len(y_test)-1)/(len(y_test)-len(X_test.columns) - 1)))
    r2_rf = r2
    adj_r2_rf =adj_r2
    
    f.write('Random Forest' + '\n')
    f.write('test RMSLE = ' + str(rmsle_rf))
    f.write('test R^2 = ' + str(r2_rf))
    f.write('test Adj R^2 = '+ str(adj_r2_rf))
    f.write('test MAE = ' + str(mae_rf))
    
    
    # K-Nearest Neighbors (KNN) Results
    knn_reg = KNeighborsRegressor(n_neighbors=2,metric='manhattan').fit(X_train,y_train)
    predictions = knn_reg.predict(X_test)
    
    rmsle_knn = np.sqrt(mean_squared_log_error(y_test,predictions))
    mae_knn = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions, multioutput='variance_weighted')
    adj_r2 = 1 - ((1 - r2)*((len(y_test)-1)/(len(y_test)-len(X_test.columns) - 1)))
    r2_knn = r2
    adj_r2_knn = adj_r2
    print('r2 ', r2)
    print('adj r2 ', adj_r2)
    
    f.write('KNN' + '\n')
    f.write('test RMSLE = ' + str(rmsle_knn))
    f.write('test R^2 = ' + str(r2_knn))
    f.write('test Adj R^2 = '+ str(adj_r2_knn))
    f.write('test MAE = ' + str(mae_knn))
    
    # Multi-Layer Perceptron
    nn = MLPRegressor(hidden_layer_sizes=(5,),activation='logistic',max_iter=1000).fit(X_train,y_train)
    predictions = nn.predict(X_test)
    
    rmsle_nn = np.sqrt(mean_squared_log_error(y_test,predictions))
    mae_nn = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions, multioutput='variance_weighted')
    adj_r2 = 1 - ((1 - r2)*((len(y_test)-1)/(len(y_test)-len(X_test.columns) - 1)))
    r2_nn = r2
    adj_r2_nn = adj_r2
    print('r2 ', r2)
    print('adj r2 ', adj_r2)

    f.write('Multi-Layer Perceptron' + '\n')
    f.write('test RMSLE = ' + str(rmsle_nn))
    f.write('test R^2 = ' + str(r2_nn))
    f.write('test Adj R^2 = '+ str(adj_r2_nn))
    f.write('test MAE = ' + str(mae_nn))
    
f.close()



