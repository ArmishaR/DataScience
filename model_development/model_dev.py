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
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
from yellowbrick.regressor import ResidualsPlot, PredictionError

df = pd.read_csv('training_data_wImpute_threshold_0.1.csv')

labels = df['meter_reading']
data = df.drop(columns=['meter_reading'])

#print(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state=42)

skf = KFold(n_splits=10,shuffle=True,random_state=42)

rmsle_dt = []
r2_dt = []
adj_r2_dt = []
mae_dt = []

rmsle_linear = []
r2_linear = []
adj_r2_linear = []
mae_linear = []

rmsle_knn = []
r2_knn = []
adj_r2_knn = []
mae_knn = []

rmsle_nn = []
r2_nn = []
adj_r2_nn = []
mae_nn = []

print("start")

for train_index, test_index in skf.split(X_train):
    print("doing kfold")

    X_sub_train, X_valid = X_train.iloc[train_index], X_train.iloc[test_index]
    y_sub_train, y_valid = y_train.iloc[train_index], y_train.iloc[test_index]

    '''reg_tree = tree.DecisionTreeRegressor(criterion='mse')
    reg_tree = reg_tree.fit(X_sub_train, y_sub_train)
    predictions = reg_tree.predict(X_valid)

    rmsle_dt.append(np.sqrt(mean_squared_log_error(y_valid,predictions)))
    mae_dt.append(mean_absolute_error(y_valid, predictions))
    r2 = r2_score(y_valid, predictions, multioutput='variance_weighted')
    adj_r2 = 1 - ((1 - r2)*((len(y_valid)-1)/(len(y_valid)-len(X_valid.columns) - 1)))
    r2_dt.append(r2)
    adj_r2_dt.append(adj_r2)'''

    #print(predictions)
    #print(np.sqrt(mean_squared_log_error(y_valid,predictions)))

    # Note: returns negative values which causes error when calculating RSMLE
    '''linear_reg = LinearRegression(fit_intercept=False).fit(X_sub_train,y_sub_train)
    predictions = linear_reg.predict(X_valid)

    #rmsle_linear.append(np.sqrt(mean_squared_log_error(y_valid,predictions)))
    mae_linear.append(mean_absolute_error(y_valid, predictions))
    r2 = r2_score(y_valid, predictions, multioutput='variance_weighted')
    adj_r2 = 1 - ((1 - r2)*((len(y_valid)-1)/(len(y_valid)-len(X_valid.columns) - 1)))
    r2_linear.append(r2)
    adj_r2_linear.append(adj_r2)
    print('r2 ', r2)
    print('adj r2 ', adj_r2)'''
    #print(predictions)
    #print(np.sqrt(mean_squared_log_error(y_valid,predictions)))

    '''knn_reg = KNeighborsRegressor(n_neighbors=2,metric='manhattan').fit(X_sub_train,y_sub_train)
    predictions = knn_reg.predict(X_valid)

    rmsle_knn.append(np.sqrt(mean_squared_log_error(y_valid,predictions)))
    mae_knn.append(mean_absolute_error(y_valid, predictions))
    r2 = r2_score(y_valid, predictions, multioutput='variance_weighted')
    adj_r2 = 1 - ((1 - r2)*((len(y_valid)-1)/(len(y_valid)-len(X_valid.columns) - 1)))
    r2_knn.append(r2)
    adj_r2_knn.append(adj_r2)
    print('r2 ', r2)
    print('adj r2 ', adj_r2)'''
    #print(predictions)
    #print(np.sqrt(mean_squared_log_error(y_test,predictions)))

    nn = MLPRegressor(activation='logistic').fit(X_sub_train,y_sub_train)
    predictions = nn.predict(X_valid)

    rmsle_nn.append(np.sqrt(mean_squared_log_error(y_valid,predictions)))
    mae_nn.append(mean_absolute_error(y_valid, predictions))
    r2 = r2_score(y_valid, predictions, multioutput='variance_weighted')
    adj_r2 = 1 - ((1 - r2)*((len(y_valid)-1)/(len(y_valid)-len(X_valid.columns) - 1)))
    r2_nn.append(r2)
    adj_r2_nn.append(adj_r2)
    print('r2 ', r2)
    print('adj r2 ', adj_r2)
    #print(predictions)
    #print(np.sqrt(mean_squared_log_error(y_valid,predictions)))


""" Model Predictions of Validation Set"""

'''print('mean RMSLE = ',np.mean(rmsle_dt))
print('mean R^2 = ',np.mean(r2_dt))
print('mean Adj R^2 = ',np.mean(adj_r2_dt))
print('mean MAE = ',np.mean(mae_dt))'''

#print('mean RMSLE = ',np.mean(rslme_linear))
'''print('mean R^2 = ',np.mean(r2_linear))
print('mean Adj R^2 = ',np.mean(adj_r2_linear))
print('mean MAE = ',np.mean(mae_linear))'''

'''print('mean RMSLE = ',np.mean(rmsle_knn))
print('mean R^2 = ',np.mean(r2_knn))
print('mean Adj R^2 = ',np.mean(adj_r2_knn))
print('mean MAE = ',np.mean(mae_knn))'''

print('mean RMSLE = ',np.mean(rmsle_nn))
print('mean R^2 = ',np.mean(r2_nn))
print('mean Adj R^2 = ',np.mean(adj_r2_nn))
print('mean MAE = ',np.mean(mae_nn))

""" Model Predictions of Test Set """

# Regression Tree Results
#predictions = reg_tree.predict(X_test)


# linear Regression Results
#predictions = linear_reg.predict(X_test)

# KNN Results
#predictions = knn_reg.predict(X_test)

# Multi Layer Perceptron Results
predictions = nn.predict(X_test)

print('test RMSLE', np.sqrt(mean_squared_log_error(y_test,predictions)))
print('test MAE', mean_absolute_error(y_test, predictions))
r2 = r2_score(y_test, predictions)
adj_r2 = 1 - ((1 - r2)*((len(y_test)-1)/(len(y_test)-len(X_test.columns) - 1)))
print('test R2', r2)
print('test adjusted R2', adj_r2)

visualizer = ResidualsPlot(nn)
#visualizer = PredictionError(knn_reg)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()
