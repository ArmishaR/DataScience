# -*- coding: utf-8 -*-
"""

"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error, r2_score
from yellowbrick.regressor import ResidualsPlot, PredictionError

df = pd.read_csv('training_data_woEncoding_threshold_0.8.csv')

# Encoding 
le = preprocessing.LabelEncoder()
le.fit(df['primary_use'])
df['primary_use'] = le.transform(df['primary_use'])

labels = df['meter_reading']
data = df.drop(columns=['meter_reading'])

# Open to print results
f = open('test_results.txt','a+')

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state=42)

""" Model Predictions of Test Set """

reg_tree = tree.DecisionTreeRegressor(criterion='friedman_mse')
reg_tree = reg_tree.fit(X_train, y_train)
predictions = reg_tree.predict(X_test)

rmsle_dt = np.sqrt(mean_squared_log_error(y_test,predictions))
mae_dt = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions, multioutput='variance_weighted')
adj_r2 = 1 - ((1 - r2)*((len(y_test)-1)/(len(y_test)-len(X_test.columns) - 1)))
r2_dt = r2
adj_r2_dt = adj_r2


rf = RandomForestRegressor(n_estimators=25, criterion='mae')
rf = rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

rmsle_rf = np.sqrt(mean_squared_log_error(y_test,predictions))
mae_rf = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions, multioutput='variance_weighted')
adj_r2 = 1 - ((1 - r2)*((len(y_test)-1)/(len(y_test)-len(X_test.columns) - 1)))
r2_rf = r2
adj_r2_rf =adj_r2

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


print('test RMSLE = ',np.mean(rmsle_dt))
print('test R^2 = ',np.mean(r2_dt))
print('test Adj R^2 = ',np.mean(adj_r2_dt))
print('test MAE = ',np.mean(mae_dt))

f.write('Decision Tree' + '\n')
f.write('test RMSLE = ' + str(rmsle_dt))
f.write('test R^2 = ' + str(r2_dt))
f.write('test Adj R^2 = '+ str(adj_r2_dt))
f.write('test MAE = ' + str(mae_dt))

print('test RMSLE = ',rmsle_rf)
print('test R^2 = ', r2_rf)
print('test Adj R^2 = ', adj_r2_rf)
print('test MAE = ', mae_rf)

f.write('Random Forest' + '\n')
f.write('test RMSLE = ' + str(rmsle_rf))
f.write('test R^2 = ' + str(r2_rf))
f.write('test Adj R^2 = '+ str(adj_r2_rf))
f.write('test MAE = ' + str(mae_rf))

'''print('test RMSLE = ', rslme_linear)
print('test R^2 = ', r2_linear)
print('test Adj R^2 = ', adj_r2_linear)
print('test MAE = ', mae_linear)'''

print('test RMSLE = ', rmsle_knn)
print('test R^2 = ', r2_knn)
print('test Adj R^2 = ', adj_r2_knn)
print('test MAE = ', mae_knn)

f.write('KNN' + '\n')
f.write('test RMSLE = ' + str(rmsle_knn))
f.write('test R^2 = ' + str(r2_knn))
f.write('test Adj R^2 = '+ str(adj_r2_knn))
f.write('test MAE = ' + str(mae_knn))

print('test RMSLE = ', rmsle_nn)
print('test R^2 = ', r2_nn)
print('test Adj R^2 = ', adj_r2_nn)
print('test MAE = ', mae_nn)

f.write('Multi-Layer Perceptron' + '\n')
f.write('test RMSLE = ' + str(rmsle_nn))
f.write('test R^2 = ' + str(r2_nn))
f.write('test Adj R^2 = '+ str(adj_r2_nn))
f.write('test MAE = ' + str(mae_nn))

f.close()

visualizer = ResidualsPlot(nn)
#visualizer = PredictionError(knn_reg)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()
