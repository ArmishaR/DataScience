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
from yellowbrick.regressor import ResidualsPlot, PredictionError, CooksDistance, AlphaSelection
import matplotlib.pyplot as plt
import seaborn as sns
import time

df = pd.read_csv('training_data_woEncoding_threshold_0.8.csv')

# Encoding
le = preprocessing.LabelEncoder()
le.fit(df['primary_use'])
df['primary_use'] = le.transform(df['primary_use'])

labels = df['meter_reading']
data = df.drop(columns=['meter_reading'])

# Open to print results
f = open('rf_results.txt','a+')

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3, random_state=42)

""" Model Predictions of Test Set """

'''
t = time.process_time()
reg_tree = tree.DecisionTreeRegressor(criterion='friedman_mse')
reg_tree = reg_tree.fit(X_train, y_train)
elapsed_train = time.process_time() - t
t = time.process_time()
predictions = reg_tree.predict(X_test)
elapsed_test = time.process_time() - t

rmsle_dt = np.sqrt(mean_squared_log_error(y_test,predictions))
mae_dt = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions, multioutput='variance_weighted')
adj_r2 = 1 - ((1 - r2)*((len(y_test)-1)/(len(y_test)-len(X_test.columns) - 1)))
r2_dt = r2
adj_r2_dt = adj_r2
'''


t = time.process_time()
print("1")
rf = RandomForestRegressor(n_estimators=25, criterion='mse')
print("2")
rf = rf.fit(X_train, y_train)
elapsed_train = time.process_time() - t
t = time.process_time()
print("3")
predictions = rf.predict(X_test)
print("4")
elapsed_test = time.process_time() - t

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
'''
t = time.process_time()
knn_reg = KNeighborsRegressor(n_neighbors=2,metric='manhattan').fit(X_train,y_train)
elapsed_train = time.process_time() - t
t = time.process_time()
predictions = knn_reg.predict(X_test)
elapsed_test = time.process_time() - t

rmsle_knn = np.sqrt(mean_squared_log_error(y_test,predictions))
mae_knn = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions, multioutput='variance_weighted')
adj_r2 = 1 - ((1 - r2)*((len(y_test)-1)/(len(y_test)-len(X_test.columns) - 1)))
r2_knn = r2
adj_r2_knn = adj_r2
'''


'''
t = time.process_time()
nn = MLPRegressor(hidden_layer_sizes=(5,),activation='logistic',max_iter=1000).fit(X_train,y_train)
elapsed_train = time.process_time() - t
t = time.process_time()
predictions = nn.predict(X_test)
elapsed_test = time.process_time() - t


rmsle_nn = np.sqrt(mean_squared_log_error(y_test,predictions))
mae_nn = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions, multioutput='variance_weighted')
adj_r2 = 1 - ((1 - r2)*((len(y_test)-1)/(len(y_test)-len(X_test.columns) - 1)))
r2_nn = r2
adj_r2_nn = adj_r2
'''

'''
print('training time: ', elapsed_train)
print('prediction time: ', elapsed_test)
print('test RMSLE = ',np.mean(rmsle_dt))
print('test R^2 = ',np.mean(r2_dt))
print('test Adj R^2 = ',np.mean(adj_r2_dt))
print('test MAE = ',np.mean(mae_dt))

f.write('Decision Tree' + '\n')
f.write('training time: ' + str(elapsed_train) + 's \n')
f.write('prediction time: ' + str(elapsed_test) + 's \n')
f.write('test RMSLE = ' + str(rmsle_dt) +'\n')
f.write('test R^2 = ' + str(r2_dt) +'\n')
f.write('test Adj R^2 = '+ str(adj_r2_dt) +'\n')
f.write('test MAE = ' + str(mae_dt) +'\n')
'''


print('training time: ', elapsed_train)
print('prediction time: ', elapsed_test)
print('test RMSLE = ',rmsle_rf)
print('test R^2 = ', r2_rf)
print('test Adj R^2 = ', adj_r2_rf)
print('test MAE = ', mae_rf)

f.write('Random Forest' + '\n')
f.write('training time: ' + str(elapsed_train) + 's \n')
f.write('prediction time: ' + str(elapsed_test) + 's \n')
f.write('test RMSLE = ' + str(rmsle_rf) +'\n')
f.write('test R^2 = ' + str(r2_rf) +'\n')
f.write('test Adj R^2 = '+ str(adj_r2_rf) +'\n')
f.write('test MAE = ' + str(mae_rf) +'\n')


'''print('test RMSLE = ', rslme_linear)
print('test R^2 = ', r2_linear)
print('test Adj R^2 = ', adj_r2_linear)
print('test MAE = ', mae_linear)'''

'''
print('training time: ', elapsed_train)
print('prediction time: ', elapsed_test)
print('test RMSLE = ', rmsle_knn)
print('test R^2 = ', r2_knn)
print('test Adj R^2 = ', adj_r2_knn)
print('test MAE = ', mae_knn)

f.write('KNN' + '\n')
f.write('training time: ' + str(elapsed_train) + 's \n')
f.write('prediction time: ' + str(elapsed_test) + 's \n')
f.write('test RMSLE = ' + str(rmsle_knn) +'\n')
f.write('test R^2 = ' + str(r2_knn) +'\n')
f.write('test Adj R^2 = '+ str(adj_r2_knn) +'\n')
f.write('test MAE = ' + str(mae_knn) +'\n')
'''

'''
print('training time: ', elapsed_train)
print('prediction time: ', elapsed_test)
print('test RMSLE = ', rmsle_nn)
print('test R^2 = ', r2_nn)
print('test Adj R^2 = ', adj_r2_nn)
print('test MAE = ', mae_nn)

f.write('Multi-Layer Perceptron' + '\n')
f.write('training time: ' + str(elapsed_train) + 's \n')
f.write('prediction time: ' + str(elapsed_test) + 's \n')
f.write('test RMSLE = ' + str(rmsle_nn) +'\n')
f.write('test R^2 = ' + str(r2_nn) +'\n')
f.write('test Adj R^2 = '+ str(adj_r2_nn) +'\n')
f.write('test MAE = ' + str(mae_nn) +'\n')
'''

f.close()

'''labels = pd.DataFrame(y_test, columns=["meter_reading"])
preds = pd.DataFrame(predictions, columns=["meter_reading"])
#labels["match"] = labels.eq(preds)
#labels["diff"] = labels["meter_reading"] - preds["meter_reading"]

print(len(labels))
print(len(preds))
labels["category"] = "true value"
labels["idx"] = [i for i in range(0, len(labels))]
preds["category"] = "predicted value"
preds["idx"] = [i for i in range(0, len(preds))]
labels_preds = labels.append(preds)'''

#print(labels_preds_match)

'''sns.set(style="darkgrid")
ax = sns.scatterplot(x="idx", y="diff", data=labels)#, hue="category", style="category")
plt.show()'''


visualizer = ResidualsPlot(rf)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()

visualizer = PredictionError(rf)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show()
