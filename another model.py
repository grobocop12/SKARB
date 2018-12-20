#! /usr/bin/python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import gc

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import ElasticNet

def preprocess_side_angle(angles):
    for i  in range(len(angles)):
        angles[i] = angles[i]%360
        if angles[i] > 90:
            angles[i] = angles[i] - 360
    return angles

def test_set_check(identifier,test_ratio,hash):
    return hash(np.int64(identifier)).digest()[-1]<256 *test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash = hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_,test_ratio,hash))
    return data.loc[~in_test_set], data.loc[in_test_set]

scaler = MinMaxScaler()
labels = ('Test set','Predictions')

data = pd.read_csv('genfile_file_seed145_winforcezx.csv', encoding= 'cp1250')
data = data.dropna()

print('Data sample')
print(data.head())

#no_wind_data = data.drop('wektor wiatru',axis =1)

train_set, test_set = train_test_split(data, test_size=0.2, random_state = 2137)

X_train = train_set.drop(['X','Z'],axis=1)
Y_train = train_set[['X','Z']].copy()

X_test = test_set.drop(['X','Z'],axis=1)
Y_test = test_set[['X','Z']].copy()


kenrle_ridge = ElasticNet()

kenrle_ridge.fit(X_train,Y_train)

predictions = kenrle_ridge.predict(X_test)


final_mse = mean_squared_error(Y_test, predictions)
final_rmse = np.sqrt(final_mse)/len(X_test)

print('Decision Tree Regressor model')
print('Mean squared error:',final_rmse)

plt.figure(1)
plt.scatter(Y_test['X'],
            Y_test['Z'],
            label='Test set',
            color ='blue')

plt.scatter(predictions[:,0],
            predictions[:,1],
            label='Predictions',
            color ='red')
plt.grid()
plt.xlabel('X')
plt.ylabel('Z')
plt.title('Kernel Ridge')
plt.legend(labels)

plt.show()
