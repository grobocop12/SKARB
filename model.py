#! /usr/bin/python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

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


data = pd.read_csv('genfile_file4.csv', encoding= 'cp1250')
data = data.dropna()

print('Data sample')
print(data.head())

no_wind_data = data.drop('wektor wiatru',axis =1)
train_set, test_set = train_test_split(no_wind_data, test_size=0.2, random_state = 2137)

X_train = train_set.drop(['X','Z'],axis=1)
Y_train = train_set[['X','Z']].copy()

X_test = test_set.drop(['X','Z'],axis=1)
Y_test = test_set[['X','Z']].copy()

print(len(X_train))
print(len(Y_train))

print(len(X_test))
print(len(Y_test))

#linear model

lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train)

predictions = lin_reg.predict(X_test)

final_mse = mean_squared_error(Y_test, predictions)
final_rmse = np.sqrt(final_mse)

print('Linear model')
print('Mean squared error:',final_rmse)

plt.figure(1)
plt.scatter(Y_test['X'],
            Y_test['Z'],
            label='Test set',
            color ='blue'
            )
plt.scatter(predictions[:,0],
            predictions[:,1],
            label='Predictions',
            color ='red')
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear model')

# model random forest
random_forest = RandomForestRegressor(max_features=5,
                                    n_estimators=100,
                                    bootstrap=True,
                                    random_state=0)

random_forest.fit(X_train,Y_train)

predictions = random_forest.predict(X_test)


final_mse = mean_squared_error(Y_test, predictions)
final_rmse = np.sqrt(final_mse)

print('Random Forest model')
print('Mean squared error:',final_rmse)

plt.figure(2)
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
plt.ylabel('Y')
plt.title('Random Forest model')

#Decission tree

decison_tree = DecisionTreeRegressor()

decison_tree.fit(X_train,Y_train)

predictions = decison_tree.predict(X_test)


final_mse = mean_squared_error(Y_test, predictions)
final_rmse = np.sqrt(final_mse)

print('Decision Tree Regressor model')
print('Mean squared error:',final_rmse)

plt.figure(3)
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
plt.ylabel('Y')
plt.title('Decision Tree Regressor')

plt.show()


