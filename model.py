#! /usr/bin/python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

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

print(data.head())
print(len(data))



print(list(data))

##side_angles = np.array(data['kat boczny'])
##angles = preprocess_side_angle(side_angles)
##data['kat boczny'] = angles

##plt.plot(angles)
##data.hist(bins = 50)
##plt.show()
##

no_wind_data = data.drop('wektor wiatru',axis =1)

##wind = np.array(data['wektor wiatru'])
##print(wind)


train_set, test_set = train_test_split(no_wind_data, test_size=0.2, random_state = 44)



X_train = train_set.drop(['X','Z'],axis=1)
Y_train = train_set[['X','Z']].copy()


X_test = test_set.drop(['X','Z'],axis=1)
Y_test = test_set[['X','Z']].copy()

print(len(X_train))
print(len(Y_train))

print(len(X_test))
print(len(Y_test))

lin_reg = RandomForestRegressor(max_features=5,
                                    n_estimators=50,
                                    bootstrap=True,
                                    random_state=0)

#lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train)

predictions = lin_reg.predict(X_test)
print(np.shape(predictions))

final_mse = mean_squared_error(Y_test, predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
print(len(Y_train))
plt.scatter(Y_test['X'],Y_test['Z'])
plt.scatter(predictions[:,0],predictions[:,1])

plt.show()
