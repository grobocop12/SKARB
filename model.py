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
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values=str, strategy='median', axis=0)

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

labels = ('Test set','Predictions')


data = pd.read_csv('genfile_file_seed145_winforcezyx_big.csv', encoding= 'cp1250')
data = data.dropna()
#data = imp.fit_transform(data)

print('Data sample')
print(data.head())

#no_wind_data = data.drop('wektor wiatru',axis =1)

train_set, test_set = train_test_split(data, test_size=0.2, random_state = 2137)

X_train = train_set.drop(['X','Z'],axis=1)


Y_train = train_set[['X','Z']].copy()


X_test = test_set.drop(['X','Z'],axis=1)


Y_test = test_set[['X','Z']].copy()



print(len(X_train))
print(len(Y_train))

print(len(X_test))
print(len(Y_test))

#linear model
'''
lin_reg = LinearRegression()
lin_reg.fit(X_train,Y_train)

predictions = lin_reg.predict(X_test)

final_mse = mean_squared_error(Y_test, predictions)
final_rmse = np.sqrt(final_mse)/len(X_test)

print('Linear model')
print('Mean squared error:',final_rmse)

plt.figure(1)
plt.scatter(Y_test[:,0],
            Y_test[:,1],
            label='Test set',
            color ='blue'
            )
plt.scatter(predictions[:,0],
            predictions[:,1],
            label='Predictions',
            color ='red')
for i in range(len(Y_test)):
    plt.plot( [Y_test[i,0] , predictions[i,0]],[Y_test[i,1], predictions[i,1]],'g-')
    
plt.grid()
plt.xlabel('X')
plt.ylabel('Z')
plt.title('Linear model')
plt.legend(labels)
gc.collect()
'''
# model random forest

random_forest = RandomForestRegressor(max_features=8,
                                    n_estimators=250,
                                    bootstrap=True,
                                    random_state=0)

random_forest.fit(X_train,Y_train)

predictions = random_forest.predict(X_test)


final_mse = mean_squared_error(Y_test, predictions)
final_rmse = np.sqrt(final_mse)/len(X_test)

print('Random Forest model')
print('Mean squared error:',final_rmse)


plt.figure(2)

plt.scatter(Y_test.iloc[:,0],
            Y_test.iloc[:,1],
            label='Test set',
            color ='blue')


plt.scatter(predictions[:,0],
            predictions[:,1],
            label='Predictions',
            color ='red')
            

for i in range(len(Y_test)):
    plt.plot( [Y_test.iloc[i,0] , predictions[i,0]],[Y_test.iloc[i,1], predictions[i,1]],'g-')
   


plt.grid()
plt.xlabel('X')
plt.ylabel('Z')
plt.title('Random Forest model')
plt.legend(labels)
gc.collect()

#a = np.array([20,20,0,0.1,1,0,1,1100])
#a = a.reshape(1,-1)
#print(random_forest.predict(a))


#Decission tree
'''
decison_tree = DecisionTreeRegressor()

decison_tree.fit(X_train,Y_train)

predictions = decison_tree.predict(X_test)


final_mse = mean_squared_error(Y_test, predictions)
final_rmse = np.sqrt(final_mse)/len(X_test)

print('Decision Tree Regressor model')
print('Mean squared error:',final_rmse)

plt.figure(3)
plt.scatter(Y_test.iloc[:,0],
            Y_test.iloc[:,1],
            label='Test set',
            color ='blue')

plt.scatter(predictions[:,0],
            predictions[:,1],
            label='Predictions',
            color ='red')

for i in range(len(Y_test)):
    plt.plot( [Y_test[i,0] , predictions[i,0]],[Y_test[i,1], predictions[i,1]],'g-')

plt.grid()
plt.xlabel('X')
plt.ylabel('Z')
plt.title('Decision Tree Regressor')
plt.legend(labels)

gc.collect()
'''
plt.show()

