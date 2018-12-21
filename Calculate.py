import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import moreshoters
import moreshoters

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def singleshot(Xtarget,Ztarget,windforce,data):
    print('Start')
    data = pd.read_csv(data, encoding= 'cp1250')
    data = data.dropna()

    train_set, test_set = train_test_split(data, test_size=0.2, random_state = 2137)

    X_train = train_set.drop(['kat podniesienia','kat boczny'],axis=1)
    Y_train = train_set[['kat podniesienia','kat boczny']].copy()

    random_forest = RandomForestRegressor(max_features=8,
                                    n_estimators=100,
                                    bootstrap=True,
                                    random_state=0,
                                    criterion='mae')


    random_forest.fit(X_train,Y_train)
    winx = windforce[0]  # wiatr równoległy
    winy = windforce[1]  # wiatr wznoszący
    winz = windforce[2]
    predictions = random_forest.predict(2,0.1,winx,winy,winz,1100,Xtarget,Ztarget)
    alfa = predictions[0]
    beta = predictions[1]
    X, Y, Z = moreshoters.precision(alfa, beta,2, 0.1, windforce, 1100)
    plt.figure(1)
    plt.scatter(Xtarget,
                Ztarget,
                label='Target',
                color='blue')

    plt.scatter(X,
                Z,
                label='Shot',
                color='red')
    print('End')

singleshot(1000,12000,[0.2,0.1,0.2],'genfile_file_seed145_winforcezyx_big_21_12_2018.csv')