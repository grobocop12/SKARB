from sklearn.neural_network import MLPClassifier
import moreshoters


import numpy as np
from sklearn.pipeline import make_pipeline

from matplotlib import pyplot as plt


plt.style.use('bmh')
def make_data():
    N = 2000
    X = 0.5*np.random.normal(size=N)+0.35

    Xt = 0.75*X-0.35
    X = X.reshape((N,1))

    Y = -(8 * Xt**2 + 0.1*Xt + 0.1) + 0.05 * np.random.normal(size=N)
    Y = np.exp(Y) + 0.05 * np.random.normal(size=N)
    Y /= max(np.abs(Y))
    return X, Y

np.random.seed(0)

X, Y = moreshoters.gentab(100)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.5, random_state=0)

plt.plot(Xtest[:,0], Ytest, '.');
plt.show()

from sklearn.linear_model import Ridge

ridge = Ridge()
ridge.fit(Xtrain, Ytrain)

Yguess = ridge.predict(Xtest)

plt.plot(Xtest[:,0], Ytest, '.')
plt.plot(Xtest[:,0], Yguess, 'r.')

mean_squared_error(Ytest, Yguess), r2_score(Ytest, Yguess)
plt.show()

from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(random_state=0)
mlp.fit(Xtrain, Ytrain)

Yguess = mlp.predict(Xtest)

plt.plot(Xtest[:,0], Ytest, '.')
plt.plot(Xtest[:,0], Yguess, 'r.')

mean_squared_error(Ytest, Yguess), r2_score(Ytest, Yguess)
plt.show()

from lecture7 import FeedforwardNN

nn = FeedforwardNN(regression=True, n_iter=400, n_hidden=16, verbose=False, plot=True)
nn.fit(Xtrain, Ytrain)

Yguess = nn.predict(Xtest)

plt.plot(Xtest[:,0], Ytest, '.')
plt.plot(Xtest[:,0], Yguess, 'r.')

mean_squared_error(Ytest, Yguess), r2_score(Ytest, Yguess)
plt.show()