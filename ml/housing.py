import os
import tarfile
from six.moves import urllib

# pobranie danych
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join('datasets','housing')
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, 
                       housing_path = HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path,'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#wczytanie danych z pliku csv
import pandas as pd
    
def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = os.path.join(housing_path,'housing.csv')
    return pd.read_csv(csv_path)

housing = load_housing_data()

#wypisanie niektórych informacji o danych
print(housing.head())
print(housing.info())
print(housing.describe())

#wyświetlenie wykresów danych


import matplotlib.pyplot as plt
'''
housing.hist(bins = 50, figsize =(20,15))
plt.show()
'''

#podzielenie danych na zbiór uczący i testowy

import numpy as np

#podzielenie całkowicie przypadkowe
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) *test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

#train_set , test_set = split_train_test(housing,0.2)

#print(len(train_set),len(test_set))

#podzielenie za pomocą funkcji skrótu
import hashlib

def test_set_check(identifier,test_ratio,hash):
    return hash(np.int64(identifier)).digest()[-1]<256 *test_ratio

def split_train_test_by_id(data, test_ratio, id_column, hash = hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_,test_ratio,hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing_with_id = housing.reset_index() #adds index column
train_set, test_set = split_train_test_by_id(housing_with_id,0.2, 'index')

# podzielenie fukcją wbudowaną w sklearn
from sklearn.model_selection import train_test_split

#train_set, test_set = train_test_split(housing, test_size=0.2,random_state = 42)


#dodanie kategorii wielkości przychodu do danych

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

#podzielenie danych, aby zawierały
#reprezentatywne próbki z każdej kategorii
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#usunięcie kategorii wielkości przychodu
for set_ in (strat_train_set,strat_test_set):
    set_.drop('income_cat', axis=1, inplace= True)

#print(strat_test_set.head())

#
housing = strat_train_set.copy()


#wyświetlenie danych według szerokości i długości geo.
#housing.plot(kind='scatter', x='longitude', y= 'latitude', alpha=0.4,
#             s=housing['population']/100, label='population', figsize=(10,7),
#             c= 'median_house_value',cmap=plt.get_cmap('jet'),colorbar = True)
#plt.legend()
#plt.show()

corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending = False))

from pandas.plotting import scatter_matrix

attributes = ['median_house_value','median_income',
              'total_rooms','housing_median_age']

# wykresy korelacji

#scatter_matrix(housing[attributes],figsize=(12,8))
#housing.plot(kind='scatter', x= 'median_income', y='median_house_value',
#             alpha = 0.1)
#plt.show();

housing['rooms_per_houselhold'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']

corr_matrix = housing.corr()
print(corr_matrix['median_house_value'].sort_values(ascending= False))

housing = strat_train_set.drop('median_house_value',axis=1)
housing_labels = strat_train_set['median_house_value'].copy()


# wypełnienie pustych wartoście w total bedrooms
median = housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(median,inplace = True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')

#pobranie tylko wartości numerycznych
housing_num = housing.drop('ocean_proximity',axis =1)

imputer.fit(housing_num)

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X,columns=housing_num.columns)


#zamiana etykiet tekstowych na liczby

from sklearn.preprocessing import LabelEncoder
housing_cat = housing['ocean_proximity']

'''
encoder = LabelEncoder()
housing_cat_encoded = encoder.fit_transform(housing_cat)


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
print(housing_cat_1hot)
'''

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()

housing_cat_1hot = encoder.fit_transform(housing_cat)


# przykład customowego Transformera
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix , household_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self,add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    
    def fit(self,X, y=None):
        return self
    
    def transform(self, X, y=None):
        rooms_per_household = X[:,rooms_ix]/X[:,household_ix]
        population_per_household = X[:,population_ix]/X[:,household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        
        else:
            return np.c_[X,rooms_per_household, population_per_household]
        
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


#pipeline

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)

#Transformer przetwarzający DataFrame z Pandas

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self, X, y= None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values

#LabelBinarizer dla pipeline

class LabelBinarizerPipelineFriendly(LabelBinarizer):
    def fit(self, X, y=None):
        """this would allow us to fit the model based on the X input."""
        super(LabelBinarizerPipelineFriendly, self).fit(X)
    def transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).transform(X)
    def fit_transform(self, X, y=None):
        return super(LabelBinarizerPipelineFriendly, self).fit(X).transform(X)
    
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', LabelBinarizerPipelineFriendly())
    ])

from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
    ])


housing_prepared = full_pipeline.fit_transform(housing)
#print(housing_prepared)


#regresja liniowa
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

#działający przykład regresji liniowej
'''
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print('Predictions:',lin_reg.predict(some_data_prepared))
print('Labels:', list(some_labels))
'''


#Decision tree

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                          scoring = 'neg_mean_squared_error', cv = 10)

tree_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print('Scores:',scores)
    print('Mean:',scores.mean())
    print('Standard deviation:',scores.std())
    
display_scores(tree_rmse_scores)
print()

# porównanie z modelem liniowym

lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring='neg_mean_squared_error', cv = 10)

lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

#model RandomForest
print()
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                             scoring='neg_mean_squared_error', cv = 10)

forest_scores_rmse = np.sqrt(-forest_scores)
display_scores(forest_scores_rmse)