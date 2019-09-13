#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import pandas as pd

path  = r"C:\Users\ejgoldbe\Downloads\airbnbnyc.csv"
airbnb_nyc = pd.read_csv(path)
airbnb_nyc = airbnb_nyc.drop(['host_name', 'name', 'last_review', 'reviews_per_month', 'id', 'host_id', 'latitude', 'longitude'], axis = 1)
airbnc_nyc = airbnb_nyc.dropna()

def get_best_params(gr, X_train, y_train):
    number_trees = list(range(100, 1100))
    max_depth = list(range(2, 20))
    param_grid = {'n_estimators': number_trees, 'max_depth': max_depth }
    grid_search = GridSearchCV(gr, param_grid, cv=3)
    grid_search.fit(X_train, y_train)
    params = grid_search.best_params_
    return params


X = airbnb_nyc.drop('price', axis = 1)
y = airbnb_nyc['price']
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)
boost = GradientBoostingRegressor()
params = get_best_params(boost, X_train, y_train)
boost = GradientBoostingRegressor(**params)
#refit model
boost.fit(X_train, y_train)
display(boost.score(X_test, y_test))



select = SelectFromModel(GradientBoostingRegressor(), threshold = "median")
select.fit(X_train, y_train)
X_train_transformed = select.fit(X_train)
print("X_train shape: {} ".format(X_train.shape))
print("X_train_transformed shape: {}" .format(X_train_transformed.shape))
X_test_transformed = select.fit(X_test)
#get best parameters with same model, but reduced features
params = get_best_params(boost, X_train_transformed, y_train)
boost_two = GradientBoostingRegressor(**params)
boost_two.fit(X_train_transformed, y_train)
display(boost_two.score(X_test_transformed, y_test))


pred_prices_one = boost.predict(X_test)
pred_prices_two = boost_two.predict(X_test_transformed)
plt.figure(figsize = (15, 10))
plt.plot(pred_prices_one, color = 'blue', label = 'GBR with Non-Reduced Features')
plt.plot(pred_prices_two, color = 'green', label = 'GBR with Reduced Features')
plt.plot(y_test, color = 'red', label = 'Actual Prices')
plt.show()


# In[ ]:




