import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import RandomizedSearchCV
import pickle

df=pd.read_table("fruits.txt")
df.to_csv("fruits.csv",index=False)
df=pd.read_csv("fruits.csv")
x=df.iloc[:,3:]
y=df.iloc[:,0:1]

#Below lines are used for Hyper parameter tuning for random forest

#x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1)
# Number of trees in random forest
#n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
#max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
#max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
#max_depth.append(None)
# Minimum number of samples required to split a node
#min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
#min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
#bootstrap = [True, False]
# Create the random grid
#random_grid = {'n_estimators': n_estimators,
 #              'max_features': max_features,
 #              'max_depth': max_depth,
 #              'min_samples_split': min_samples_split,
 #            'min_samples_leaf': min_samples_leaf,
  #             'bootstrap': bootstrap}
#print(random_grid)
#{'bootstrap': [True, False],
 #'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
 #'max_features': ['auto', 'sqrt'],
 #'min_samples_leaf': [1, 2, 4],
 #'min_samples_split': [2, 5, 10],
 #'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

 # Use the random grid to search for best hyperparameters
# First create the base model to tune
#rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
#rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
#rf_random.fit(x_train, np.ravel(y_train,order='C'))

#rf_random.best_estimator_

#rf_random.best_params_
#rf_random.best_score_


model=RandomForestClassifier(n_estimators= 2000,min_samples_split= 5,
 min_samples_leaf= 1,
 max_features= 'sqrt',
 max_depth= 10,
 bootstrap= True).fit(x,np.ravel(y,order='C'))

pickle.dump(model,open("fruit.pkl","wb"))



