import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header=None)

X = df.loc[:, 2:].values
y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

param_grid = [{'max_depth': [1,2,3,4,5,6,7,None]}]
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0), param_grid=param_grid, scoring='accuracy', cv=5)
scores = cross_val_score(gs, X_train, y_train, scoring='accuracy', cv=2)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
#gs = gs.fit(X_train, y_train)
#print(gs.best_score_)
#print(gs.best_params_)
#
#clf = gs.best_estimator_
#clf.fit(X_train, y_train)
#print('Test accuracy: %.3f' % clf.score(X_test, y_test))