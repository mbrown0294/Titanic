import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split, GridSearchCV


def grid_search(trainx, trainy):
    parameter_candidates = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
    ]
    clf = GridSearchCV(estimator=svm.SVR(), param_grid=parameter_candidates, n_jobs=-1)
    trainy = np.squeeze(trainy)
    clf.fit(trainx, trainy)
    print('Best score: ', clf.best_score_)
    print('\nBest C:', clf.best_estimator_.C)
    print('\nBest Kernel:', clf.best_estimator_.kernel)
    print('\nBest Gamma:', clf.best_estimator_.gamma)


if __name__ == '__main__':
    train_set = pd.read_csv('featurized_train.csv')
    test_set = pd.read_csv('featurized_test.csv')
    index = train_set['PassengerId'].values
    train_set.set_index('PassengerId', drop=True, inplace=True)
    test_set.set_index('PassengerId', drop=True, inplace=True)
    # print(train_set, '\n\n\n', test_set)
    metric = mean_squared_log_error
    model = RandomForestRegressor()

    X = train_set.values  # Shape: (891, 8)
    y = pd.read_csv('survived.csv')  # Shape after transformations: (891, 1)
    y.set_index(index, inplace=True)
    y.drop('Id', 1, inplace=True)
    X_test = test_set.values  # Shape: (418, 8)

    X_train, X_val, y_train, y_val = train_test_split(
       X, y, test_size=0.33, random_state=42)    # X_train.shape = (978, 79)    # y_train.shape = (978, 1)
    # print('X_train: ', X_train.shape, '\ny_train: ', y_train.shape)

    grid_search(X_train, y_train)
