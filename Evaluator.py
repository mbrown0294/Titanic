import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split, GridSearchCV


def grid_search(trainx, trainy, valx, valy):
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
    print(clf.score(valx, valy))


if __name__ == '__main__':
    train_set = pd.read_csv('featurized_train.csv')
    test_set = pd.read_csv('featurized_test.csv')
    index = train_set['PassengerId'].values
    test_ind = test_set['PassengerId'].values
    train_set.set_index('PassengerId', drop=True, inplace=True)
    test_set.set_index('PassengerId', drop=True, inplace=True)
    # print(train_set, '\n\n\n', test_set)
    metric = mean_squared_log_error
    model = RandomForestRegressor()

    X = train_set.values  # Shape: (891, 8)
    y = pd.read_csv('survived.csv')  # Shape after transformations: (891, 1)
    y.set_index(index, inplace=True)
    y.drop('Unnamed: 0', 1, inplace=True)
    X_test = test_set.values  # Shape: (418, 8)

    X_train, X_val, y_train, y_val = train_test_split(
       X, y, test_size=0.33, random_state=42)    # X_train.shape = (978, 79)    # y_train.shape = (978, 1)
    # print('X_train: ', X_train.shape, '\ny_train: ', y_train.shape)

    # grid_search(X_train, y_train, X_val, y_val)
    parameter_candidates = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
    ]
    y_train = np.squeeze(y_train)
    y = y.squeeze()
# SVC
    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)
    # clf.fit(X_train, y_train)  # Best score:  0.786912751678 ## C= 100 ## kernel='linear' ## gamma='auto'
    # ## Val Score:  0.790268456376
    SVClass = svm.SVC(C=100, kernel='linear', gamma='auto')
    # SVC Score: 0.790268456376
    # SVClass.fit(X_train, y_train)
    SVClass.fit(X, y)
    prediction = SVClass.predict(X_test)

# SVR
    # clf = GridSearchCV(estimator=svm.SVR(), param_grid=parameter_candidates, n_jobs=-1)
    # # clf.fit(X_train, y_train)  # Best score:  0.273422145744 ## C= 100 ## kernel='rbf' ## gamma=0.0001
    # ## Val Score:  0.313698798796
    # SVReg = svm.SVR(C=100, kernel='rbf',gamma=0.0001)
    # # SVM Score:  0.313698798796
    # # SVReg.fit(X_train,y_train)
    # SVReg.fit(X, y)
    # prediction = SVReg.predict(X_test)

# Print Statements
    # print('Best score: ', clf.best_score_)
    # print('\nC=', clf.best_estimator_.C)
    # print("\nkernel='", clf.best_estimator_.kernel, "'")
    # print('\ngamma=', clf.best_estimator_.gamma)
    # print('Val Score: ', clf.score(X_train, y_train))

# Both
    # pred = []
    # for obj in prediction:
    #     pred.append(round(obj))

    d = {'PassengerId': test_ind, 'Survived': prediction}
    submission = pd.DataFrame(data=d)
    print(submission)
    # submission.to_csv('titanic_sub_SVC.csv', index=False)
    submission.to_csv('titanic_sub_SVC.csv', index=False)
