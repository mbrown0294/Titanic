import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split, GridSearchCV


'''
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
'''


def time_passed(start, end):
    elapsed = end - start
    day_tup = divmod(elapsed.total_seconds(), 86400)
    days = day_tup[0]
    hour_tup = divmod(day_tup[1], 3600)
    hours = hour_tup[0]
    min_tup = divmod(hour_tup[1], 60)
    minutes = min_tup[0]
    seconds = min_tup[1]
    if days > 0:
        print("\nDays: ", days)
    if hours > 0:
        print("Hours: ", hours)
    if minutes > 0:
        print("Minutes: ", minutes)
    print("Seconds: ", seconds)


if __name__ == '__main__':
    time_start = datetime.now()
    print(str(time_start), "\n")

    train_set = pd.read_csv('featurized_train.csv')
    test_set = pd.read_csv('featurized_test.csv')
    train_index = train_set['PassengerId'].values
    test_ind = test_set['PassengerId'].values
    train_set.set_index('PassengerId', drop=True, inplace=True)
    test_set.set_index('PassengerId', drop=True, inplace=True)

    X = train_set.values  # Shape: (891, 8)
    y = pd.read_csv('survived.csv')  # Shape after transformations: (891,)
    y.set_index(train_index, inplace=True)
    y.drop('Unnamed: 0', 1, inplace=True)
    y = y.squeeze()
    X_test = test_set.values  # Shape: (418, 8)

    metric = mean_squared_log_error
    model = RandomForestRegressor()

    X_train, X_val, y_train, y_val = train_test_split(
       X, y, test_size=0.33, random_state=42)    # X_train.shape = (596, 8)    # y_train.shape = (596,)
    y_train = np.squeeze(y_train)

    parameter_candidates = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
    ]

    clf = GridSearchCV(estimator=svm.SVC(), param_grid=parameter_candidates, n_jobs=-1)

    clf.fit(X, y)  # 18 min, 58.80 sec
    '''
        Score (best): 0.789001122334
        C=1000
        gamma='auto'
        kernel=linear
    '''
    
    # clf.fit(X_train, y_train)  # 7 min, 3.42 sec
    ''' 
        Score (best):  0.786912751678
        C=100
        gamma='auto'
        kernel='linear' 
        Val Score:  0.793220338983 
        SVC Score: 0.793220338983 
    '''

    b_c = clf.best_estimator_.C
    b_gamma = clf.best_estimator_.gamma
    b_kernel = clf.best_estimator_.kernel
    print("Score: ", clf.best_score_, "\nC= ", b_c, "\ngamma='", b_gamma, "'", "\nkernel=", b_kernel)
    # print("Val Score: ", clf.score(X_val, y_val))
    svc = svm.SVC(C=b_c, kernel=b_kernel, gamma=b_gamma)

    # svc.fit(X_train, y_train)
    # print("SVC Score: ", svc.score(X_val, y_val))

    svc.fit(X, y)
    prediction = svc.predict(X_test)
    d = {'PassengerId': test_ind, 'Survived': prediction}
    submission = pd.DataFrame(data=d)
    print(submission)
    submission.to_csv('submission.csv', index=False)
    time_end = datetime.now()
    print(str(time_end))
    print(time_passed(time_start, time_end))
