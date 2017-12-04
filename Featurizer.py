import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from sklearn.preprocessing import LabelEncoder


def object_clean(train, test):
    train_objects = train.select_dtypes(include=['object'])
    test_objects = test.select_dtypes(include=['object'])
    le = LabelEncoder()
    for column in train_objects.columns:
        train_objects[column] = le.fit_transform(train_objects[column].astype(str))
        test_objects[column] = le.transform(test_objects[column].astype(str))
        train_objects[column] = train_objects[column].astype(str)
        test_objects[column] = test_objects[column].astype(str)
    train_dummies = pd.get_dummies(train_objects, drop_first=True)
    test_dummies = pd.get_dummies(test_objects, drop_first=True)
    train_dummies.drop('Embarked_3', 1, inplace=True)
    train = train.drop(train_objects, 1)
    train = train.join(train_dummies)
    test = test.drop(test_objects, 1)
    test = test.join(test_dummies)
    return train, test


if __name__ == '__main__':
    train_set = pd.read_csv('preprocessed_train.csv')
    train_set.set_index('PassengerId', drop=True, inplace=True)
    test_set = pd.read_csv('preprocessed_test.csv')
    test_set.set_index('PassengerId', drop=True, inplace=True)
    train_set, test_set = object_clean(train_set, test_set)
    train_set.to_csv('featurized_train.csv')
    test_set.to_csv('featurized_test.csv')
    # plt.figure()
    # train_set['Age'].hist(bins=20)
    # plt.xlabel('Age')
    # plt.ylabel('Frequency')
    # plt.title('Titanic Ages')
    # plt.show()

