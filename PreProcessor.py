import pandas as pd


def clean(df):
    df_objects = df.select_dtypes(include=['object'])
    df_numeric = df.drop(df_objects, 1)
    # print('Numeric: ', df_numeric.columns, '\nObject: ', df_objects.columns)
    for column in df_numeric.columns:
        # print(df_numeric[column].isnull().sum())
        if df_numeric[column].isnull().sum() > 0:
            median = df_numeric[column].median()
            df_numeric[column].fillna(median, inplace=True)
        # print(df_numeric[column].isnull().sum(), '\nNext')
    # for column in df_objects.columns:
    #     print(df_objects[column].isnull().sum())
    for column in df_objects.columns:
        if df_objects[column].isnull().sum() > 0:
            df_objects[column].fillna('NA', inplace=True)
    # for column in df_objects.columns:
    #     print(df_objects[column].isnull().sum())
    new_df = df_objects.join(df_numeric)
    # print(new_df)
    return new_df


if __name__ == '__main__':
    train_set = pd.read_csv('train.csv')
    train_set.set_index('PassengerId', drop=True, inplace=True)
    test_set = pd.read_csv('test.csv')
    test_set.set_index('PassengerId', drop=True, inplace=True)
    survived = train_set.Survived.values
    survived_df = pd.DataFrame({'Survivor': survived})
    survived_df.to_csv('survived.csv')
    train_set.drop('Survived', 1, inplace=True)
    train_set.drop(['Cabin', 'Name', 'Ticket'], 1, inplace=True)
    test_set.drop(['Cabin', 'Name', 'Ticket'], 1, inplace=True)
    clean_train = clean(train_set)
    clean_train.to_csv('preprocessed_train.csv')
    clean_test = clean(test_set)
    clean_test.to_csv('preprocessed_test.csv')
    # print(clean_train, '\n\n\nNext\n\n\n', clean_test)
