import sys

import random

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score


def create_custom_ts_cv_splits(df: pd.DataFrame, start_year: int, end_year: int, cv_folds: int) -> list:
    """
    Creates a set of custom cross validation splits based on year. The function takes a start year and an end year along
    with a number of cv folds. Based on the number of years between the provided years, it will create an equal number
    of array splits based on cv folds. The folds are arranged as a time-series cross validation problem. That is,
    in the first split, the first split is the training data and the second split is the testing data. In the second
    split, the first two splits are the training data, and the third split is the testing data. And so on.

    :param df: pandas dataframe of training data
    :param start_year: start year of the cross validation folds
    :param end_year: end year of the cross validation folds
    :param cv_folds: number of cv folds
    :return: list of tuples, with each tuple containing two items - the first is the index of the training observations
    and the second is the index of the testing observations
    """
    cv_splits = []
    years = list(np.arange(start_year, end_year + 1, 1))
    year_splits = np.array_split(years, cv_folds)
    for n, year_split in enumerate(year_splits):
        if n != cv_folds - 1:
            train_ids = year_splits[:n + 1]
            train_ids = np.concatenate(train_ids)
            test_ids = year_splits[n + 1]
            train_indices = df.loc[df['year'].isin(train_ids)].index.values.astype(int)
            test_indices = df.loc[df['year'].isin(test_ids)].index.values.astype(int)
            cv_splits.append((train_indices, test_indices))
    return cv_splits


def create_custom_panel_cv(train_df, individual_id, folds):
    """
    Creates a custom train-test split for cross validation. This helps prevent leakage of individual-level
    effects.

    :param train_df: pandas dataframe
    :param individual_id: the id column to uniquely identify individual players
    :param folds: the number of folds we want to use in k-fold cross validation
    :return: a list of tuples; each list item represent a fold in the k-fold cross validation; the first tuple element
    contains the indices of the training data, and the second tuple element contains the indices of the testing data
    """
    unique_ids = list(set(train_df[individual_id].tolist()))
    test_set_id_sets = np.array_split(unique_ids, folds)
    cv_splits = []
    for test_set_id_set in test_set_id_sets:
        temp_train_ids = train_df.loc[~train_df[individual_id].isin(test_set_id_set)].index.values.astype(int)
        temp_test_ids = train_df.loc[train_df[individual_id].isin(test_set_id_set)].index.values.astype(int)
        cv_splits.append((temp_train_ids, temp_test_ids))
    return cv_splits


if __name__ == "__main__":
    date_range = pd.date_range(start='1950-01-01', end='2023-01-01', freq='Y')
    time_series_target = np.random.randint(0, 100, size=len(date_range))
    time_series_predictor = np.random.randint(0, 100, size=len(date_range))
    time_series_df = pd.DataFrame({'date': date_range, 'target': time_series_target, 'predictor': time_series_predictor})
    time_series_df['year'] = time_series_df['date'].astype(str).str[:4].astype(int)
    time_series_df = time_series_df.drop('date', axis=1)
    print(time_series_df.head())
    print()

    cv_splits = create_custom_ts_cv_splits(time_series_df, start_year=1950, end_year=2023, cv_folds=5)
    for cv_split in cv_splits:
        print(cv_split[0])
        print(cv_split[1])
        print()

    y = time_series_df['target']
    x = time_series_df.drop('target', axis=1)

    score = cross_val_score(Lasso(), x, y, cv=cv_splits, scoring='neg_mean_squared_error', n_jobs=-1)
    print(score)

    import sys
    sys.exit()

    ###################################################################################################################

    n_obs = 500
    panel_target = np.random.randint(0, 100, size=n_obs)
    panel_predictor = np.random.randint(0, 100, size=n_obs)
    states = [random.choice(['MO', 'KS', 'OK', 'IA', 'IL']) for _ in range(n_obs)]
    panel_df = pd.DataFrame({
        'target': panel_target,
        'predictor': panel_predictor,
        'state': states
    })
    print(panel_df.head(10))
    print()

    cv_splits = create_custom_panel_cv(train_df=panel_df, individual_id='state', folds=3)
    for cv_split in cv_splits:
        fold_train = panel_df.loc[panel_df.index.isin(cv_split[0])]
        fold_test = panel_df.loc[panel_df.index.isin(cv_split[1])]
        print('training fold states')
        print(fold_train['state'].unique())
        print('testing fold states')
        print(fold_test['state'].unique())
        print()
