import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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


if __name__ == "__main__":
    date_range = pd.date_range(start='1950-01-01', end='2023-01-01', freq='Y')
    time_series_target = np.random.randint(0, 100, size=len(date_range))
    time_series_predictor = np.random.randint(0, 100, size=len(date_range))
    time_series_df = pd.DataFrame({'date': date_range, 'target': time_series_target, 'predictor': time_series_predictor})
    time_series_df['year'] = time_series_df['date'].astype(str).str[:4].astype(int)
    time_series_df = time_series_df.drop('date', axis=1)
    print(time_series_df.head())

    cv_splits = create_custom_ts_cv_splits(time_series_df, start_year=1950, end_year=2023, cv_folds=5)
    print(cv_splits)

    y = time_series_df['target']
    x = time_series_df.drop('target', axis=1)

    score = cross_val_score(LogisticRegression(), x, y, cv=cv_splits, scoring='neg_mean_squared_error', n_jobs=-1)
    print(score)
