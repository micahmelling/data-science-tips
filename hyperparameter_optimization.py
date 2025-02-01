from typing import Union

import joblib
import numpy as np
import pandas as pd
from hyperopt import Trials, fmin, hp, space_eval, tpe
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline


FOREST_PARAM_GRID = {
    'model__max_depth': hp.uniformint('model__max_depth', 3, 16),
    'model__min_samples_leaf': hp.uniform('model__min_samples_leaf', 0.001, 0.01),
    'model__max_features': hp.choice('model__max_features', ['log2', 'sqrt']),
}


def train_model_with_hyperopt(estimator: Union[Pipeline, RegressorMixin, ClassifierMixin],
                              x_train: pd.DataFrame, y_train: Union[pd.DataFrame, pd.Series],
                              model_uid: str, param_space: dict, iterations: int,
                              cv_strategy: Union[int, iter],
                              cv_scoring: str) -> Union[Pipeline, RegressorMixin, ClassifierMixin]:
    """
    Trains a model using Hyperopt.

    :param estimator: an estimator, such as a regression model, a classification model, or a fuller modeling pipeline
    :param x_train: x train
    :param y_train: y train
    :param model_uid: model uid
    :param param_space: parameter search space to optimize
    :param iterations: number of iterations to run the optimization
    :param cv_strategy: cross validation strategy
    :param cv_scoring: how to score cross validation folds
    :return: optimized estimator
    """
    cv_scores_df = pd.DataFrame()

    def _model_objective(params):
        estimator.set_params(**params)
        score = cross_val_score(estimator, x_train, y_train, cv=cv_strategy, scoring=cv_scoring, n_jobs=-1)
        temp_cv_scores_df = pd.DataFrame(score)
        temp_cv_scores_df = temp_cv_scores_df.reset_index()
        temp_cv_scores_df['index'] = 'fold_' + temp_cv_scores_df['index'].astype(str)
        temp_cv_scores_df = temp_cv_scores_df.T
        temp_cv_scores_df = temp_cv_scores_df.add_prefix('fold_')
        temp_cv_scores_df = temp_cv_scores_df.iloc[1:]
        temp_cv_scores_df['mean'] = temp_cv_scores_df.mean(axis=1)
        temp_cv_scores_df['std'] = temp_cv_scores_df.std(axis=1)
        temp_params_df = pd.DataFrame(params, index=list(range(0, len(params) + 1)))
        temp_cv_scores_df = pd.concat([temp_params_df, temp_cv_scores_df], axis=1)
        temp_cv_scores_df = temp_cv_scores_df.dropna()
        nonlocal cv_scores_df
        cv_scores_df = pd.concat([cv_scores_df, temp_cv_scores_df], axis=0)
        return 1 - score.mean()

    trials = Trials()
    best = fmin(_model_objective, param_space, algo=tpe.suggest, max_evals=iterations, trials=trials)
    best_params = space_eval(param_space, best)

    cv_scores_df = cv_scores_df.sort_values(by=['mean'], ascending=False)
    cv_scores_df = cv_scores_df.reset_index(drop=True)
    cv_scores_df = cv_scores_df.reset_index()
    cv_scores_df = cv_scores_df.rename(columns={'index': 'ranking'})
    cv_scores_df.to_csv(f'{model_uid}_cv_scores.csv', index=False)

    estimator.set_params(**best_params)
    estimator.fit(x_train, y_train)

    joblib.dump(estimator, f'{model_uid}_model.pkl')
    return estimator


if __name__ == "__main__":
    df = pd.DataFrame({
        'target': np.random.randint(0, 2, size=1_000),
        'price': np.random.randint(0, 100, size=1_000),
        'amount': np.random.randint(0, 100, size=1_000),
    })

    y = df['target']
    x = df.drop('target', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    pipeline = Pipeline(steps=[('model', RandomForestRegressor())])
    best_model = train_model_with_hyperopt(
        estimator=pipeline,
        x_train=x_train,
        y_train=y_train,
        model_uid='random_forest',
        param_space=FOREST_PARAM_GRID,
        iterations=5,
        cv_strategy=5,
        cv_scoring='neg_mean_squared_error'
    )

    # https://endtoenddatascience.com/chapter10-machine-learning
    # https://towardsdatascience.com/hyperopt-demystified-3e14006eb6fa
    # https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f
