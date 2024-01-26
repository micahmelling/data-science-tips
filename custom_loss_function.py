import numpy as np
import pandas as pd
from hyperopt import hp
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from hyperparameter_optimization import train_model_with_hyperopt


def calculate_custom_loss(y_true, y_pred, over_penalty):
    y_true = y_true.to_frame()
    y_true = y_true.reset_index(drop=True)
    y_true.columns = ['y_true']

    if isinstance(y_pred, np.ndarray):
        y_pred = pd.DataFrame(y_pred, columns=['y_pred'])
    else:
        y_pred = y_pred.to_frame()
        y_pred.columns = ['y_pred']

    y_pred = y_pred.reset_index(drop=True)
    temp_df = pd.concat([y_true, y_pred], axis=1)

    temp_df['error'] = temp_df['y_pred'] - temp_df['y_true']
    temp_df['abs_error'] = abs(temp_df['error'])
    temp_df['over_prediction'] = np.where(
        temp_df['error'] > 0,
        1,
        0
    )
    temp_df['loss'] = np.where(
        temp_df['over_prediction'] == 1,
        temp_df['abs_error'] * over_penalty,
        temp_df['abs_error']
    )

    score = temp_df['loss'].mean()
    return score


if __name__ == "__main__":
    x, y = load_diabetes(return_X_y=True, as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    cv_scorer = make_scorer(calculate_custom_loss, greater_is_better=False, over_penalty=2)

    param_grid = {
        'model__max_depth': hp.uniformint('model__max_depth', 3, 16),
        'model__min_samples_leaf': hp.uniform('model__min_samples_leaf', 0.001, 0.01),
        'model__max_features': hp.choice('model__max_features', ['log2', 'sqrt']),
    }

    pipeline = Pipeline(steps=[('model', RandomForestRegressor())])
    best_model = train_model_with_hyperopt(
        estimator=pipeline,
        x_train=x_train,
        y_train=y_train,
        model_uid='random_forest',
        param_space=param_grid,
        iterations=5,
        cv_strategy=5,
        cv_scoring=cv_scorer
    )
