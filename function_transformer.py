import numpy as np
import pandas as pd
from hyperopt import hp
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from hyperparameter_optimization import train_model_with_hyperopt

FEATURES_TO_DROP = ['amount']

FOREST_PARAM_GRID = {
    'model__max_depth': hp.uniformint('model__max_depth', 3, 16),
    'model__min_samples_leaf': hp.uniform('model__min_samples_leaf', 0.001, 0.01),
    'model__max_features': hp.choice('model__max_features', ['log2', 'sqrt']),
}


def drop_features(df, features_list):
    """
    Drops features from a dataframe.

    :param df: pandas dataframe
    :param features_list: list of features to drop
    :returns: pandas dataframe
    """
    df = df.drop(features_list, 1)
    return df


def get_pipeline(model: RegressorMixin or ClassifierMixin) -> Pipeline:
    """
    Creates a scikit-learn modeling pipeline for our modeling problem. In this case, a set of features can be dropped
    per the FEATURES_TO_DROP global defined in modeling.config. A model is then applied.

    :param model: regression or classification model
    :return: scikit-learn pipeline
    """
    pipeline = Pipeline(steps=[
        ('dropper', FunctionTransformer(drop_features, validate=False,
                                        kw_args={
                                            'features_list': FEATURES_TO_DROP
                                        })),
        ('model', model)
        ])
    return pipeline


if __name__ == "__main__":
    df = pd.DataFrame({
        'target': np.random.randint(0, 2, size=1_000),
        'price': np.random.randint(0, 100, size=1_000),
        'amount': np.random.randint(0, 100, size=1_000),
    })

    y = df['target']
    x = df.drop('target', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    pipeline = get_pipeline(RandomForestClassifier())
    best_model = train_model_with_hyperopt(
        estimator=pipeline,
        x_train=x_train,
        y_train=y_train,
        model_uid='random_forst',
        param_space=FOREST_PARAM_GRID,
        iterations=5,
        cv_strategy=5,
        cv_scoring='neg_log_loss'
    )
