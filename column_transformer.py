import warnings

import numpy as np
import pandas as pd
from hyperopt import hp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import (SelectPercentile, VarianceThreshold,
                                       chi2, f_classif)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from hyperparameter_optimization import train_model_with_hyperopt

warnings.filterwarnings('ignore')


FOREST_PARAM_GRID = {
    'model__max_depth': hp.uniformint('model__max_depth', 3, 16),
    'model__min_samples_leaf': hp.uniform('model__min_samples_leaf', 0.001, 0.01),
    'model__max_features': hp.choice('model__max_features', ['log2', 'sqrt']),
}


def clip_feature_bounds(df, feature, cutoff, new_amount, clip_type):
    """
    Re-assigns values above or below a certain threshold.

    :param df: pandas dataframe
    :param feature: name of the feature to clip
    :param cutoff: the point beyond which the value is changed
    :param new_amount: the amount to assign points beyond cutoff
    :param clip_type: denotes if we want to change values above or below the cutoff; can either be upper or lower
    :returns: pandas dataframe
    """
    if clip_type == 'upper':
        df[feature] = np.where(df[feature] > cutoff, new_amount, df[feature])
    elif clip_type == 'lower':
        df[feature] = np.where(df[feature] < cutoff, new_amount, df[feature])
    else:
        raise Exception('clip_type must either be upper or lower')
    return df


def fill_missing_values(df, fill_value):
    """
    Fills all missing values in a dataframe with fill_value.

    :param df: pandas dataframe
    :param fill_value: the fill value
    :returns: pandas dataframe
    """
    df = df.fillna(value=fill_value)
    df = df.replace('nan', fill_value)
    return df


class FeaturesToDict(BaseEstimator, TransformerMixin):
    """
    Converts dataframe, or numpy array, into a dictionary oriented by records. This is a necessary pre-processing step
    for DictVectorizer().
    """
    def __int__(self):
        pass

    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X = X.to_dict(orient='records')
        return X


class TakeLog(BaseEstimator, TransformerMixin):
    """
    Based on the argument, takes the log of the numeric columns.
    """

    def __init__(self, take_log='yes'):
        self.take_log = take_log

    def fit(self, X, Y=None):
        return self

    def transform(self, X, Y=None):
        if self.take_log == 'yes':
            for col in list(X):
                X[col] = X[col] + 1
                X[col] = np.log(X[col])
                X[col] = X[col].replace([np.inf, -np.inf], 0)
                return X
        elif self.take_log == 'no':
            return X
        else:
            return X


class CombineCategoryLevels(BaseEstimator, TransformerMixin):
    """
    Combines category levels that individually fall below a certain percentage of the total.
    """
    def __init__(self, combine_categories='yes', sparsity_cutoff=0.001):
        self.combine_categories = combine_categories
        self.sparsity_cutoff = sparsity_cutoff
        self.mapping_dict = {}

    def fit(self, X, Y=None):
        for col in list(X):
            percentages = X[col].value_counts(normalize=True)
            combine = percentages.loc[percentages <= self.sparsity_cutoff]
            combine_levels = combine.index.tolist()
            self.mapping_dict[col] = combine_levels
        return self

    def transform(self, X, Y=None):
        if self.combine_categories == 'yes':
            for col in list(X):
                combine_cols = self.mapping_dict.get(col, [None])
                X.loc[X[col].isin(combine_cols), col] = 'sparse_combined'
            return X
        elif self.combine_categories == 'no':
            return X
        else:
            return X


def get_pipeline(model):
    """
    Generates a scikit-learn modeling pipeline with model as the final step.

    :param model: instantiated model
    :returns: scikit-learn pipeline
    """
    numeric_transformer = Pipeline(steps=[
        ('mouse_movement_clipper', FunctionTransformer(clip_feature_bounds, validate=False,
                                                       kw_args={'feature': 'price', 'cutoff': 0,
                                                                'new_amount': 0, 'clip_type': 'lower'})),
        ('log_creator', TakeLog()),
        ('dict_creator', FeaturesToDict()),
        ('dict_vectorizer', DictVectorizer(sparse=False)),
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('feature_selector', SelectPercentile(f_classif)),
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', FunctionTransformer(fill_missing_values, validate=False,
                                        kw_args={'fill_value': 'missing'})),
        ('category_combiner', CombineCategoryLevels()),
        ('dict_creator', FeaturesToDict()),
        ('dict_vectorizer', DictVectorizer(sparse=False)),
        ('feature_selector', SelectPercentile(chi2)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric_transformer', numeric_transformer, selector(dtype_include='number')),
            ('categorical_transformer', categorical_transformer, selector(dtype_exclude='number'))
        ],
        remainder='passthrough',
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('variance_thresholder', VarianceThreshold()),
        ('model', model)
    ])

    return pipeline


if __name__ == "__main__":
    df = pd.DataFrame({
        'target': np.random.randint(0, 2, size=1_000),
        'price': np.random.randint(0, 100, size=1_000),
        'channel': np.random.choice(['a', 'b', 'c'], size=1_000),
        'medium': np.random.choice(['d', 'e', 'f'], size=1_000),
    })

    y = df['target']
    x = df.drop('target', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    pipeline = get_pipeline(RandomForestClassifier())
    best_model = train_model_with_hyperopt(
        estimator=pipeline,
        x_train=x_train,
        y_train=y_train,
        model_uid='random_forest',
        param_space=FOREST_PARAM_GRID,
        iterations=5,
        cv_strategy=5,
        cv_scoring='neg_log_loss'
    )
