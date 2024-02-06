import joblib
from hyperopt import hp
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from hyperparameter_optimization import train_model_with_hyperopt


class CombineCategoryLevels(BaseEstimator, TransformerMixin):
    def __init__(self, sparsity_cutoff=0.001):
        self.sparsity_cutoff = sparsity_cutoff
        self.mapping_dict = {}
        self.object_columns = []

    def fit(self, X, Y=None):
        self.object_columns = df.select_dtypes(include=['object']).columns.tolist()
        for col in self.object_columns:
            percentages = X[col].value_counts(normalize=True)
            combine = percentages.loc[percentages <= self.sparsity_cutoff]
            combine_levels = combine.index.tolist()
            self.mapping_dict[col] = combine_levels
        return self

    def transform(self, X, Y=None):
        if self.sparsity_cutoff > 0:
            for col in self.object_columns:
                combine_cols = self.mapping_dict.get(col, [None])
                X.loc[X[col].isin(combine_cols), col] = 'sparse_combined'
            return X
        else:
            return X


class CentralityEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, node_column, edge_column):
        self.mapping_dict = {}
        self.node_column = node_column
        self.edge_column = edge_column

    def fit(self, X, Y=None):
        G = nx.from_pandas_edgelist(X, self.node_column, self.edge_column)
        centrality = nx.eigenvector_centrality(G)
        self.mapping_dict.update(centrality)
        return self

    def transform(self, X, Y=None):
        X[self.node_column] = X[self.node_column].map(self.mapping_dict)
        X[self.edge_column] = X[self.edge_column].map(self.mapping_dict)
        return X


if __name__ == "__main__":
    df = pd.DataFrame({
        'pass': [1, 0, 0, 1, 1, 1, 1, 0, 1, 1],
        'teacher': ['a', 'a', 'b', 'a', 'b', 'b', 'c', 'c', 'a', 'd'],
        'advisor': ['b', 'c', 'c', 'a', 'd', 'd', 'a', 'b', 'a', 'c']
    })
    print(df)
    encoder = CentralityEncoder(node_column='teacher', edge_column='advisor')
    df = encoder.fit_transform(df)
    print(df)
    print()
    print(encoder.mapping_dict)
    print()
    joblib.dump(encoder, 'encoder.pkl')
    encoder = joblib.load('encoder.pkl')
    print(encoder.mapping_dict)
    print()

    df2 = pd.DataFrame({
        'pass': [1, 0],
        'teacher': ['a', 'b'],
        'advisor': ['c', 'd']
    })
    print(df2)
    df2 = encoder.transform(df2)
    print(df2)

    import sys
    sys.exit()

    df = pd.DataFrame({
        'target': np.random.randint(0, 2, size=1_000),
        'price': np.random.randint(0, 100, size=1_000),
        'channel': np.random.choice(['a', 'b', 'c', 'd', 'e'],
                                    p=[0.50, 0.40, 0.06, 0.03, 0.01], size=1_000),
        'message': np.random.choice(['a', 'b', 'c', 'd', 'e', 'f', 'g'],
                                    p=[0.50, 0.40, 0.05, 0.02, 0.01, 0.01, 0.01], size=1_000),
    })

    y = df['target']
    x = df.drop('target', axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    FOREST_PARAM_GRID = {
        'category_combiner__sparsity_cutoff': hp.uniform('category_combiner__sparsity_cutoff', 0.0, 0.05),
        'model__max_depth': hp.uniformint('model__max_depth', 3, 16),
        'model__min_samples_leaf': hp.uniform('model__min_samples_leaf', 0.001, 0.01),
        'model__max_features': hp.choice('model__max_features', ['log2', 'sqrt']),
    }

    pipeline = Pipeline(steps=[
        ('category_combiner', CombineCategoryLevels()),
        ('ohc', OneHotEncoder(sparse_output=False, handle_unknown='ignore')),
        ('model', RandomForestClassifier())
    ])

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

    print(best_model)
