import joblib
import networkx as nx
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CombineCategoryLevels(BaseEstimator, TransformerMixin):
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
    joblib.dump(encoder, 'encoder.pkl')
    encoder = joblib.load('encoder.pkl')
    print(encoder.mapping_dict)
