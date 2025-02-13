from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import pandas as pd
from ray import tune
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from hyperband_hyperopt import train_model


def create_oversampling_dict(y):
    counter = Counter(y)
    pos_count = counter.get(1)
    pos_count = int(pos_count * 2)
    neg_count = counter.get(0)
    return {0: neg_count, 1: pos_count}


if __name__ == "__main__":
    pipeline = Pipeline(steps=[
        ('oversampling', RandomOverSampler(sampling_strategy=create_oversampling_dict)),
        ('model', RandomForestClassifier())
    ])

    df = pd.read_csv('student_performance.csv')
    df = df[['gpa_b_and_up', 'Age', 'StudyTimeWeekly']]
    y_df = df['gpa_b_and_up']
    x_df = df.drop('gpa_b_and_up', axis=1)

    rf_search_space = {
        "model__max_depth": tune.randint(3, 16),
        "model__min_samples_leaf": tune.uniform(0.001, 0.01),
        "model__max_features": tune.choice(['log2', 'sqrt']),
    }

    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.25, random_state=42)
    best_pipe = train_model(pipeline, x_df, y_df, rf_search_space)
    print(best_pipe)
