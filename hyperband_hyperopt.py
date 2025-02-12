import pandas as pd
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def train_model(pipeline, x_train, y_train, search_space):
    def objective(params):
        pipeline.set_params(**params)
        score = cross_val_score(pipeline, x_train, y_train, cv=3, scoring='f1', n_jobs=-1)
        score = 1 - score.mean()
        return {"score": score}

    hyperband = ASHAScheduler(metric="score", mode="min")
    algo = HyperOptSearch(metric="score", mode="min")

    tuner = tune.Tuner(
        objective,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=hyperband,
            num_samples=20,
            search_alg=algo,
        )
    )  # ③

    results = tuner.fit()
    best_config = results.get_best_result(metric="score", mode="min").config
    pipeline.set_params(**best_config)
    pipeline.fit(x_train, y_train)
    return pipeline


if __name__ == "__main__":
    rf_pipeline = Pipeline(steps=[
        ('model', RandomForestClassifier())
    ])

    rf_search_space = {  # ②
        "model__max_depth": tune.randint(3, 16),
        "model__min_samples_leaf": tune.uniform(0.001, 0.01),
        "model__max_features": tune.choice(['log2', 'sqrt']),
    }

    df = pd.read_csv('student_performance.csv')
    df = df[['gpa_b_and_up', 'Age', 'StudyTimeWeekly']]
    y_df = df['gpa_b_and_up']
    x_df = df.drop('gpa_b_and_up', axis=1)

    best_pipe = train_model(rf_pipeline, x_df, y_df, rf_search_space)
    print(best_pipe)
