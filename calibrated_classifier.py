import os
from collections import namedtuple

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
from hyperopt import hp
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from hyperparameter_optimization import train_model_with_hyperopt


FOREST_PARAM_GRID = {
    'model__estimator__max_depth': hp.uniformint('model__estimator__max_depth', 3, 16),
    'model__estimator__min_samples_leaf': hp.uniform('model__estimator__min_samples_leaf', 0.001, 0.01),
    'model__estimator__max_features': hp.choice('model__estimator__max_features', ['log2', 'sqrt']),
}


model_named_tuple = namedtuple('model_config', {'model_name', 'model', 'param_space', 'iterations'})
MODEL_TRAINING_LIST = [
    model_named_tuple(model_name='random_forest', model=CalibratedClassifierCV(estimator=RandomForestClassifier()),
                      param_space=FOREST_PARAM_GRID, iterations=15),
]


def calculate_probability_lift(y_test, probability_predictions, n_bins=10):
    """
    Calculates the lift provided by the probability estimates. Lift is determined by how much improvement is experienced
    by using the predicted probabilities over assuming that each observation has the same probability of being in the
    positive class (i.e. applying the overall rate of occurrence of the positive class to all observations).

    This process takes the following steps:
    - find the overall rate of occurrence of the positive class
    - cut the probability estimates into n_bins
    - for each bin, calculate:
       - the average predicted probability
       - the actual probability
    -  for each bin, calculate
       - the difference between the average predicted probability and the true probability
       - the difference between the overall rate of occurrence and the true probability
    - take the sum of the absolute value for each the differences calculated in the previous step
    - take the ratio of the two sums, with the base rate sum as the numerator

    Values above 1 indicate the predicted probabilities have lift over simply assuming each observation has the same
    probability.

    :param y_test: y_test series
    :param probability_predictions: positive probability predictions series
    :param n_bins: number of bins to segment the probability predictions
    """
    y_test = y_test.reset_index(drop=True)
    prediction_series = probability_predictions.reset_index(drop=True)
    df = pd.concat([y_test, prediction_series], axis=1)
    columns = list(df)
    class_col = columns[0]
    proba_col = columns[1]
    base_rate = df[class_col].mean()

    df['1_prob_bin'] = pd.qcut(df[proba_col], q=n_bins, labels=list(range(1, n_bins + 1)))
    grouped_df = df.groupby('1_prob_bin').agg({proba_col: 'mean', class_col: 'mean'})
    grouped_df.reset_index(inplace=True)
    grouped_df['1_prob_diff'] = grouped_df[proba_col] - grouped_df[class_col]
    grouped_df['base_rate_diff'] = base_rate - grouped_df[class_col]

    prob_diff = grouped_df['1_prob_diff'].abs().sum()
    base_rate_diff = grouped_df['base_rate_diff'].abs().sum()
    lift = base_rate_diff / prob_diff
    pd.DataFrame({'lift': [lift]}).to_csv(os.path.join('proba_lift.csv'), index=False)


def plot_cumulative_gains_chart(y_test, probability_predictions):
    """
    Produces a cumulative gains chart and saves it locally.

    :param y_test: y_test series
    :param probability_predictions: dataframe of probability predictions, with the first column being the negative
    class predictions and the second column being the positive class predictions
    :param model_uid: model uid
    """
    skplt.metrics.plot_cumulative_gain(y_test, probability_predictions)
    plt.savefig(os.path.join('cumulative_gains_plot.png'))
    plt.clf()


def plot_lift_curve_chart(y_test, probability_predictions):
    """
    Produces a lift curve and saves it locally.

    :param y_test: y_test series
    :param probability_predictions: dataframe of probability predictions, with the first column being the negative
    class predictions and the second column being the positive class predictions
    :param model_uid: model uid
    """
    skplt.metrics.plot_lift_curve(y_test, probability_predictions)
    plt.savefig(os.path.join('lift_curve.png'))
    plt.clf()


def plot_calibration_curve(y_test, predictions, n_bins, bin_strategy):
    """
    Produces a calibration plot and saves it locally. The raw data behind the plot is also written locally.

    :param y_test: y_test series
    :param predictions: predictions series
    :param n_bins: number of bins for the predictions
    :param bin_strategy: uniform - all bins have the same width; quantile - bins have the same number of observations
    """
    try:
        prob_true, prob_pred = calibration_curve(y_test, predictions, n_bins=n_bins, strategy=bin_strategy)
        fig, ax = plt.subplots()
        plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='model')
        line = mlines.Line2D([0, 1], [0, 1], color='black')
        transform = ax.transAxes
        line.set_transform(transform)
        ax.add_line(line)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        fig.suptitle(f' {bin_strategy.title()} Calibration Plot {n_bins} Requested Bins')
        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('True Probability in Each Bin')
        plt.legend()
        plt.savefig(os.path.join(f'{bin_strategy}_{n_bins}_calibration_plot.png'))
        plt.clf()
        calibration_df = pd.DataFrame({'prob_true': prob_true, 'prob_pred': prob_pred})
        calibration_df.to_csv(os.path.join(f'{bin_strategy}_{n_bins}_calibration_summary.csv'), index=False)
    except Exception as e:
        print(e)


def main():
    x, y = load_breast_cancer(return_X_y=True, as_frame=True)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    for model in MODEL_TRAINING_LIST:
        pipeline = Pipeline(steps=[('model', model.model)])
        best_model = train_model_with_hyperopt(
            estimator=pipeline,
            x_train=x_train,
            y_train=y_train,
            model_uid=model.model_name,
            param_space=model.param_space,
            iterations=model.iterations,
            cv_strategy=5,
            cv_scoring='neg_log_loss'
        )

        predictions_df = pd.concat(
            [
                pd.DataFrame(best_model.predict_proba(x_test), columns=['0_prob', '1_prob']),
                y_test.reset_index(drop=True)
            ], axis=1)

        calculate_probability_lift(y_test=y_test, probability_predictions=predictions_df['1_prob'])
        plot_calibration_curve(y_test=y_test, predictions=predictions_df['1_prob'], n_bins=10, bin_strategy='uniform')
        plot_calibration_curve(y_test=y_test, predictions=predictions_df['1_prob'], n_bins=10, bin_strategy='quantile')
        plot_cumulative_gains_chart(y_test=y_test, probability_predictions=predictions_df[['0_prob', '1_prob']])
        plot_lift_curve_chart(y_test=y_test, probability_predictions=predictions_df[['0_prob', '1_prob']])


if __name__ == "__main__":
    main()

    # https://endtoenddatascience.com/chapter11-machine-learning-calibration

    # lift and gain charts measure cardinality, not calibration
    # https://howtolearnmachinelearning.com/articles/the-lift-curve-in-machine-learning/
    # https://www2.cs.uregina.ca/~dbd/cs831/notes/lift_chart/lift_chart.html
