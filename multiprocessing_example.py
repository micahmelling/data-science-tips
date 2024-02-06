import itertools
import multiprocessing as mp
import random

import numpy as np
import pandas as pd


def _assemble_negative_and_positive_pairs(y_test, probability_predictions, subset_percentage=0.1):
    """
    Finds the combination of every predicted probability in the negative class and every predicted probability in the
    positive class.

    :param y_test: y_test series
    :param probability_predictions: positive probability predictions series
    :param subset_percentage: percentage of observations to keep, as finding all the combinations of positive and
    negative can result in a combinatorial explosion; default is 0.1
    :returns: list
    """
    df = pd.concat([y_test, probability_predictions], axis=1)
    df = df.sample(frac=subset_percentage)
    columns = list(df)
    true_label = columns[0]
    predicted_prob = columns[1]
    neg_df = df.loc[df[true_label] == 0]
    neg_probs = neg_df[predicted_prob].tolist()
    pos_df = df.loc[df[true_label] == 1]
    pos_probs = pos_df[predicted_prob].tolist()
    return list(itertools.product(neg_probs, pos_probs))


def _find_discordants(pairs):
    """
    Finds the number of discordants, defined as the number of cases where predicted probability in the negative
    class observation is greater than the predicted probability of the positive class observation.

    :param pairs: tuple where the first element is the negative probability and the second element is the positive
    probability
    :returns: integer
    """
    discordants = 0
    if pairs[0] >= pairs[1]:
        discordants += 1
    return discordants


def find_concordant_discordant_ratio_and_somers_d(y_test, probability_predictions, model_uid):
    """
    Finds the concordant-discordant ratiio and Somer's D and saved them locally

    :param y_test: y_test series
    :param probability_predictions: positive probability predictions series
    :param model_uid: model uid
    """
    pairs = _assemble_negative_and_positive_pairs(y_test, probability_predictions)
    with mp.Pool(processes=mp.cpu_count()) as pool:
        result = pool.map(_find_discordants, pairs)
    pairs = len(result)
    discordant_pairs = sum(result)
    concordant_discordant_ratio = 1 - (discordant_pairs / pairs)
    concordant_pairs = pairs - discordant_pairs
    somers_d = (concordant_pairs - discordant_pairs) / pairs
    pd.DataFrame({'concordant_discordant_ratio': [concordant_discordant_ratio], 'somers_d': [somers_d]}).to_csv(
        f'{model_uid}_concordant_discordant.csv', index=False)


if __name__ == "__main__":
    df = pd.DataFrame({
        'target': np.random.randint(0, 2, size=5_000),
        'pred': [random.random() for _ in range(5_000)]
    })
    find_concordant_discordant_ratio_and_somers_d(df['target'], df['pred'], model_uid='test')
