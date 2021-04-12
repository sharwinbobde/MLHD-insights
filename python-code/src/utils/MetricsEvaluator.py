import logging
import sys

import pandas as pd
import recmetrics
import numpy as np
from config import experiment_years
from src.utils.RecommendationUtils import RecommendationUtils
import matplotlib.pyplot as plt

data_stem = "../../scala-code/data/processed/"
rec_utils = RecommendationUtils(data_stem, "RS")


class MetricsEvaluator:

    def __init__(self, year: int):
        self.models = ["CF"]
        self.year = year
        print('initializing Metrics utilities for year ' + str(year) + '...')

        self.catalogues = dict()
        self.truths = dict()
        self.catalog = rec_utils.read_catalog(year)

        self.truths = dict()
        for set_no in [1, 2, 3]:
            self.truths[set_no] = rec_utils.read_ground_truth(year, set_no)

        self.novelty_threshold = rec_utils.get_novelty_threshold(self.catalog)
        print('Ready!')

    # ========================== Metrics ======================

    def NaN_Proportion(self, recs: dict, year: int, set_num: int):
        truth = self.truths[set_num]
        df_truth = pd.DataFrame(truth.items(), columns=["user_id", 'truth'])
        df_recs = pd.DataFrame(recs.items(), columns=["user_id", 'recommended'])

        df = df_truth.join(df_recs.set_index('user_id'), on="user_id")
        df['truth_len'] = df['truth'].str.len()
        df['recommended_len'] = df['recommended'].str.len()
        len_nan = df.recommended_len.isna().sum()
        len_tot = df.recommended_len.size
        # print("NaN count: " + str(len_nan))
        # print("total records: " + str(len_tot))
        return len_nan / len_tot, len_nan, len_tot

    def mark(self, recs: dict, year: int, set_num: int, k: int):
        truth = self.truths[set_num]
        df_truth = pd.DataFrame(truth.items(), columns=["user_id", 'truth'])
        df_recs = pd.DataFrame(recs.items(), columns=["user_id", 'recommended'])

        df = df_truth.join(df_recs.set_index('user_id'), on="user_id") \
            [["truth", "recommended"]]
        df.loc[df['recommended'].isnull(), ['recommended']] = df.loc[df['recommended'].isnull(), 'recommended'] \
            .apply(lambda x: [])
        return recmetrics.mark(df.truth.tolist(), df.recommended.tolist(), k)

    def mark_filter_valid(self, recs: dict, year: int, set_num: int, k: int):
        truth = self.truths[set_num]
        df_truth = pd.DataFrame(truth.items(), columns=["user_id", 'truth'])
        df_recs = pd.DataFrame(recs.items(), columns=["user_id", 'recommended'])

        df = df_truth.join(df_recs.set_index('user_id'), on="user_id") \
            [["truth", "recommended"]] \
            .dropna()
        return recmetrics.mark(df.truth.tolist(), df.recommended.tolist(), k)

    def personalization(self, recs: dict):
        """
        Personalization measures recommendation similarity across users.
        A high score indicates good personalization (user's lists of recommendations are different).
        A low score indicates poor personalization (user's lists of recommendations are very similar).
        A model is "personalizing" well if the set of recommendations for each user is different.
        """
        arr = list(recs.values())
        try:
            metric = recmetrics.personalization(arr)
        except TypeError:
            logging.error("TypeError: All lists need to be of the same size")
            arr = []
            for items in list(recs.items()):
                arr.append(len(items[1]))
            logging.error(set(arr))
            sys.exit(0)
        return metric

    def novelty(self, recs: dict, k: int):
        """
        assumes numeric item codes
        """
        predicted = list(recs.values())
        metrics = []
        for sublist in predicted:
            unpop_count = len([i for i in sublist if self.catalog[i] < self.novelty_threshold])
            metrics.append(unpop_count/k)
        return np.mean(metrics)

    def coverage(self, recs: dict):
        """
        Coverage in percentage
        :param recs: recommendations dictionary
        :param year: year of experiment
        :return:
        """
        arr = list(recs.values())
        catalog = list(self.catalog.keys())
        return recmetrics.prediction_coverage(arr, catalog) / 100.0

    def familiarity(self, recs):
        # TODO
        return -1

    def get_all_metrics(self, recs: dict, year: int, set_num: int, k: int) -> dict:
        m = {}
        m['MAR'] = self.mark(recs, year, set_num, k)
        m['MAR_filtered'] = self.mark_filter_valid(recs, year, set_num, k)
        m['NaN_Prop'], _, _ = self.NaN_Proportion(recs, year, set_num)
        m['Pers'] = self.personalization(recs)
        m['Nov'] = self.novelty(recs, k)
        m['Cov'] = self.coverage(recs)
        m['Fam'] = self.familiarity(recs)
        return m


if __name__ == '__main__':
    metrics_evaluators = {}
    for yr in experiment_years:
        metrics_evaluators[yr] = MetricsEvaluator(yr)
    # TODO filter users to have some specified number of items
    k_ = 20
    set_ = 1
    print("Test for single recommenders")
    for model in ["CF-user_rec", "CF-user_artist"]:
        print("\nmodel = " + model)
        for yr in experiment_years:
            print(f"year: {yr}")
            recs_ = rec_utils.get_recommendations_dict_single_model(yr, model, set_, k_)
            m = metrics_evaluators[yr].get_all_metrics(recs_, yr, set_, k_)
            print(m)

    print("\nTest for multiple recommenders")
    models = ["CF-user_rec", "CF-user_artist"]
    weights = [0.05, 0.95]
    K_ = k_
    for yr in experiment_years:
        print(f"year: {yr}")
        recs_ = rec_utils.get_recommendations_dict_many_model(yr, models, set_, k_, K_, weights)
        m_ = metrics_evaluators[yr].get_all_metrics(recs_, yr, set_, K_)
        print(m_)
    print("DONE")
