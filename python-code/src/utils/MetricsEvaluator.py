import logging
import sys

import pandas as pd
import recmetrics
import numpy as np
from config import experiment_years
from src.utils.RecommendationUtils import RecommendationUtils
from pyspark.sql import SparkSession

data_stem = "../../scala-code/data/processed/"

valid_models = ["CF-user_rec", "CF-user_artist", "Pop"]


class MetricsEvaluator:

    def __init__(self, models, year: int, k: int, archive_size: int = None, RS_or_EA: str = "RS"):
        for model in models:
            if model not in valid_models:
                raise ValueError(f"value for test_set_type should be in {valid_models}")

        self.models = models
        self.year = year
        print(f'Initializing MetricsEvaluator for year:{year} and k:{k}')
        if archive_size is None:
            self.rec_utils = RecommendationUtils(data_stem, RS_or_EA=RS_or_EA)
        else:
            self.rec_utils = RecommendationUtils(f"{data_stem}archived-{archive_size}-parts/", RS_or_EA=RS_or_EA)

        self.k = k
        self.catalogues = dict()
        self.truths = dict()
        self.catalog = self.rec_utils.read_catalog(year)

        self.truths = dict()
        self.recommendation_dfs = dict()
        for set_no in [1, 2, 3]:
            self.truths[set_no] = self.rec_utils.read_ground_truth(year, set_no)
            for model in self.models:
                self.recommendation_dfs[(model, set_no)] = self.rec_utils \
                    .read_recommendations_df(year=year,
                                             model=model,
                                             set_num=set_no,
                                             k=k)
                # print((model, set_no))
                # print(self.recommendation_dfs[(model, set_no)].shape)

        self.novelty_threshold = self.rec_utils.get_novelty_threshold(self.catalog)
        print(f'Completed initialization')

    # ========================== Metrics ======================

    def NaN_Proportion(self, recs: dict, set_num: int):
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

    def mark(self, recs: dict, set_num: int, k: int):
        truth = self.truths[set_num]
        df_truth = pd.DataFrame(truth.items(), columns=["user_id", 'truth'])
        df_recs = pd.DataFrame(recs.items(), columns=["user_id", 'recommended'])

        df = df_truth.join(df_recs.set_index('user_id'), on="user_id") \
            [["truth", "recommended"]]
        df.loc[df['recommended'].isnull(), ['recommended']] = df.loc[df['recommended'].isnull(), 'recommended'] \
            .apply(lambda x: [])
        return recmetrics.mark(df.truth.tolist(), df.recommended.tolist(), k)

    def mark_filter_valid(self, recs: dict, set_num: int, k: int):
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

    def novelty(self, recs: dict, K: int):
        """
        assumes numeric item codes
        """
        predicted = list(recs.values())
        metrics = []
        for sublist in predicted:
            unpop_count = len([i for i in sublist if self.catalog[i] < self.novelty_threshold])
            metrics.append(unpop_count / K)
        return np.mean(metrics)

    def coverage(self, recs: dict):
        """
        Coverage in percentage
        :param recs: recommendations dictionary
        :return:
        """
        arr = list(recs.values())
        catalog = list(self.catalog.keys())
        return recmetrics.prediction_coverage(arr, catalog) / 100.0

    def modified_diversity(self, recs: dict):
        arr = list(recs.values())
        df = pd.DataFrame(np.array(arr))

    def familiarity(self, recs):
        # TODO
        return -1

    def get_all_metrics(self, recs: dict, set_num: int, K: int) -> dict:
        m = dict()
        m['MAR'] = self.mark(recs, set_num, K)
        m['MAR_filtered'] = self.mark_filter_valid(recs, set_num, K)
        m['NaN_Prop'], _, _ = self.NaN_Proportion(recs, set_num)
        m['Pers'] = self.personalization(recs)
        m['Nov'] = self.novelty(recs, K)
        m['Cov'] = self.coverage(recs)
        m['Fam'] = self.familiarity(recs)
        return m

    def get_metrics(self, metrics: list[str], recs: dict, set_num: int, K: int) -> dict:
        m = dict()
        for met in metrics:
            if met == 'MAR':
                m['MAR'] = self.mark(recs, set_num, K)

            if met == 'MAR_filtered':
                m['MAR_filtered'] = self.mark_filter_valid(recs, set_num, K)

            if met == 'NaN_Prop':
                m['NaN_Prop'], _, _ = self.NaN_Proportion(recs, set_num)

            if met == 'Pers':
                m['Pers'] = self.personalization(recs)

            if met == 'Nov':
                m['Nov'] = self.novelty(recs, K)

            if met == 'Cov':
                m['Cov'] = self.coverage(recs)

            if met == 'Fam':
                m['Fam'] = self.familiarity(recs)
        return m


if __name__ == '__main__':
    k_ = 10
    set_ = 1
    metrics_evaluators = {}
    models_ = ["CF-user_rec", "CF-user_artist"]
    archive_parts_test = 100
    for yr in experiment_years:
        metrics_evaluators[yr] = MetricsEvaluator(models=models_, year=yr, k=k_, archive_size=archive_parts_test)
    # TODO filter users to have some specified number of items

    print("Test for single recommenders")
    for model_ in models_:
        print("\nmodel = " + model_)
        for yr in experiment_years:
            print(f"year: {yr}")
            recs_df = metrics_evaluators[yr].recommendation_dfs[(model_, set_)]
            recs_ = RecommendationUtils.get_recommendations_dict_from_single_df(df=recs_df)
            m_ = metrics_evaluators[yr].get_all_metrics(recs=recs_, set_num=set_, K=k_)
            print(m_)

    print("\nTest for multiple recommenders")
    models = ["CF-user_rec", "CF-user_artist"]
    weights = [0.4, 0.6 ]
    K_ = k_
    for yr in experiment_years:
        print(f"year: {yr}")
        recs_ = RecommendationUtils \
            .get_recommendations_dict_from_many_df(models=models_,
                                                   model_recs_df=metrics_evaluators[yr].recommendation_dfs,
                                                   set_num=set_,
                                                   reranking_weights=weights,
                                                   K=k_)
        m_ = metrics_evaluators[yr].get_all_metrics(recs=recs_, set_num=set_, K=K_)
        print(m_)
    print("DONE")
