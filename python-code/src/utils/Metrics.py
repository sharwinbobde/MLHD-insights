import logging

import pandas as pd
import recmetrics

from src.utils.FileUtils import FileUtils

experiment_years = [2005, 2008, 2012]
data_stem = "../../scala-code/data/processed/"
fu = FileUtils(data_stem)


class Metrics:

    def __init__(self, year: int):
        self.models = ["CF"]
        self.year = year
        print('initializing Metrics utilities for year ' + str(year)+ '...')

        self.catalogues = dict()
        self.truths = dict()
        self.catalog = fu.read_catalog(year)

        self.truths = dict()
        for set_no in [1, 2, 3]:
            self.truths[set_no] = fu.read_ground_truth(year, set_no)

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
        metric = recmetrics.personalization(arr)
        return metric

    def novelty(self, recs: dict, year: int, k: int):
        """
        assumes numeric item codes
        """
        arr = list(recs.values())
        pop = self.catalog
        # TODO error because for user -> artist -> rec we need to filter buy songs that exist in the year
        metric, _ = recmetrics.novelty(arr, pop, len(arr), k)
        return metric

    def coverage(self, recs: dict, year: int):
        """
        Coverage in percentage
        :param recs: recommendations dictionary
        :param year: year of experiment
        :return:
        """
        arr = list(recs.values())
        catalog = list(self.catalog.keys())
        return recmetrics.prediction_coverage(arr, catalog)

    def familiarity(self, recs):
        # TODO
        return -1

    def get_all_metrics(self, recs: dict, year: int, set_num: int, k: int):
        m = {}
        m['MAR@' + str(k)] = self.mark(recs, year, set_num, k)
        m['MAR_filtered@' + str(k)] = self.mark_filter_valid(recs, year, set_num, k)
        m['NaN_Prop@' + str(k)], _, _ = self.NaN_Proportion(recs, year, set_num)
        m['Nov@' + str(k)] = self.novelty(recs, year, k)
        try:
            m['Pers@' + str(k)] = self.personalization(recs)
        except TypeError:
            logging.error("TypeError")
        arr = []
        for items in list(recs.items()):
            arr.append(len(items[1]))
        print(set(arr))
        m['Cov@' + str(k)] = self.coverage(recs, year)
        m['Fam@' + str(k)] = self.familiarity(recs)
        return m


if __name__ == '__main__':
    metrics_evaluators = {}
    for yr in experiment_years:
        metrics_evaluators[yr] = Metrics(yr)
    # TODO filter users to have some specified number of items
    k = 93
    set_ = 1
    for model in ["CF_user-rec", "CF_user-artist"]:
        print("\n\nmodel = " + model)
        for yr in experiment_years:
            print("year: " + str(yr) + " ==================================")
            recs = fu.get_recommendations_dict_single_model(yr, model, set_, k)
            m = metrics_evaluators[yr].get_all_metrics(recs, yr, set_, k)
            print(m)

    print("DONE")
