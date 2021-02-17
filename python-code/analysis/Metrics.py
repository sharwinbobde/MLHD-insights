from glob import glob

import pandas as pd
import recmetrics

experiment_years = [2005, 2008, 2012]


class Metrics:
    data_stem = "../../scala-code/data/processed/"

    def __init__(self):
        self.models = ["CF"]
        self.years = experiment_years
        print('initializing Metrics utilities...')

        self.catalogues = dict()
        self.truths = dict()
        for yr in experiment_years:
            print("initializing " + str(yr))
            self.catalogues[yr] = self.create_catelog(yr)

            self.truths[yr] = dict()
            for set_no in [1, 2, 3]:
                self.truths[yr][set_no] = self.read_ground_truth(yr, set_no)

        print('Ready!')

    def read_recommendations_file(self, year: int, model: str, set_num: int, k: int) -> dict:
        for f in glob(
                self.data_stem + "output/year_" + str(year) + "_" + model + "_set_" + str(set_num) + ".csv/part-*.csv"):
            return pd.read_csv(f) \
                .query("rank <= " + str(k)) \
                .groupby("user_id")['rec_id'] \
                .apply(list) \
                .to_dict()

    def read_ground_truth(self, year: int, set_num: int) -> dict:
        for f in glob(self.data_stem + "holdout/year_" + str(year) + "_test_test_interactions_user-item_set_" + str(
                set_num) + ".csv/part-*.csv"):
            return pd.read_csv(f) \
                .groupby("user_id")['rec_id'] \
                .apply(list) \
                .to_dict()

    def create_catelog(self, year: int) -> dict:
        for f in glob(self.data_stem + "item_listens_per_year.csv/part-*.csv"):
            df = pd.read_csv(f) \
                .rename(columns={"sum(yr_" + str(year) + ")": "count"}) \
                .query("count > 0") \
                .set_index('rec_id') \
                ["count"]

            return df.to_dict()

    # ========================== Metrics ======================

    def NaN_Proportion(self, recs: dict, year: int, set_num: int):
        truth = self.truths[year][set_num]
        df_truth = pd.DataFrame(truth.items(), columns=["user_id", 'truth'])
        df_recs = pd.DataFrame(recs.items(), columns=["user_id", 'recommended'])

        df = df_truth.join(df_recs.set_index('user_id'), on="user_id")
        df['truth_len'] = df['truth'].str.len()
        df['recommended_len'] = df['recommended'].str.len()
        len_nan = df.recommended_len.isna().sum()
        len_tot = df.recommended_len.size
        print("NaN count: " + str(len_nan))
        print("total records: " + str(len_tot))
        return len_nan / len_tot

    def mark(self, recs: dict, year: int, set_num: int, k: int):
        truth = self.truths[year][set_num]
        df_truth = pd.DataFrame(truth.items(), columns=["user_id", 'truth'])
        df_recs = pd.DataFrame(recs.items(), columns=["user_id", 'recommended'])

        df = df_truth.join(df_recs.set_index('user_id'), on="user_id") \
            [["truth", "recommended"]]
        df.loc[df['recommended'].isnull(), ['recommended']] = df.loc[df['recommended'].isnull(), 'recommended'] \
            .apply(lambda x: [])
        return recmetrics.mark(df.truth.tolist(), df.recommended.tolist(), k)

    def mark_filter_valid(self, recs: dict, year: int, set_num: int, k: int):
        truth = self.truths[year][set_num]
        df_truth = pd.DataFrame(truth.items(), columns=["user_id", 'truth'])
        df_recs = pd.DataFrame(recs.items(), columns=["user_id", 'recommended'])

        df = df_truth.join(df_recs.set_index('user_id'), on="user_id") \
            [["truth", "recommended"]] \
            .query("recommended.notnull() ")
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
        pop = self.catalogues[year]
        metric, _ = recmetrics.novelty(arr, pop, len(arr), k)
        print('------ Novelty')
        print(metric)

    def coverage(self, recs: dict, year: int):
        """
        Coverage in percentage
        :param recs: recommendations dictionary
        :param year: year of experiment
        :return:
        """
        arr = list(recs.values())
        catalogue = list(self.catalogues[year].keys())
        return recmetrics.prediction_coverage(arr, catalogue)

    def familiarity(self):
        # TODO
        pass

    def get_all_metrics(self, recs: dict, year: int, set_num: int, k: int):
        m = {}
        m['MAR@' + str(K)] = self.mark(recs, year, set_, k)
        m['MAR_filtered@' + str(K)] = self.mark_filter_valid(recs, year, set_, k)
        m['Pers@' + str(K)] = self.personalization(recs)
        m['Nov@' + str(K)] = self.personalization(recs)
        m['NaN_Prop@' + str(K)] = self.NaN_Proportion(recs, year, set_)
        m['Cov@' + str(K)] = self.coverage(recs, year)
        # m['Fam@' + str(K)] = metrics.familiarity(recs)
        return m


if __name__ == '__main__':
    metrics = Metrics()
    yr = 2012
    K = 100
    set_ = 3
    model = "CF"

    for yr in experiment_years:
        print("year: " + str(yr) + " ==================================")
        recs = metrics.read_recommendations_file(yr, model, set_, K)
        m = metrics.get_all_metrics(recs, yr, set_, K)
        print(m)
    print("DONE")
