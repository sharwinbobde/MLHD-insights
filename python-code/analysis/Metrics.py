from glob import glob

import pandas as pd
import recmetrics


class Metrics:
    data_stem = "../../scala-code/data/processed/"

    def __init__(self):
        self.models = ["CF"]
        self.years = range(2005, 2007)
        print('initializing Metrics utilities...')
        for f in glob(self.data_stem + "item_frequencies.csv/part-*.csv"):
            self.item_frequencies_all = pd.read_csv(f)

        self.catalogues = dict()
        self.item_frequencies_per_year = dict()
        self.truths = dict()
        for yr in [2005, 2007, 2012]:
            print("here " + str(yr))
            self.catalogues[yr] = self.create_catelog(yr)
            # item_frequencies depends on catalogues
            self.item_frequencies_per_year[yr] = self.read_item_frequencies(yr)

            self.truths[yr] = dict()
            for set_no in [1, 2, 3]:
                print("here")
                self.truths[yr][set_no] = self.read_ground_truth(yr, set_no)

        print('Ready!')

    def read_recommendations_file(self, year: int, model: str, set_num: int, k: int) -> dict:
        for f in glob(
                self.data_stem + "output/year_" + str(year) + "_" + model + "_set_" + str(set_num) + ".csv/part-*.csv"):
            return pd.read_csv(f) \
                .query("rank <= " + str(k)) \
                .groupby("user")['item'] \
                .apply(list) \
                .to_dict()

    def read_ground_truth(self, year: int, set_num: int) -> dict:
        for f in glob(self.data_stem + "holdout/year_" + str(year) + "_test_test_interactions_user-item_set_" + str(
                set_num) + ".csv/part-*.csv"):
            return pd.read_csv(f) \
                .groupby("user")['item'] \
                .apply(list) \
                .to_dict()

    def read_item_frequencies(self, year: int) -> dict:
        catalogue = self.catalogues[year]
        df = self.item_frequencies_all \
            .rename(columns={"sum(yr_" + str(year) + ")": "count"}) \
            [["item", "count"]]

        df = df[df['item'].isin(list(catalogue.keys()))] \
            .set_index('item') \
            ["count"]

        return df.to_dict()

    def create_catelog(self, year: int) -> dict:
        for f in glob(
                self.data_stem + "item_listens_per_year.csv/part-*.csv"):
            df = pd.read_csv(f) \
                .rename(columns={"sum(yr_" + str(year) + ")": "count"}) \
                .query("count > 0") \
                .set_index('item') \
                ["count"]

            return df.to_dict()

    # ========================== Metrics ======================

    def mark(self, recs: dict, year: int, set_num: int, k: int):
        truth = self.truths[year][set_num]
        df_truth = pd.DataFrame(truth.items(), columns=["user", 'truth'])
        df_recs = pd.DataFrame(recs.items(), columns=["user", 'recommended'])

        df = df_truth.join(df_recs.set_index('user'), on="user")
        df['truth_len'] = df['truth'].str.len()
        df['recommended_len'] = df['recommended'].str.len()
        # print(df)
        len_nan = df.recommended_len.isna().sum()
        len_tot = df.recommended_len.size
        print("NaN count: " + str(len_nan))
        print("total records: " + str(len_tot))
        print("% : " + str(len_nan / len_tot * 100))
        # recmetrics.mark()

    def personalization(self, recs: dict):
        """
        Personalization measures recommendation similarity across users.
        A high score indicates good personalization (user's lists of recommendations are different).
        A low score indicates poor personalization (user's lists of recommendations are very similar).
        A model is "personalizing" well if the set of recommendations for each user is different.
        """
        arr = list(recs.values())
        metric = recmetrics.personalization(arr)
        print('------ Personalization')
        print(metric)

    def novelty(self, recs: dict, year: int, k: int):
        """
        assumes numeric item codes
        """
        arr = list(recs.values())
        pop = self.read_item_frequencies(year)
        metric, _ = recmetrics.novelty(arr, pop, len(arr), k)
        print('------ Novelty')
        print(metric)

    def coverage(self):
        # prediction_coverage
        # TODO
        pass

    def familiarity(self):
        # TODO
        pass


if __name__ == '__main__':
    metrics = Metrics()
    yr = 2012
    K = 100
    set_ = 3
    model = "CF"

    for yr in [2005, 2007, 2012]:
        print("year: " + str(yr))
        recs_dict = metrics.read_recommendations_file(yr, model, set_, K)
        metrics.mark(recs_dict, yr, set_, K)

    print("DONE")
