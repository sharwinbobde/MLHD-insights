from glob import glob
import pandas as pd


def df_to_dict(df: pd.DataFrame) -> dict:
    return df \
        .groupby("user_id")['rec_id'] \
        .apply(list) \
        .to_dict()


class FileUtils:

    def __init__(self, data_stem: str):
        self.data_stem = data_stem

    def get_recommendations_dict_single_model(self, year: int, model: str, set_num: int, k: int,
                                              reranking_weight: float = 1.0) -> dict:

        df = self.get_recommendations_df(year, model, set_num, k, reranking_weight)
        return df_to_dict(df)

    def get_recommendations_dict_many_model(self, year: int, models: list[str], set_num: int, k: int, K: int,
                                            reranking_weights: list[float]) -> dict:
        if len(models) != len(reranking_weights):
            raise ValueError("models and reranking_weights must be lists with the same size")
        list_df = []
        for i in range(len(models)):
            df = self.get_recommendations_df(year, models[i], set_num, k, reranking_weights[i])
            list_df.append(df)
        df = pd.concat(list_df) \
            .sort_values(by=["user_id", "rank"])

        # code snippet for showing duplicate user-item entries, (a bug in the chosen recommender)
        # dups = df[["user_id", 'rec_id']].pivot_table(index=["user_id", 'rec_id'], aggfunc='size')
        # print(dups[dups>1])

        # find new rank
        df["new_rank"] = df.groupby("user_id")['rank'].rank(method='first')
        df = df.query("new_rank <= " + str(K))
        print(df)
        return df_to_dict(df)

    def get_recommendations_df(self, year: int, model: str, set_num: int, k: int,
                               reranking_weight: float = 1.0) -> pd.DataFrame:
        for f in glob(
                self.data_stem + "output/year_" + str(year) + "_" + model + "_set_" + str(set_num) + ".csv/part-*.csv"):
            # code snippet for showing duplicate user-item entries, (a bug in the chosen recommender)
            # dups = pd.read_csv(f)[["user_id", 'rec_id']].pivot_table(index=["user_id", 'rec_id'], aggfunc='size')
            # print(dups[dups>1])

            df = pd.read_csv(f) \
                .query("rank <= " + str(k))
            df['rank'] = df['rank'].apply(lambda x: x * reranking_weight)
            return df

    def read_ground_truth(self, year: int, set_num: int) -> dict:
        for f in glob(self.data_stem + "holdout/year_" + str(year) + "_test_test_interactions_user-item_set_" + str(
                set_num) + ".csv/part-*.csv"):
            df = pd.read_csv(f)
            return df_to_dict(df)

    def read_catalog(self, year: int) -> dict:
        for f in glob(self.data_stem + "item_listens_per_year.csv/part-*.csv"):
            df = pd.read_csv(f) \
                .rename(columns={"sum(yr_" + str(year) + ")": "count"}) \
                .query("count > 0") \
                .set_index('rec_id') \
                ["count"]

            return df.to_dict()
