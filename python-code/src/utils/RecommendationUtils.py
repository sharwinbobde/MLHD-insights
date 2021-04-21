from math import isclose

import pandas as pd
import numpy as np
from pyspark.sql import SparkSession


def df_to_dict(df: pd.DataFrame) -> dict:
    return df \
        .groupby("user_id")['rec_id'] \
        .apply(list) \
        .to_dict()


class RecommendationUtils:

    def __init__(self, data_stem: str, test_set_type: str):
        if test_set_type not in ["RS", "EA"]:
            raise ValueError("value for test_set_type fhould be either \"RS\" or \"EA\"")

        self.test_set_type = test_set_type
        self.data_stem = data_stem

    @staticmethod
    def get_recommendations_dict_from_single_df(df: pd.DataFrame, reranking_weight: float = 1.0):
        df_ = df.copy()
        df_["rank"] = df_["rank"].apply(lambda x: x * reranking_weight)
        return df_to_dict(df_)

    @staticmethod
    def get_recommendations_dict_from_many_df(models: list[str], model_recs_df: dict, set_num: int,
                                              reranking_weights: list[float], K: int):
        if len(models) != len(reranking_weights):
            raise ValueError("models and reranking_weights must be lists with the same size")

        if not isclose(np.sum(reranking_weights), 1.0, rel_tol=0.01):
            raise ValueError(
                "the list reranking_weights should sum up to 1 (with acceptable error of 1%)\nprovided: " + str(
                    reranking_weights))

        list_df = []
        for i in range(len(models)):
            df = model_recs_df[(models[i], set_num)]
            # rerank
            df["rank"] = df["rank"].apply(lambda x: x * reranking_weights[i])
            list_df.append(df)

        # sort by rank
        df = pd.concat(list_df) \
            .sort_values(by=["user_id", "rank"]) \
            .drop_duplicates(subset=["user_id", "rec_id"])

        # code snippet for showing duplicate user-item entries, (a bug in the chosen recommender)
        # dups = df[["user_id", 'rec_id']].pivot_table(index=["user_id", 'rec_id'], aggfunc='size')
        # print(dups[dups>1])

        # find new rank
        df["new_rank"] = df.groupby("user_id")['rank'].rank(method='first')
        df = df.query("new_rank <= " + str(K))
        return df_to_dict(df)

    def read_recommendations_df(self, year: int, model: str, set_num: int, k: int = -1) -> pd.DataFrame:
        spark = SparkSession \
            .builder \
            .getOrCreate()
        filename = self.data_stem + f"output-{self.test_set_type}/{model}/year_{year}-{model}-set_{set_num}.orc"
        if k > 1:
            out = spark.read.orc(filename) \
                .filter(f"rank <= {k}") \
                .toPandas()
        else:
            out = spark.read.orc(filename) \
                .toPandas()
        spark.stop()
        return out

    def read_ground_truth(self, year: int, set_num: int) -> dict:
        spark = SparkSession \
            .builder \
            .getOrCreate()
        filename = self.data_stem + \
                   f"holdout/interactions/" \
                   f"year_{year}-test_{self.test_set_type}_test-interactions-user_item-set_{set_num}.orc"
        df = spark.read.orc(filename).toPandas()

        spark.stop()
        return df_to_dict(df)

    def read_catalog(self, year: int) -> dict:
        spark = SparkSession \
            .builder \
            .getOrCreate()
        filename = self.data_stem + "item_listens_per_year.orc"
        df = spark.read.orc(filename) \
            .withColumnRenamed(f"sum_{year}", "count") \
            .filter("count > 0") \
            .toPandas()

        df = df.set_index('rec_id') \
            ["count"]

        spark.stop()
        return df.to_dict()

    @staticmethod
    def get_novelty_threshold(catalog: dict) -> (list[int], list[int], int):
        df = pd.DataFrame()
        df["rec_id"] = list(catalog.keys())
        df["count"] = df.rec_id.map(lambda x: catalog[x])
        novelty_threshold = df['count'].quantile(0.8)

        return novelty_threshold
