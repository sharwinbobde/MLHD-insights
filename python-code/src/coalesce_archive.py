from pyspark.sql import SparkSession


def read_ground_truth(year: int, set_num: int):
    filename = data_stem + \
               f"holdout/interactions/" \
               f"year_{year}-test_{RS_or_EA}_test-interactions-user_item-set_{set_num}.orc"
    outname = data_stem + \
              f"holdout/interactions/" \
              f"year_{year}-test_{RS_or_EA}_test-interactions-user_item-set_{set_num}.parquet"
    spark.read.orc(filename) \
        .coalesce(1) \
        .write \
        .parquet(outname, mode='overwrite')


def read_recommendations_df(year: int, model: str, set_num: int):
    filename = data_stem + f"output-{RS_or_EA}/{model}/year_{year}-{model}-set_{set_num}.orc"
    outname = data_stem + f"output-{RS_or_EA}/{model}/year_{year}-{model}-set_{set_num}.parquet"

    spark.read.orc(filename) \
        .coalesce(1) \
        .write \
        .parquet(outname, mode='overwrite')


def read_catalog() -> dict:
    filename = data_stem + "item_listens_per_year.orc"
    outname = data_stem + "item_listens_per_year.parquet"

    spark.read.orc(filename) \
        .coalesce(1) \
        .write \
        .parquet(outname, mode='overwrite')

    
if __name__ == '__main__':
    archive_size = 100
    # data_stem = f"../../scala-code/data/processed/archived-{archive_size}-parts/"
    data_stem = f"../../scala-code/data/processed/"
    spark = SparkSession \
        .builder \
        .getOrCreate()

    years = [2005, 2008, 2012]
    sets = range(1, 4)
    models = ["CF-user_rec", "CF-user_artist", "Tailored"]

    read_catalog()
    for RS_or_EA in ["RS", "EA"]:
        for yr in years:
            for set_ in sets:
                read_ground_truth(year=yr, set_num=set_)

                for model_ in models:
                    read_recommendations_df(year=yr, model=model_, set_num=set_)

