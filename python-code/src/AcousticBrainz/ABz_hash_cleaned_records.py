import os
import re
import sys

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.linalg import DenseVector
from pyspark.sql.functions import col, input_file_name, udf, lit
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
import re

import config
from src.AcousticBrainz.LSHBias import LSHBias

LSH_NUM_BITS = int(2 ** 13)


def assemble(df, feature_name):
    #     Vector assembler + Max-Min(0-1) scaling
    assembler = VectorAssembler(inputCols=feature_columns_dict[feature_name],
                                outputCol="FeatureVector_unscaled_" + feature_name)
    df = assembler.transform(df)
    return df, "FeatureVector_unscaled_" + feature_name


def scale(df, feature_name):
    scaler = MinMaxScaler(inputCol="FeatureVector_unscaled_" + feature_name,
                          outputCol="FeatureVector_" + feature_name)
    df = scaler.fit(df).transform(df)
    return df


def hash_features_matching_regex(df, feature_name):
    hash_column_name = str(feature_name) + "_hash_" + str(LSH_NUM_BITS) + "_bits"
    df = df \
        .withColumn(
            hash_column_name,
            hash_feature_vector_to_array_of_ints(col("FeatureVector_" + feature_name), lit(feature_name))) \
        .drop("FeatureVector_" + feature_name)
    return df, hash_column_name


@udf(returnType=ArrayType(IntegerType()))
def hash_feature_vector_to_array_of_ints(v, feature_name):
    # same logic as hashing for a single item LSH.hash_single()
    hash_arr = LSH_per_feature_set[feature_name].hash_single(v)

    # make 32 bit hashes and store as integers
    out = []
    for i in range(int(LSH_NUM_BITS / 32)):
        arr = hash_arr[32 * i: 32 * i + 32]
        int_val = int(''.join(map(str, arr)), base=2)
        out.append(int_val)
    return out


if __name__ == '__main__':
    # driver_memory = '3g'
    # pyspark_submit_args = ' --driver-memory ' + driver_memory + ' pyspark-shell'
    # os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

    spark = SparkSession \
        .builder \
        .appName("ABz Preprocessing + LSH") \
        .config("spark.executor.memory", "5G") \
        .config("spark.driver.memory", "10G") \
        .getOrCreate()

    sc = spark.sparkContext

    df_cleaned = spark.read.orc(config.ABz_cleaned_orc).limit(10)
    df_cleaned.printSchema()

    LSH_per_feature_set = {}
    feature_columns_dict = {}
    params = [
        (r"rhythm.*", "rhythm"),
        #           (r"tonal.*", "tonal"),
        #           (r"lowlevel.*", "lowlevel"),
        #           (r".*", "all_features"),
    ]
    for p in params:
        r = re.compile(p[0])
        feature_columns = df_cleaned.columns
        feature_columns.remove('rec_MBID')
        feature_columns = list(filter(r.match, feature_columns))

        LSH = LSHBias(feature_dim=len(feature_columns), bits=LSH_NUM_BITS)
        LSH_per_feature_set[p[1]] = LSH
        feature_columns_dict[p[1]] = feature_columns
    params

    df_preprocessed = df_cleaned

    refined_cols = ["rec_MBID"]

    for p in params:
        df_preprocessed, c = assemble(df_preprocessed, p[1])
        refined_cols.append(c)

    df_preprocessed = df_preprocessed.select(*refined_cols)

    for p in params:
        df_preprocessed = scale(df_preprocessed, p[1])

    cols = ['rec_MBID'] + ['FeatureVector_' + p[1] for p in params]
    df_trimmed = df_preprocessed.select(*cols).persist(StorageLevel.DISK_ONLY)

    refined_cols = ["rec_MBID"]
    df_with_hashes = df_trimmed

    for p in params:
        _, c = hash_features_matching_regex(df_with_hashes, feature_name=p[1])

        _.select("rec_MBID", c) \
            .write \
            .mode(saveMode='overwrite') \
            .orc(config.out_dir + "2M-hashes.orc")
