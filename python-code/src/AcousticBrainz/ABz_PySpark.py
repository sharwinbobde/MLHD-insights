import os
import re
import sys

from pyspark import StorageLevel
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, MinMaxScaler, RobustScaler
from pyspark.ml.linalg import DenseVector
from pyspark.mllib.linalg import Vectors, VectorUDT
from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, input_file_name, udf, lit
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
import pyspark.ml.feature
from pyspark.ml.feature import BucketedRandomProjectionLSH, BucketedRandomProjectionLSHModel
import re

import config
from src.AcousticBrainz.LSHBias import LSHBias

LSH_NUM_BITS = int(2 ** 13)


def flatten_df_structs(nested_df):
    stack = [((), nested_df)]
    columns = []

    while len(stack) > 0:
        parents, sub_df = stack.pop()

        flat_cols = [
            col(".".join(parents + (c[0],))).alias("_".join(parents + (c[0],)))
            for c in sub_df.dtypes
            if c[1][:6] != "struct"
        ]

        nested_cols = [
            c[0]
            for c in sub_df.dtypes
            if c[1][:6] == "struct"
        ]

        columns.extend(flat_cols)

        for nested_col in nested_cols:
            projected_df = sub_df.select(nested_col + ".*")
            stack.append((parents + (nested_col,), projected_df))

    return nested_df.select(columns)


def get_array_and_non_array_column_names(df):
    columns_to_flatten = []
    columns_not_to_flatten = []
    fields_by_datatype = df.schema.fields
    for struct_field in fields_by_datatype:
        if isinstance(struct_field.dataType, ArrayType):
            columns_to_flatten.append(struct_field.name)
        else:
            columns_not_to_flatten.append(struct_field.name)
    return columns_to_flatten, columns_not_to_flatten


def flatten_df_arrays(df):
    columns_to_flatten, columns_not_to_flatten = get_array_and_non_array_column_names(df)

    # columns_to_flatten = df.drop('rec_MBID').columns
    df_sizes = df.select(*[F.size(col_).alias(col_) for col_ in columns_to_flatten])
    df_max = df_sizes.agg(*[F.max(col_).alias(col_) for col_ in columns_to_flatten])
    max_dict = df_max.collect()[0].asDict()

    df_result = df.select(*[col_ for col_ in columns_not_to_flatten],
                          *[df[col_][i] for col_ in columns_to_flatten for i in range(max_dict[col_])])

    # split columns by array and non array to see more nested structures
    columns_to_flatten, columns_not_to_flatten = get_array_and_non_array_column_names(df_result)
    if len(columns_to_flatten) != 0:
        return flatten_df_arrays(df_result)
    else:
        return df_result


@udf(returnType=StringType())
def MBID_from_filename(s):
    return re.search(r'[^/][^/]*(?=\.json)', s)[0]


@udf(returnType=ArrayType(DoubleType()))
def one_hot_vector_to_array(v):
    v = DenseVector(v)
    new_array = list([float(x) for x in v])
    return new_array


@udf(returnType=ArrayType(IntegerType()))
def hash_feature_vector_to_array_of_ints(v, feature_name):
    # same logic as hashing for a single item LSH.hash_single()
    # hash_arr = LSH.hash_single(v)
    lsh = LSH_per_feature_set[feature_name]
    hash_arr = lsh.hash_single(v)

    # make 32 bit hashes and store as integers
    out = []
    for i in range(int(LSH_NUM_BITS / 32)):
        arr = hash_arr[32 * i: 32 * i + 32]
        int_val = int(''.join(map(str, arr)), base=2)
        out.append(int_val)
    return out


def hash_features_matching_regex(df: DataFrame, feature_name: str):
    # Vector assembler + Max-Min(0-1) scaling
    assembler = VectorAssembler(inputCols=feature_columns_dict[feature_name], outputCol="FeatureVector_unscaled")
    scaler = MinMaxScaler(inputCol="FeatureVector_unscaled", outputCol="FeatureVector")
    pipeline = Pipeline(stages=[assembler, scaler])

    # drop feature columns and the unscaled features.
    df = pipeline.fit(df).transform(df) \
        .drop("FeatureVector_unscaled")

    hash_column_name = f"{feature_name}_hash_{LSH_NUM_BITS}_bits"
    df = df \
        .withColumn(
        hash_column_name,
        hash_feature_vector_to_array_of_ints(col("FeatureVector"), lit(feature_name))) \
        .drop("FeatureVector") \
        .cache()
    return df, hash_column_name


if __name__ == '__main__':
    driver_memory = '3g'
    pyspark_submit_args = ' --driver-memory ' + driver_memory + ' pyspark-shell'
    os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

    spark = SparkSession \
        .builder \
        .appName("ABz Preprocessing + LSH") \
        .config("spark.executor.memory", "8G") \
        .config("spark.driver.memory", "3G") \
        .getOrCreate()

    sc = spark.sparkContext
    path = config.ABz_directory
    # read jsons and generate rec_MBID
    df_raw = spark.read.json(path, multiLine=True) \
        .withColumn("rec_MBID", MBID_from_filename(input_file_name()))

    # drop irrelevant:  metadata ... all of it
    df_cleaned = df_raw.drop("metadata")

    # flatten struct columns
    df_ = flatten_df_structs(df_cleaned)
    # drop noisy item: parents:['rhythm'];  key:beats_position
    df_ = df_.drop("rhythm_beats_position")
    # flatten array columns
    df_ = flatten_df_arrays(df_)

    feature_columns = df_.columns
    feature_columns.remove('rec_MBID')
    print(f"Num features = {len(feature_columns)}")

    # one-hot encoding for string columns columns
    # string item: parents:['tonal'];  key:chords_key
    # string item: parents:['tonal'];  key:chords_scale
    # string item: parents:['tonal'];  key:key_key
    # string item: parents:['tonal'];  key:key_scale
    # we keep dropLast as False because we will be using neighbourhood methods for recommendation.
    categorical_features_names = ['tonal_chords_key', 'tonal_chords_scale', 'tonal_key_key', 'tonal_key_scale']
    indexed_names = list(map(lambda x: x + '_indexed', categorical_features_names))
    encoded_names = list(map(lambda x: x + '_encoded', indexed_names))
    indexer = StringIndexer(inputCols=categorical_features_names,
                            outputCols=indexed_names)
    df_ = indexer.fit(df_).transform(df_) \
        .drop(*categorical_features_names)

    encoder = OneHotEncoder(inputCols=indexed_names,
                            outputCols=encoded_names,
                            dropLast=False)
    df_ = encoder.fit(df_).transform(df_)
    for name in encoded_names:
        df_ = df_.withColumn(name + "_arr", one_hot_vector_to_array(col(name)))

    # cleanup
    for name in indexed_names:
        df_ = df_.drop(name)
    for name in encoded_names:
        df_ = df_.drop(name)

    # flatten array columns
    df_ = flatten_df_arrays(df_) \
        .persist(StorageLevel.DISK_ONLY)

    feature_columns = df_.columns
    feature_columns.remove('rec_MBID')
    print(f"Num features after one-hot encoding= {len(feature_columns)}")

    LSH_per_feature_set = {}
    feature_columns_dict = {}
    refined_cols = ["rec_MBID"]
    params = [(r".*", "all_features"),
              (r"tonal.*", "tonal"),
              (r"rhythm.*", "rhythm"),
              (r"lowlevel.*", "lowlevel"),
              ]
    for p in params:
        r = re.compile(p[0])
        feature_columns = df_.columns
        feature_columns.remove('rec_MBID')
        feature_columns = list(filter(r.match, feature_columns))

        LSH = LSHBias(feature_dim=len(feature_columns), bits=LSH_NUM_BITS)
        LSH_per_feature_set[p[1]] = LSH
        feature_columns_dict[p[1]] = feature_columns

    for p in params:
        df_, colname = hash_features_matching_regex(df_, feature_name=p[1])
        refined_cols.append(colname)

    df_ = df_.select(*refined_cols)

    df_.printSchema()
    df_ \
        .write \
        .mode(saveMode='overwrite') \
        .option("header", True) \
        .orc("./scala-code/data/processed/ABzFeatures.orc")

    sc.stop()
    print("Done")