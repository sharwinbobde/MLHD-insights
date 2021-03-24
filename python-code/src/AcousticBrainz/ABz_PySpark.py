import os
import re

from pyspark import StorageLevel
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, MinMaxScaler
from pyspark.ml.linalg import DenseVector
from pyspark.sql import Column
from pyspark.sql.functions import col, input_file_name, udf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as F
import pandas as pd
import numpy as np
import pyspark.ml.feature
from pyspark.ml.feature import BucketedRandomProjectionLSH, BucketedRandomProjectionLSHModel


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


if __name__ == '__main__':
    memory = '3g'
    pyspark_submit_args = ' --driver-memory ' + memory + ' pyspark-shell'
    os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args

    spark = SparkSession \
        .builder \
        .appName("ABz") \
        .config("spark.executor.memory", "8G") \
        .config("spark.driver.memory", "3G") \
        .getOrCreate()

    sc = spark.sparkContext
    # path = "/run/media/sharwinbobde/SharwinThesis/mlhd-ab-features/acousticbrainz-mlhd-0123/00/"
    path = "/run/media/sharwinbobde/SharwinThesis/ABz_features_subset/"
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
    df_ = flatten_df_arrays(df_)

    feature_columns = df_.columns
    feature_columns.remove('rec_MBID')
    print(f"Num features after one-hot encoding= {len(feature_columns)}")

    # Vector assembler + Max-Min normalization
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="FeatureVector")
    scaler = MinMaxScaler(inputCol="FeatureVector", outputCol="FeatureVector_scaled")
    pipeline = Pipeline(stages=[assembler, scaler])
    df_ = pipeline.fit(df_).transform(df_) \
        .drop(*feature_columns) \
        .drop("FeatureVector")
    df_.printSchema()

    # # experiment to select bucketLength and numHashTables
    # for num_tables in [1, 5, 10, 25, 50, 100]:
    #     for bucket_len in [1.0, 5.0, 10, 25.0, 50.0, 100.0]:
    #         brp = BucketedRandomProjectionLSH(
    #             inputCol="FeatureVector_scaled",
    #             outputCol=f"hash_{num_tables}bit_{bucket_len}len",
    #             seed=4242,
    #             bucketLength=bucket_len,
    #             numHashTables=num_tables)
    #         df_ = brp.fit(df_).transform(df_)
    # df_.show()

    # If input vectors are normalized, 1-10 times of pow(numRecords, -1/inputDim) would be a reasonable value
    # https://spark.apache.org/docs/2.3.0/api/scala/index.html#org.apache.spark.ml.feature.BucketedRandomProjectionLSHModel
    num_tables_selected = 10
    # bucket_len_selected = 5.0 * df_.count() ** (-1 / len(feature_columns))
    bucket_len_selected = 1.0
    print(f"num_tables = {num_tables_selected}\nbucket_len = {bucket_len_selected}")

    brp = BucketedRandomProjectionLSH(
        inputCol="FeatureVector_scaled",
        outputCol=f"hash_1bit",
        seed=4242,
        bucketLength=bucket_len_selected,
        numHashTables=num_tables_selected)
    model = brp.fit(df_)

    # key_df =
    # model.approxNearestNeighbors(df_,key_df, 3).show()

    df_.printSchema()
    df_.show()
    print(df_.show(20, False))

    # df_ \
    #     .write \
    #     .mode(saveMode='overwrite') \
    #     .option("header", True) \
    #     .orc("./scala-code/data/processed/ABzFeatures.orc")

    sc.stop()
    print("Done")
