import os
import re

from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, udf, col, lit, count, isnull, when
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, ArrayType
from pyspark.sql.utils import ParseException

import config


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


@udf(returnType=StringType())
def MBID_from_filename(s):
    out = re.search(r'[^\/][^\/]*(?=\.json)', s).group(0)
    return out


if __name__ == '__main__':
    driver_memory = '5G'
    pyspark_submit_args = ' --driver-memory ' + driver_memory + ' pyspark-shell'
    os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args
    spark = SparkSession \
        .builder \
        .appName("ABz Preprocessing + LSH") \
        .config("spark.executor.memory", "2G") \
        .config("spark.driver.memory", "5G") \
        .getOrCreate()

    sc = spark.sparkContext


    # ignore_subdirs = [] # already computed
    # for a in range(1):
    #     for b in range(16):
    #         subdir = hex(a)[2:] + hex(b)[2:]
    #         ignore_subdirs.append(subdir)
    #
    # for a in range(16):
    #     for b in range(16):
    #         subdir = hex(a)[2:] + hex(b)[2:]
    #         print(subdir)
    #         if subdir in ignore_subdirs:
    #             print('skipped')
    #             continue
    #
    #         folder = config.ABZ_nested_directories + subdir + '/'
    #         df_ = spark.read.json(folder, multiLine=True) \
    #             .withColumn("filename", input_file_name())
    #
    #         df_.coalesce(1) \
    #             .write \
    #             .mode(saveMode='overwrite') \
    #             .orc(config.ABZ_orc_location + subdir)
    # print("Done Writing")

    # def col_union_expr(cols, allCols):
    #     out = []
    #     for x in allCols:
    #         if x in cols:
    #             out.append(col(x))
    #         else:
    #             out.append(lit(None).alias(x))
    #     return out


    print("Reading")
    for a in range(16):
        for b in range(16):
            subdir = hex(a)[2:] + hex(b)[2:]
            print(subdir)
            folder = config.ABZ_orc_location + subdir + '/'
            if a == 0 and b == 0:
                df_ = spark.read.orc(folder).drop("metadata")
                df_ = flatten_df_structs(df_)
            else:
                try:
                    df__ = spark.read.orc(folder).drop("metadata")
                    df__ = flatten_df_structs(df__)
                    # union for the intersect of cols
                    union_df_cols = set(df__.columns).union(set(df_.columns))
                    intersect_df_cols = set(df__.columns).intersection(set(df_.columns))
                    print(f"rejected = {len(set(df__.columns).difference(intersect_df_cols))}")
                    df_ = df_.select(*intersect_df_cols) \
                        .union(df__.select(*intersect_df_cols)).select(*intersect_df_cols)
                except ParseException:
                    print("oh damn!")

            print(len(df_.columns))
    # df_.show()

    print(f"final contained columns = {df_.columns}")
    print("Null Distribution is as follows")
    df_.select([count(when(isnull(c), c)).alias(c) for c in df_.columns]).show()
    sc.stop()
    print("Done")
