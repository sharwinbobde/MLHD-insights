import re
from glob import glob
import os
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
from pyspark.sql import SparkSession
import pyarrow.orc
import pandas as pd
from numba import njit, prange
from tqdm import tqdm
import pycuda
import config
from src.AcousticBrainz.LSHBias import LSHBias

linalg.init()

# convert to 0,1 instead of +1,-1
def sign_to_binary(x):
    return 0 if x < 0 else 1


vectorized_sign_to_func = np.vectorize(sign_to_binary)


@njit(parallel=True)
def np_array_binary_to_grouped_integers(np_arr):
    num_records = np_arr.shape[0]
    num_ints = int(LSH_NUM_BITS / 32)
    # big array that goes to pandas DataFrame as a column
    out = np.empty(shape=(num_records, num_ints), dtype=np.uint32)

    for i in prange(num_records):
        out_sub = np.empty(shape=num_ints, dtype=np.uint32)
        for j in range(num_ints):
            bits = np_arr[i, 32 * j: 32 * j + 32]
            # int_val = int(''.join(map(str, arr)), base=2)
            int_val = 0
            for digit in bits:
                int_val = (int_val << 1) + digit
            out_sub[j] = int_val
        out[i, :] = out_sub
    return out


if __name__ == '__main__':
    basepath = "/run/media/sharwinbobde/ExtraStorage/2M-scaled-array-1.orc/"
    # we need a sample file to get the number of features.
    sample_file = "part-00012-7d53a446-d692-475a-853f-9e55ccc8e9fa-c000.snappy.orc"
    # df = pd.read_orc(basepath + sample_file)
    # df = df.rename(columns={"FeatureVector_all_features": "vec"})
    #
    # num_records = df.shape[0]
    # num_features = len(df['vec'][0])
    # print(num_features)
    # print(f"num_records = {num_records}")
    # print(f"num_features = {num_features}")
    #
    LSH_NUM_BITS = int(2 ** 13)
    #
    # LSH = LSHBias(feature_dim=num_features, bits=LSH_NUM_BITS)
    #
    # W = np.array(LSH.W, dtype=np.float32)
    # b_gpu = gpuarray.to_gpu(W)  # reuse this every time

    # count = 0
    # # hashing different .orc DataFrames
    # for filename in tqdm(glob(basepath + "part-*.orc")):
    #     df = pd.read_orc(filename)
    #     df = df.rename(columns={"FeatureVector_all_features": "vec"})
    #     count += 1
    #     vec_np = np.array(df['vec'].values.tolist(), dtype=np.float32)
    #     # add bias term
    #     ones = np.ones(shape=(vec_np.shape[0], 1), dtype=np.float32)
    #     X = np.concatenate((vec_np, ones), axis=1)
    #
    #     # do the matrix multiplication
    #     a_gpu = gpuarray.to_gpu(X)
    #     mul = linalg.mdot(a_gpu, b_gpu)
    #     # get binary: 1 if value >= 0, else 0
    #     res = gpuarray.if_positive(mul >= gpuarray.zeros(mul.shape, dtype=np.float32),
    #                                then_=gpuarray.ones_like(mul),
    #                                else_=gpuarray.zeros_like(mul))
    #     res = np.array(res.get(), dtype=np.uint32)
    #
    #     # convert grouped bits to integers
    #     res = np_array_binary_to_grouped_integers(res)
    #
    #     df[f"hash_{LSH_NUM_BITS}_bits"] = [x for x in res]
    #     df = df[["rec_MBID", f"hash_{LSH_NUM_BITS}_bits"]]
    #     df.to_parquet(f"{config.ABz_GPU_hashed_output_dir}{count}.parquet", index=False)

    # # save as a single parquet file
    # spark = SparkSession \
    #     .builder \
    #     .appName("hashed file coalesce") \
    #     .config("spark.executor.memory", "2G") \
    #     .config("spark.driver.memory", "2G") \
    #     .getOrCreate()
    #
    # sc = spark.sparkContext
    # spark.read.parquet(config.ABz_GPU_hashed_output_dir)\
    #     .coalesce(1) \
    #     .write.mode(saveMode='overwrite') \
    #     .parquet(config.ABz_GPU_hashed_coalesced)
    #
    # sc.stop()
    #
