import pandas as pd
from numba import njit, prange
import os
import numpy as np
from sklearn import preprocessing

from src.AcousticBrainz.ApproxNearestNeighboursCUDA import ApproxNearestNeighboursCUDA
from src.AcousticBrainz.LSHBias import LSHBias
import config
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from sklearn.utils import shuffle

LSH_NUM_BITS = int(2 ** 13)


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


def multi_gauss_clusters(n_clusters: int):
    cov = np.eye(num_features) * 0.001
    res = np.empty(shape=(0, num_features))
    for i in range(n_clusters):
        mu = np.array(np.random.uniform(-1, 1, num_features))
        res = np.vstack((res, np.random.multivariate_normal(mean=mu, cov=cov, size=int(samples / n_clusters))))
    res = preprocessing.MinMaxScaler().fit_transform(res)
    return res


def make_sample_data(set_: int):
    np.random.seed(set_ * 4347)
    if set_ == 1:  # Uniform distribution
        data = np.random.uniform(0, 1, size=(samples, num_features))
    if set_ == 2:  # 3 Gaussian distribution
        data = multi_gauss_clusters(n_clusters=3)
    if set_ == 3:  # 10 Gaussian distribution
        data = multi_gauss_clusters(n_clusters=10)
    df = pd.DataFrame()
    np.random.shuffle(data)
    df['vec'] = data.tolist()

    # find nearest neighbours
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=51, algorithm='ball_tree', leaf_size=30).fit(data)
    _, nbrs_indices = nbrs.kneighbors(data)
    for n_nbr in range(10, 51, 5):
        df[f"known_neighbours_{n_nbr}"] = [x[1:(n_nbr + 1)] for x in nbrs_indices]

    # hash using random hyperplane LSH
    import pycuda.gpuarray as gpuarray
    import skcuda.linalg as linalg
    import pycuda.autoinit
    linalg.init()
    os.environ['CUDA_HOME'] = "/opt/cuda/"
    vec_np = np.array(df['vec'].values.tolist(), dtype=np.float32)
    LSH = LSHBias(feature_dim=num_features, bits=LSH_NUM_BITS)
    W = np.array(LSH.W, dtype=np.float32)
    b_gpu = gpuarray.to_gpu(W)
    ones = np.ones(shape=(vec_np.shape[0], 1), dtype=np.float32)
    X = np.concatenate((vec_np, ones), axis=1)

    # do the matrix multiplication
    a_gpu = gpuarray.to_gpu(X)
    mul = linalg.mdot(a_gpu, b_gpu)
    # get binary: 1 if value >= 0, else 0
    res = gpuarray.if_positive(mul >= gpuarray.zeros(mul.shape, dtype=np.float32),
                               then_=gpuarray.ones_like(mul),
                               else_=gpuarray.zeros_like(mul))
    res = np.array(res.get(), dtype=np.uint32)

    # convert grouped bits to integers
    res = np_array_binary_to_grouped_integers(res)
    df[f"hash_{LSH_NUM_BITS}_bits"] = [x for x in res]
    df.to_parquet(f"{config.CUDA_neighbour_search_df_dir}df-{set_}.parquet", index=False)

    print("created test-data")


@njit(parallel=True)
def explore_neighbours_found(arr1, arr2):
    percents = np.empty(shape=samples, dtype=np.float32)
    for row in prange(samples):
        count = 0
        for i in range(arr1.shape[1]):
            for j in range(arr2.shape[1]):
                if arr1[row, i] == arr2[row, j]:
                    count += 1
        percents[row] = count / 10 * 100
    return percents


def plot_found_neighbours():
    fig = plt.figure(figsize=(10, 8))
    # fig.suptitle(set_desc[set_no])
    experiment_percents = range(10, 51, 5)
    actual_percents_found = []
    for n_nbr in experiment_percents:
        col_1 = np.array(df[f"known_neighbours_{n_nbr}"].to_list())
        col_2 = np.array(df['neighbours'].to_list())
        percents = explore_neighbours_found(col_1, col_2)
        actual_percents_found.append(percents.tolist())

    vis_df = pd.DataFrame({'percents': actual_percents_found, 'experiment_percent': experiment_percents})
    vis_df = vis_df.explode('percents').sort_values(by='experiment_percent')
    vis_df['percents'] = vis_df['percents'].astype('float64')
    sns.violinplot(x='experiment_percent', y='percents', data=vis_df, palette="flare",
                   bw=0.5,)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.ylim((-10, 110))
    plt.ylabel("percentage non-trivial neighbours found")
    plt.xlabel(f"known neighbours (50 = nearest {50 / samples * 100}% samples in dataset)")
    plt.savefig(f'../../images/approx-NN/set-{set_no}-found-violin.pdf')
    plt.show()


def plot_stats():
    fig = plt.figure(figsize=(10, 8))
    curr_indices = []
    for curr in stats.curriculum_level.unique():
        curr_indices.append(stats.curriculum_level.searchsorted(curr, side='left'))

    ylims = (np.min(stats.quality_indicator_mean) - np.percentile(stats.quality_indicator_std, 0.99),
             np.max(stats.quality_indicator_mean) + np.percentile(stats.quality_indicator_std, 0.99))

    plt.vlines(np.array(curr_indices) - 0.5,
               # ymin=np.mean(stats.quality_indicator_mean) - np.percentile(stats.quality_indicator_std, .95),
               # ymax=np.mean(stats.quality_indicator_mean) + np.percentile(stats.quality_indicator_std, .95),
               ymin=ylims[0], ymax=ylims[1],
               colors='r', linewidth=1,
               label='curriculum start')

    plt.errorbar(x=stats.index, y=stats['quality_indicator_mean'], yerr=stats['quality_indicator_std'],
                 fmt='o-', elinewidth=1, markersize=3,
                 label='quality indicator\nmean with std.')

    plt.grid(True)
    plt.ylim(ylims)
    plt.ylabel("quality indicator value")
    plt.xlabel("absolute iteration number")
    plt.legend(fontsize=16)
    plt.savefig(f'../../images/approx-NN/set-{set_no}-stats.pdf')
    plt.show()


if __name__ == '__main__':
    # for filename in glob(config.ABz_GPU_hashed_coalesced + "/part-*.parquet"):
    #     df = pd.read_parquet(filename).head(1000)
    # for filename in glob(config.spark_output_dir + "recordings_in_ABz.orc/part-*.orc"):
    #     subset_MBIDs = pd.read_orc(filename)
    # df = df.join(subset_MBIDs.set_index('rec_MBID'), on='rec_MBID', how='inner')
    set_desc = {
        1: "Uniformly distributed data",
        2: "3-Gaussian data",
        3: "10-Gaussian data"
    }
    samples = 10000
    num_features = 200
    for set_no in range(1, 4):
        ## Step 1
        # make_sample_data(set_=set_no)

        ## Step 2
        # df = pd.read_parquet(f"{config.CUDA_neighbour_search_df_dir}df-{set_no}.parquet")
        # df, stats = ApproxNearestNeighboursCUDA().run(df, features_column=f'hash_{LSH_NUM_BITS}_bits', verbose=True)
        # df.to_parquet(f"{config.CUDA_neighbour_search_df_result_dir}df-{set_no}.parquet", index=False)
        # stats.to_parquet(f"{config.CUDA_neighbour_search_df_result_dir}df-{set_no}-stats.parquet", index=False)

        df = pd.read_parquet(f"{config.CUDA_neighbour_search_df_result_dir}df-{set_no}.parquet")
        stats = pd.read_parquet(f"{config.CUDA_neighbour_search_df_result_dir}df-{set_no}-stats.parquet")
        plot_found_neighbours()
        # plot_stats()
