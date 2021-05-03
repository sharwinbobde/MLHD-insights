from datetime import datetime
from hurry.filesize import filesize
from numba import cuda
import numpy as np
import pandas as pd
import os
os.environ['CUDA_HOME'] = "/opt/cuda/"

ARCHIVE_REGISTER_THRESHOLD = 1000
NEIGHBOURS_ARCHIVE_SIZE = 20

SEARCH_SIZE = 200

UPDATE_SUPPORT_NEIGHBOURS = 3

CURRICULUM_SIZE = 5000

rows = CURRICULUM_SIZE  # start with small number of records and increase at every level
ITERATIONS_PER_CURRICULUM_LEVEL = 10


@cuda.jit
def neighbour_search_step(H, search_i, archive_i, archive_d, rows_this_curr, curriculum_num, iteration_num):
    x = cuda.grid(1)
    # execute for each row
    N = rows_this_curr[0]
    if x >= N:
        return

    for run in range(SEARCH_SIZE):
        if run < (SEARCH_SIZE / 2):
            search_index = search_i[x, run] % N
        else:
            other_row = (x + curriculum_num[0] * ITERATIONS_PER_CURRICULUM_LEVEL + iteration_num[0]) % N
            search_index = search_i[other_row, run] % N
        if search_index == x:
            continue

        dist = 0
        for i in range(256):  # for all integer encodings of LSH bits
            # xor to see flipped bits
            var0 = H[x, i] ^ H[search_index, i]

            # population count algorithm
            var0 -= var0 >> 1 & 1431655765
            var0 = (var0 & 858993459) + (var0 >> 2 & 858993459)
            var0 = var0 + (var0 >> 4) & 252645135
            var0 += var0 >> 8
            var0 += var0 >> 16
            dist += var0 & 63

        # find location to insert
        FLAG_NOT_duplicate = True
        insert_index = 0
        while archive_d[x, insert_index] <= dist and insert_index < NEIGHBOURS_ARCHIVE_SIZE:
            if archive_i[x, insert_index] == search_index:
                FLAG_NOT_duplicate = False
                break
            insert_index += 1

        if insert_index < NEIGHBOURS_ARCHIVE_SIZE and FLAG_NOT_duplicate:
            if archive_i[x, insert_index] == -1:
                # add in an empty space
                archive_i[x, insert_index] = search_index
                archive_d[x, insert_index] = dist
            else:
                # insert item and move other items right
                temp_i_1, temp_d_1 = archive_i[x, insert_index], archive_d[x, insert_index]
                archive_i[x, insert_index] = search_index
                archive_d[x, insert_index] = dist

                itr = insert_index + 1
                while itr < NEIGHBOURS_ARCHIVE_SIZE:
                    temp_i_2, temp_d_2 = archive_i[x, itr], archive_d[x, itr]
                    archive_i[x, itr] = temp_i_1
                    archive_d[x, itr] = temp_d_1
                    temp_i_1, temp_d_1 = temp_i_2, temp_d_2
                    itr += 1


@cuda.jit()
def update_step(search_i, archive_i, rows_this_curr):
    """
    Update step where points look at closest neighbours' archives and add it to their own search array
    search array of any point can have duplicate entries but the neighbour-search makes sure duplicates are not added to the archive.
    """
    x = cuda.grid(1)
    # execute for each row
    if x >= rows_this_curr[0]:
        return
    update_window = int(SEARCH_SIZE / 2 / UPDATE_SUPPORT_NEIGHBOURS)

    for i in range(UPDATE_SUPPORT_NEIGHBOURS):
        close_neighbour = archive_i[x, i]
        for j in range(update_window):
            archive_item_to_search = archive_i[close_neighbour, j]
            search_i[x, int(i * update_window + j)] = archive_item_to_search


def calc_quality(archive_d_qual, rows_, rows_curr_gpu):
    blocks_per_grid_quality = min([rows_, 1000])
    threads_per_block_quality = int(rows_ / blocks_per_grid_quality) + 1
    quality_sum = cuda.to_device(np.zeros(shape=rows_, dtype=np.int32))

    @cuda.jit
    def quality_sum_kernal(archive_d, qual_sum, rows_this_curr):
        x = cuda.grid(1)
        # execute for each row
        if x >= rows_this_curr[0]:
            return
        sum_ = 0
        for i in range(int(NEIGHBOURS_ARCHIVE_SIZE / 2)):
            sum_ += archive_d[x, i]
        qual_sum[x] = sum_

    quality_sum_kernal[blocks_per_grid_quality, threads_per_block_quality](archive_d_qual, quality_sum,
                                                                           rows_curr_gpu)
    cuda.synchronize()
    quality_sum = quality_sum.copy_to_host()
    return np.mean(quality_sum), np.std(quality_sum)


def get_observed_dist():
    # TODO
    pass


def run(df: pd.DataFrame, features_column: str):
    stats = []
    ROWS_MAX = df.shape[0]
    CURRICULUM_LEVELS = int(ROWS_MAX / CURRICULUM_SIZE)
    if ROWS_MAX % CURRICULUM_SIZE != 0:
        CURRICULUM_LEVELS += 1
    A = np.array(df[features_column].values.tolist(), dtype=np.int32)

    A_gpu = cuda.to_device(A)
    print(f"A_gpu.alloc_size = {A_gpu.alloc_size}")

    archive_index = cuda.to_device(np.ones(shape=(ROWS_MAX, NEIGHBOURS_ARCHIVE_SIZE), dtype=np.uint32) * -1)
    print(f"archive_index.alloc_size = {filesize.size(archive_index.alloc_size)}")

    archive_dist = cuda.to_device(
        np.ones(shape=(ROWS_MAX, NEIGHBOURS_ARCHIVE_SIZE), dtype=np.uint16) * np.iinfo(np.uint16).max)
    print(f"archive_dist.alloc_size = {filesize.size(archive_dist.alloc_size)}")

    search_matrix_index = cuda.to_device(
        np.random.randint(low=0, high=np.iinfo(np.int32).max, size=(ROWS_MAX, SEARCH_SIZE), dtype=np.uint32))
    print(f"search_matrix_index.alloc_size = {filesize.size(search_matrix_index.alloc_size)}")
    FLAG_tempering = False
    for curriculum_level in range(CURRICULUM_LEVELS):
        print(f"CURRICULUM_LEVEL = {curriculum_level + 1}/{CURRICULUM_LEVELS}")
        rows = int(CURRICULUM_SIZE * (curriculum_level + 1))
        if rows >= ROWS_MAX:
            rows = ROWS_MAX
            FLAG_tempering = True

        print(f"number of rows (recordings) = {rows}")
        curriculum_level_gpu = cuda.to_device(np.array([curriculum_level], dtype=np.int32))
        rows_gpu = cuda.to_device(np.array([rows], dtype=np.int32))

        blocks_per_grid = min([int(rows / 10), 50000])
        threads_per_block = int(rows / blocks_per_grid) + 1

        with cuda.defer_cleanup():
            # iterations of the Search and Update steps
            iterations_this_round = ITERATIONS_PER_CURRICULUM_LEVEL
            if FLAG_tempering:
                iterations_this_round = int(ITERATIONS_PER_CURRICULUM_LEVEL * 5)

            for iteration in range(iterations_this_round):
                iteration_gpu = cuda.to_device(np.array([iteration], dtype=np.int32))
                # neighbour search
                neighbour_search_step[blocks_per_grid, threads_per_block](
                    A_gpu, search_matrix_index, archive_index, archive_dist, rows_gpu, curriculum_level_gpu,
                    iteration_gpu)

                # Wait for GPU to complete
                cuda.synchronize()

                # update search direction based on archive: sees archive_index and updates search_matrix_index
                update_step[blocks_per_grid, threads_per_block](search_matrix_index, archive_index, rows_gpu)

                # print(search_matrix_index.copy_to_host()[200000, :])

                # Evaluate Solution Quality: sees archive_dist and updates search_i
                quality_mean, quality_std = calc_quality(archive_dist, rows, rows_gpu)
                print(f" quality indicator: mean = {quality_mean} \t std={quality_std}")

                stats.append([datetime.now(), curriculum_level, iteration, rows, quality_mean, quality_std])

                # wait for all computations to complete
                cuda.synchronize()
