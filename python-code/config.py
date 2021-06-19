num_cpu = 12
# experiment_years = [2005, 2008]
experiment_years = [2005, 2008, 2012]

# number of listens for an item upto which its considered as `novel`(unexplored/new)
novelty_threshold = 40

# out_dir = "/run/media/sharwinbobde/ExtraStorage/spark-output/"
spark_output_dir = "/home/sharwinbobde/Studies/Thesis/repos/MLHD-insights/scala-code/data/processed/"

ABz_directory_subset = "/run/media/sharwinbobde/SharwinThesis/ABz_features_subset/"
ABz_directory_aggregated = "/run/media/sharwinbobde/SharwinThesis/ABz_features_aggregated/"
ABz_directory = "/run/media/sharwinbobde/SharwinThesis/mlhd-ab-features/"
ABz_nested_directories = "/run/media/sharwinbobde/SharwinBackup/mlhd-ab/acousticbrainz-mlhd/"
ABz_orc_location = "/run/media/sharwinbobde/ExtraStorage/acousticbrainz-features-orc/"
ABz_cleaned_orc = "/run/media/sharwinbobde/ExtraStorage/2M-cleaned.orc"
ABz_GPU_hashed_output_dir = "/run/media/sharwinbobde/ExtraStorage/2M-hashed.parquet/"
ABz_GPU_hashed_coalesced = "/run/media/sharwinbobde/ExtraStorage/2M-hashed-coalesced.parquet"

CUDA_neighbour_search_stats_dir = "/run/media/sharwinbobde/ExtraStorage/neighbour_search/stats/"
CUDA_neighbour_search_df_dir = "/run/media/sharwinbobde/ExtraStorage/neighbour_search/test-data/"
CUDA_neighbour_search_df_result_dir = "/run/media/sharwinbobde/ExtraStorage/neighbour_search/test-results/"

EA_experiment_output_dir = "/run/media/sharwinbobde/ExtraStorage/EA-output/"
EA_experiment_reference_fronts = "/run/media/sharwinbobde/ExtraStorage/reference_fronts/"
