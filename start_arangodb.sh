#
# Copyright (c) 2020. Sharwin P. Bobde. Location: Delft, Netherlands. Coded for Master's Thesis project.
#

# user: root
# pass: Happy2Help!
database_path=/run/media/sharwinbobde/SharwinThesis/agarodb_data
#database_path="/home/sharwinbobde/tudelft-HPC-cluster/arangodb_data/"
arangodb --starter.mode single --starter.data-dir $database_path &
