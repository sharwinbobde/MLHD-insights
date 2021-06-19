#
# Copyright (c) 2020. Sharwin P. Bobde. Location: Delft, Netherlands. Coded for Master's Thesis project.
#

sudo sysctl -w vm.swappiness=80

# user: root
# pass: "Happy2Help!"
#database_path="/run/media/sharwinbobde/SharwinThesis/agarodb_data"
#database_path="/run/media/sharwinbobde/SharwinThesis/agarodb_data_25"
#database_path="/run/media/sharwinbobde/SharwinThesis/agarodb_data_large"
# database_path="/run/media/sharwinbobde/ExtraStorage/agarodb_data_large"
database_path="/home/sharwinbobde/Studies/Thesis/agarodb_data_200"

# 1 - Raise the vm map count value
# sudo sysctl -w "vm.max_map_count=2048000"

sudo bash -c "echo 0 > /proc/sys/vm/overcommit_memory"

arangodb --starter.mode single --starter.data-dir $database_path \
        --dbservers.rocksdb.write-buffer-size 10012340 \
        --dbservers.rocksdb.max-write-buffer-number 2 \
        --dbservers.rocksdb.total-write-buffer-size 101234000 \
        --dbservers.cache.size 10000000 \
        --all.arangosearch.threads-limit 2 \
        --all.rocksdb.block-cache-size 1000000 \
        --all.rocksdb.enforce-block-cache-size-limit true \
        --all.rocksdb.max-background-jobs 10 \
        --all.server.maintenance-threads 4 \
        --all.server.maximal-threads 10 \
        --all.server.minimal-threads 5 \
        --dbservers.batch-size 100000 
        
#         --dbservers.rocksdb.intermediate-commit-size 10000000
        
        
#         --all.query.memory-limit 5056000000
#         --dbservers.rocksdb.dynamic-level-bytes false \
