#
# Copyright (c) 2020. Sharwin P. Bobde. Location: Delft, Netherlands. Coded for Master's Thesis project.
#

# user: root
# pass: "Happy2Help!"
database_path="/run/media/sharwinbobde/SharwinThesis/agarodb_data"

arangodb --starter.mode single --starter.data-dir $database_path \
        --dbservers.rocksdb.write-buffer-size 100123400 \
        --dbservers.rocksdb.max-write-buffer-number 3 \
        --dbservers.rocksdb.total-write-buffer-size 1012340000 \
        --dbservers.rocksdb.dynamic-level-bytes false \
        --dbservers.cache.size 20123400 \
        --all.rocksdb.block-cache-size 20560000 \
        --all.rocksdb.enforce-block-cache-size-limit true \
