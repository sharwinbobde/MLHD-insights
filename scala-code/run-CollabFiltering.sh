source ./run.conf.sh

mainClass=CollaborativeFiltering

sudo docker run -it --rm -v "`pwd`":/io -v "`pwd`"/spark-events:/spark-events \
  spark-submit --class $mainClass \
               --num-executors $num_executors \
               --executor-cores $executor_cores \
               --executor-memory $executor_memory \
               --driver-memory $driver_memory \
               $jarfile $outdir
