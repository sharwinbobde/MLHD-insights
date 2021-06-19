OUTPUT_VOLUME_NAME=sharwinbobde-spark-output
SPARK_EVENTS_VOLUME=sharwinbobde-spark-events
NETWORK_BRIDGE=sharwinbobde-nw-bridge

sudo docker run -it --rm \
      --network $NETWORK_BRIDGE \
      -v "`pwd`":/io \
      --mount source=$OUTPUT_VOLUME_NAME,destination=/spark-output \
      --mount source=$SPARK_EVENTS_VOLUME,destination=/spark-events \
      ubuntu