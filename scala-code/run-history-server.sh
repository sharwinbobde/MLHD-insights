SPARK_EVENTS_VOLUME=sharwinbobde-spark-events

# sudo docker run -it --rm -v $SPARK_EVENTS_VOLUME:/spark-events -p 18080:18080 spark-history-server
# sudo docker run -it --rm -v $PWD/spark-events:/spark-events -p 18080:18080 spark-history-server
#DIR=/home/sharwinbobde/Studies/Thesis/spark-events-serg3/events
DIR=/home/sharwinbobde/Studies/Thesis/200-parts-record-keeping/spark-events-serg3/events

sudo docker run -it --rm \
    -v $DIR:/spark-events \
    -p 18080:18080 spark-history-server