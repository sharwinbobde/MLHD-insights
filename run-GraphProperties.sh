java -Xmx512M -Dsbt.log.noformat=true -jar /home/sharwinbobde/.cache/JetBrains/IntelliJIdea2020.3/sbtexe/sbt-launch-1.3.9.jar assembly exit
sudo docker run -it --rm -v "`pwd`":/io -v "`pwd`"/spark-events:/spark-events \
  spark-submit --class GraphProperties \
               --num-executors 1 \
               --executor-cores 10 \
               --executor-memory 6G \
               target/scala-2.12/MLHD-insights-assembly-1.0.jar \
               file:/io/out_data/
