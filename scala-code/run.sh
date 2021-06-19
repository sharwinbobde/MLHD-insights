
#--------- Spark Configuration START --------------------------

num_executors=1
executor_cores=12
executor_memory=5g
driver_memory=10g
jarfile='target/scala-2.12/MLHD-insights-assembly-1.0.jar'
OUTPUT_VOLUME_NAME=sharwinbobde-spark-output
SPARK_EVENTS_VOLUME=sharwinbobde-spark-events
NETWORK_BRIDGE=sharwinbobde-nw-bridge
outdir='file:/spark-output/data/processed/'
#--------- Spark Configuration END ----------------------------
echo $#

if [ $# -eq 0 ]
  then
    # No arguments supplied
    echo here
    HEIGHT=20
    WIDTH=40
    CHOICE_HEIGHT=4
    BACKTITLE="Backtitle here"
    TITLE="Title here"
    MENU="Choose one of the following options:"

    OPTIONS=(
             1 "add IDs for Nodes"
             2 "MLHD Analysis"
             3 "Stats for Graph Properties"
             4 "Divide Dataset for Experiments"
             5 "User-Record Collaborative Filtering"
             6 "User-Artist Collaborative Filtering"
             7 "Save MBIDs With Features"
             8 "ABz Nearest-Neighbour Recommender"
             9 "Tailored Recommender"
             )

    CHOICE=$(dialog --clear \
                    --backtitle "$BACKTITLE" \
                    --title "$TITLE" \
                    --menu "$MENU" \
                    $HEIGHT $WIDTH $CHOICE_HEIGHT \
                    "${OPTIONS[@]}" \
                    2>&1 >/dev/tty)
    echo $CHOICE
  else
    # Option code is provided
    CHOICE=$1
fi


#clear
case $CHOICE in
        1)
            mainClass=IDsForNodes
            ;;
        2)
            mainClass=MLHD_Analysis
            ;;
        3)
            mainClass=GraphProperties
            ;;
        4)
            mainClass=DatasetDivision
            ;;
        5)
            mainClass=CollaborativeFiltering_UserRecord
            ;;
        6)
            mainClass=CollaborativeFiltering_UserArtist
            ;;
        7)
            mainClass=SaveMBIDsWithFeatures
            ;;
        8)
            mainClass=ABzRecommender
            ;;
        9)
            mainClass=TailoredRecommender
            ;;
        *)
            echo "Option did not match. run as \"./run.sh 0\" to show menu."
            exit 0
            ;;
esac

echo $mainClass

docker volume create $OUTPUT_VOLUME_NAME
docker volume create $SPARK_EVENTS_VOLUME
docker network create --driver bridge $NETWORK_BRIDGE

# shellcheck disable=SC2006
docker run -it --rm \
      --network $NETWORK_BRIDGE \
      -v "`pwd`":/io \
      --mount source=$OUTPUT_VOLUME_NAME,destination=/spark-output \
      --mount source=$SPARK_EVENTS_VOLUME,destination=/spark-events \
      spark-submit --class $mainClass \
                   --num-executors $num_executors \
                   --executor-cores $executor_cores \
                   --executor-memory $executor_memory \
                   --driver-memory $driver_memory \
                   $jarfile $outdir
