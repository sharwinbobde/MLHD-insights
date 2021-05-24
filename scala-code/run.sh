
#--------- Spark Configuration START --------------------------

num_executors=1
executor_cores=12
executor_memory=5g
driver_memory=10g
jarfile='target/scala-2.12/MLHD-insights-assembly-1.0.jar'
outdir='file:/io/data/processed/'
#--------- Spark Configuration END ----------------------------


if [ $# -eq 0 ]
  then
    # No arguments supplied
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
  else
    # Option code is provided
    CHOICE=$1
fi


clear
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
            echo "Option did not match"
            exit 0
            ;;
esac

echo $mainClass

sudo docker run -it --rm -v "`pwd`":/io -v "`pwd`"/spark-events:/spark-events \
  spark-submit --class $mainClass \
               --num-executors $num_executors \
               --executor-cores $executor_cores \
               --executor-memory $executor_memory \
               --driver-memory $driver_memory \
               $jarfile $outdir
