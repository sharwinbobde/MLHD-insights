#--------- Spark Configuration START --------------------------

num_executors=1
executor_cores=12
executor_memory=20g
driver_memory=3g
jarfile='target/scala-2.12/MLHD-insights-assembly-1.0.jar'
outdir='file:/io/data/processed/'
#--------- Spark Configuration END ----------------------------



HEIGHT=15
WIDTH=40
CHOICE_HEIGHT=4
BACKTITLE="Backtitle here"
TITLE="Title here"
MENU="Choose one of the following options:"

OPTIONS=(1 "Setup Experiments"
         2 "Stats for Graph Properties"
         3 "Divide Dataset for Experiments"
         4 "User-Record Collaborative Filtering"
         5 "User-Artist Collaborative Filtering"
         )

CHOICE=$(dialog --clear \
                --backtitle "$BACKTITLE" \
                --title "$TITLE" \
                --menu "$MENU" \
                $HEIGHT $WIDTH $CHOICE_HEIGHT \
                "${OPTIONS[@]}" \
                2>&1 >/dev/tty)

clear
case $CHOICE in
        1)
            mainClass=SetupExperiments
            ;;
        2)
            mainClass=GraphProperties
            ;;
        3)
            mainClass=DatasetDivision
            ;;
        4)
            mainClass=CollaborativeFiltering_UserRecord
            ;;
        5)
            mainClass=CollaborativeFiltering_UserArtist
            ;;
        *)
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