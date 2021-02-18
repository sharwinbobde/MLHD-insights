source ./run.conf.sh
HEIGHT=15
WIDTH=40
CHOICE_HEIGHT=4
BACKTITLE="Backtitle here"
TITLE="Title here"
MENU="Choose one of the following options:"

OPTIONS=(1 "Setup Experiments"
         2 "Stats for Graph Properties"
         3 "Divide Dataset for Experiments"
         4 "User-Record Collaborative Filtering")

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