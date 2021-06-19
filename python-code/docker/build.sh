echo Building sharwinbobde/ea_experiments:latest
docker build --no-cache -t sharwinbobde/ea_experiments:latest .

# change this to appropriate data location
docker volume create ea_expr_data
docker volume create ea_expr_src
docker volume create ea_expr_out


DATA_DIR=/home/sharwinbobde/Studies/Thesis/repos/MLHD-insights/scala-code/data/processed/

docker run \
  --name ea_expr_build \
  -v $DATA_DIR:/external_data \
  --mount source=ea_expr_data,destination=/data \
  --mount source=ea_expr_src,destination=/src \
  -v $PWD/../:/external_src \
  sharwinbobde/ea_experiments /bin/bash -c "
echo =========================================
cd /
ls /src
ls /data

# copy required code to the src directory
echo Copying code
cp external_src/config.py src/
cp -r external_src/src src/

# copy required code to the src directory
req_dirs=\"holdout item_listens_per_year.parquet output-EA output-RS\"
for val in \$req_dirs; do
    echo Copying data/\$val
#    cp -r external_data/\$val data/
done
echo All data transferred
echo =========================================
"
CONTAINER_ID=`docker ps -a -aqf "name=ea_expr_build"`
echo CONTAINER_ID = $CONTAINER_ID

# cleanup
docker container rm $CONTAINER_ID