SAVE_LOCATION=$PWD'/out-data/'
docker volume create ea_expr_data
docker volume create ea_expr_src
docker volume create ea_expr_out

docker run -it \
  --mount source=ea_expr_out,destination=/out \
  -v $SAVE_LOCATION:/external_out \
  sharwinbobde/ea_experiments /bin/bash -c "
#tree /out
cp -r /out/* /external_out
"