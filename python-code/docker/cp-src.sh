docker run -it \
  --mount source=ea_expr_src,destination=/src \
  -v $PWD/../:/external_src \
  sharwinbobde/ea_experiments /bin/bash -c "
cd /
echo Copying code
cp external_src/config.py src/
cp -r external_src/src src/
"
docker container create --name dummy -v ea_expr_src:/src/ ubuntu
docker cp $PWD/../src/ dummy:/src/
docker rm dummy
