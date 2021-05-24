docker run -it \
  -v $PWD:$PWD \
  -v /run/media/sharwinbobde:/run/media/sharwinbobde \
  test-docker /bin/bash -c "
cd $PWD/src
export PYTHONPATH=$PYTHONPATH:$PWD
python EA_Experiments.py
"
#docker run -it -v $PWD:$PWD test-docker /bin/bash -c "cd $PWD; ls"
