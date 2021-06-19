# Run Parameters
RS_or_EA=RS
year=2012
set=2

# ===================================

docker run -it \
  --mount source=ea_expr_data,destination=/data,readonly \
  --mount source=ea_expr_src,destination=/src \
  --mount source=ea_expr_out,destination=/out \
  sharwinbobde/ea_experiments /bin/bash -c "
cd /src
export PYTHONPATH=$PYTHONPATH:/src/
python src/EA_Experiments.py 1 /data/ /out/ $RS_or_EA $year $set
ls /out
tree /out
"

#docker run -it -v $PWD:$PWD test-docker /bin/bash -c "cd $PWD; ls"
