# start docker daemon
sudo systemctl start docker

# build sbd
sudo docker build \
--build-arg BASE_IMAGE_TAG="8" \
--build-arg SBT_VERSION="1.3.13" \
--build-arg SCALA_VERSION="2.12.12" \
-t sbt \
github.com/hseeberger/scala-sbt.git#:debian

# build spark-shell
sudo docker build --target spark-shell -t spark-shell .

# build spark-submit
sudo docker build --target spark-submit -t spark-submit .


# build spark-history-server
sudo docker build --target spark-history-server -t spark-history-server .



