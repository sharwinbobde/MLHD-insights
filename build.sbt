name := "MLHD-insights"
version := "1.0"
scalaVersion := "2.12.2"


scalastyleFailOnWarning := true

fork in run := true

val sparkVersion = "2.4.7"
resolvers ++= Seq(
  "osgeo" at "https://repo.osgeo.org/repository/release",
  "confluent" at "https://packages.confluent.io/maven"
)

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
  "com.arangodb" %% "arangodb-spark-connector" % "1.1.0"
//  "org.locationtech.geomesa" %% "geomesa-spark-jts" % "3.0.0",
//  "com.uber" % "h3" % "3.6.4"
)
