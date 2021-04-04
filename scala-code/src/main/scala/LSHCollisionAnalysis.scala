import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

import utils.LSHUtils

object LSHCollisionAnalysis {
  var out_dir = ""

  def main(args: Array[String]) {
    // Turn off copious logging
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    out_dir = args(0)

    val spark: SparkSession =
      SparkSession
        .builder()
        .config("arangodb.hosts", "plb.sharwinbobde.com:8529") // system ip as docker ip won't be loopback
        .config("arangodb.user", "root")
        .config("arangodb.password", "Happy2Help!")
        .config("spark.sql.crossJoin.enabled", value = true)
        .appName("CollabFiltering User-Artist")
        .getOrCreate()

    val sc = spark.sparkContext
    var df = spark.read.orc(out_dir + "ABzFeatures.orc")
    df.printSchema()

    val count_records = df.count()
    println(s"number of records = $count_records")

    df = df.as("_1").crossJoin(df.as("_2"))
    df.printSchema()

    val LSH_bits = 8192
    val hash_cols = Array(
      s"all_features_hash_${LSH_bits}_bits",
      s"tonal_hash_${LSH_bits}_bits",
      s"rhythm_hash_${LSH_bits}_bits",
      s"lowlevel_hash_${LSH_bits}_bits")

    for (selected_col <- hash_cols) {
      println(selected_col)
      df = df.withColumn("test_dist",
        LSHUtils.hammingDistUDF(col(s"_1.$selected_col"), col(s"_2.$selected_col")))

      val num_collisions = count_records - df
        .select("test_dist")
        .filter("test_dist == 0")
        .count()
      println(s"number of collisions = $num_collisions out of $count_records")
    }

    // Stop the underlying SparkContext
    sc.stop
    System.exit(0)

  }

}
