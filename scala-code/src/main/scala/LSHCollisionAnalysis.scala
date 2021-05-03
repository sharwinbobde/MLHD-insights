import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import utils.LSHUtils


object LSHCollisionAnalysis {
  val LSH_bits: Int = scala.math.pow(2, 13).toInt
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
    var df = spark.read.parquet(out_dir + "2M-hashed.parquet")
    df.printSchema()

    val count_records = df.count()
    println(s"number of records = $count_records")

    df = df.as("_1").crossJoin(df.as("_2"))
    df.printSchema()

    val hash_col = s"hash_${LSH_bits}_bits"

    df = df.withColumn("test_dist",
      LSHUtils.hammingDistUDF(col(s"_1.$hash_col"), col(s"_2.$hash_col")))

    val num_collisions = count_records - df
      .select("test_dist")
      .filter("test_dist == 0")
      .count()
    println(s"number of collisions = $num_collisions out of $count_records")

    // Stop the underlying SparkContext
    sc.stop
    System.exit(0)

  }

}
