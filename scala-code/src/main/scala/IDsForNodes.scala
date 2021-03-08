import com.arangodb.spark.{ArangoSpark, WriteOptions}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.sql.{Dataset, Row, SparkSession}

object IDsForNodes {

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
        .appName("IDsForNodes")
        .getOrCreate()

    val sc = spark.sparkContext
    val arangoDBHandler = new ArangoDBHandler(spark)
    addIDsToNodes(arangoDBHandler.getUsers(), arangoDBHandler.getArtists(), arangoDBHandler.getRecordings())

    // Stop the underlying SparContext
    sc.stop
    System.exit(0)
  }

  def addIDsToNodes(users: Dataset[Row], artists: Dataset[Row], recordings: Dataset[Row]): Unit = {
    val users_with_id = users.withColumn("node_id", monotonically_increasing_id() * 10 + 1)
    val artists_with_id = artists.withColumn("node_id", monotonically_increasing_id() * 10 + 2)
    val recs_with_id = recordings.withColumn("node_id", monotonically_increasing_id() * 10 + 3)

    ArangoSpark.saveDF(users_with_id, "users", WriteOptions(database = "MLHD_processing", method = WriteOptions.UPDATE))
    ArangoSpark.saveDF(artists_with_id, "artists", WriteOptions(database = "MLHD_processing", method = WriteOptions.UPDATE))
    ArangoSpark.saveDF(recs_with_id, "recordings", WriteOptions(database = "MLHD_processing", method = WriteOptions.UPDATE))
  }
}