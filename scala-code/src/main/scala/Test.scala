import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.SparkSession

object Test {
  var out_dir = ""

  def main(args: Array[String]): Unit = {
    // Turn off copious logging
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

//    out_dir = args(0)

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


    val NUM_SAMPLES = 10e3.toInt
    val count = sc.parallelize(1 to NUM_SAMPLES).filter { _ =>
      val x = math.random
      val y = math.random
      x*x + y*y < 1
    }.count()
    println(s"Pi is roughly ${4.0 * count / NUM_SAMPLES}")

    val arangoDBHandler = new ArangoDBHandler(spark)
    val users_count = arangoDBHandler.getUsers().count()
    println(s"Number of users in ArangoDB = $users_count")

    sc.stop()
    System.exit(0)
  }


}
