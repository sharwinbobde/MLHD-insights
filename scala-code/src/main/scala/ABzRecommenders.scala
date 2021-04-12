import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.{Dataset, Row, SaveMode, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel
import utils.LSHUtils

object ABzRecommenders {
  val user_colname = "user_id"
  val item_colname = "rec_id"
  val items_to_recommend = 100
  val rating_lower_threshold = 25
  val experiment_years: Array[Int] = Array(2005, 2008, 2012)
  val feature_sets: Array[String] = Array("all_features", "tonal", "rhythm", "lowlevel")


  val LSH_NUM_BITS: Int = scala.math.pow(2, 13).toInt
  val hash_cols = Map(
    "all_features" -> s"all_features_hash_${LSH_NUM_BITS}_bits",
    "tonal" -> s"tonal_hash_${LSH_NUM_BITS}_bits",
    "rhythm" -> s"rhythm_hash_${LSH_NUM_BITS}_bits",
    "lowlevel" -> s"lowlevel_hash_${LSH_NUM_BITS}_bits")

  var out_dir = ""

  def main(args: Array[String]): Unit = {
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
        .appName("ABz Recommenders")
        .getOrCreate()

    val sc = spark.sparkContext
    val arangoDBHandler = new ArangoDBHandler(spark)
    var feature_hashes = spark.read.orc(out_dir + "ABzFeatures.orc")

    val recs = arangoDBHandler.getRecordings()
      .select(col("_key").as("rec_MBID"), col("node_id").as("rec_id"))
    val user_recs_interactions = arangoDBHandler.getUserToRecordingEdges()
      .persist(StorageLevel.DISK_ONLY)
    user_recs_interactions.printSchema()

    // give rec_id to feature hashes
    feature_hashes = feature_hashes
      .join(recs, Seq("rec_MBID"), "inner")
      .drop("rec_MBID")
      .persist(StorageLevel.DISK_ONLY)
    feature_hashes.printSchema()
    val num_valid_records = feature_hashes.count()
    println(s"num_valid_records = $num_valid_records")

    // compute distances only once and reuse them
    var distances = feature_hashes
      .as("_1")
      .withColumnRenamed("rec_id", "rec_id_from")
      .crossJoin(feature_hashes
        .as("_2")
        .withColumnRenamed("rec_id", "rec_id_to"))
      .persist(StorageLevel.DISK_ONLY)

    feature_sets.foreach(feature_set => {
      distances = distances
        .withColumn("dist",
          LSHUtils.hammingDistUDF(
            col(s"_1.${hash_cols(feature_set)}"),
            col(s"_2.${hash_cols(feature_set)}"),
          )
        )
    })
    distances.persist(StorageLevel.DISK_ONLY)
    distances.printSchema()

    // Generate recommendations
    testRecommendationsGeneration(user_recs_interactions, distances, spark)

    // Stop the underlying SparkContext
    sc.stop
    System.exit(0)
  }

  def testRecommendationsGeneration(user_recs_interactions: Dataset[Row], distances: Dataset[Row], sparkSession: SparkSession): Unit = {
    println("Generatinng recommendations for test users.")
    experiment_years.foreach(year => {
      feature_sets.foreach(feature_set => {

        println("year " + year.toString)
        //      val hyperparameters = CF_selected_hyperparms.CF_selected_hyperparms.getOrElse(year, Map()).asInstanceOf[Map[String, Any]]
        val train_user_ids = sparkSession.read
          .option("header", "true")
          .csv(out_dir + "holdout/year_" + year.toString + "_train.csv")

        val test_user_ids = sparkSession.read
          .option("header", "true")
          .csv(out_dir + "holdout/year_" + year.toString + "_test.csv")

        val train_interactions = preprocessEdges(user_recs_interactions, train_user_ids, year)
          .persist(StorageLevel.DISK_ONLY)
        // various test sets
        Array(1, 2, 3).foreach(set => {
          println("set " + set.toString)

          val test_train_interaction_keys = sparkSession.read
            .option("header", "true")
            .csv(out_dir + "holdout/year_" + year.toString + "_test_train_interactions_set_" + set.toString + ".csv")

          val test_train_interactions = preprocessEdges(
            user_recs_interactions.join(test_train_interaction_keys, Seq("_key"), "inner"),
            test_user_ids, year)
            .persist(StorageLevel.DISK_ONLY)

          val recommendations = recommend(test_train_interactions, train_interactions, distances)


          test_train_interactions.unpersist()
          train_interactions.unpersist()

          recommendations
            .write
            .mode(SaveMode.Overwrite)
            .orc(s"${out_dir}output/year_${year}_ABz_${feature_set}_set_${set}.orc")
        })
      })
    })

  }

  def preprocessEdges(interactions: Dataset[Row], user_ids: Dataset[Row], selected_year: Int): Dataset[Row] = {
    interactions
      .select(user_colname, item_colname, "years.*")
      .withColumnRenamed("yr_" + selected_year.toString, "listens")
      .filter(s"listens >= ${rating_lower_threshold}")
      .join(user_ids, Seq(user_colname), joinType = "inner")
      .select(user_colname, item_colname, "listens")
  }

  def recommend(test_train_interactions: Dataset[Row],
                train_interactions: Dataset[Row],
                distances: Dataset[Row]): Dataset[Row] = {
    // Nearest Neighbour Algorithm
    // get all rec_ids (train + test_train), drop duplicates and join with feature_hashes
//    val feature_hashes_for_set = test_train_interactions
//      .select("rec_id")
//      .union(train_interactions.select("rec_id"))
//      .dropDuplicates()
//      .join(distances, Seq("rec_id"), "inner")
//      .select("rec_id", hash_cols(feature_set))

    // which songs did the test user listen to and how often?
    val test_listens = test_train_interactions
      .select("user_id", "rec_id", "listens")

    // In the training data, how popular was each recording?
    //    root
    //    |-- rec_id: long (nullable = false)
    //    |-- total_train_listens: long (nullable = true)
    //    |-- total_train_users: long (nullable = true)
    val train_rec_listens = train_interactions
      .select(
        col("rec_id"),
        col("listens"),
        lit(1).as("lit_1"))
      .groupBy("rec_id")
      .agg(
        sum("listens").as("total_train_listens"),
        sum("lit_1").as("total_train_users"),
      )
    //    train_rec_listens.printSchema()

    // In the training data, which recordings were listened to together and how much?
    //    root
    //    |-- rec_id_from: long (nullable = false)
    //    |-- rec_id_to: long (nullable = false)
    //    |-- count: long (nullable = false)
    var train_rec_inferred_relation = train_interactions
      .select("user_id", "rec_id")
    train_rec_inferred_relation = train_rec_inferred_relation.as("_1")
      .withColumnRenamed("rec_id", "rec_id_from")
      .join(train_rec_inferred_relation.as("_2")
        .withColumnRenamed("rec_id", "rec_id_to"),
        Seq("user_id"))
      .drop("user_id")
      .groupBy("rec_id_from", "rec_id_to")
      .count()
      .withColumnRenamed("count", "inferred_relation")
    //    train_rec_inferred_relation.printSchema()


    // aggregate together with joins
    //    root
    //    |-- rec_id_from: long (nullable = false)
    //    |-- rec_id_to: long (nullable = false)
    //    |-- user_id: long (nullable = false)
    //    |-- listens: integer (nullable = false)
    //    |-- dist: integer (nullable = false)
    //    |-- total_train_listens: long (nullable = true)
    //    |-- total_train_users: long (nullable = true)
    //    |-- inferred_relation: long (nullable = false)
    var df = test_listens.withColumnRenamed("rec_id", "rec_id_from")
      .join(distances,
        "rec_id_from")
      .join(train_rec_listens.withColumnRenamed("rec_id", "rec_id_from"),
        "rec_id_from")
      .join(train_rec_inferred_relation,
        Seq("rec_id_from", "rec_id_to")
      )
    //    user_rec_interactions.printSchema()

    // use Windowing to group by user, find nearest neighbours and rank by listens.
    val window = Window.partitionBy("user_id")
      .orderBy(
        col("dist").desc,
        col("inferred_relation").desc,
        col("total_train_users").desc,
        col("total_train_listens").desc,
      )

    df = df
      .withColumn("rank", row_number.over(window))
      .filter(s"rank <= $items_to_recommend")
      .withColumnRenamed("rec_id_to", "rec_id")
      .select("user_id", "rec_id", "rank")
    //    user_rec_interactions.printSchema()

    df
  }

}
