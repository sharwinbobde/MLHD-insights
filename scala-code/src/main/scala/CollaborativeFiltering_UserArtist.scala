import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{rand, row_number}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SaveMode, SparkSession}
import utils.{CF_selected_hyperparms, CollabFilteringUtils}

object CollaborativeFiltering_UserArtist {

  val ratingCol = "rating"
  val items_to_recommend = 100
  val experiment_years: Array[Int] = Array(2005, 2008, 2012)
  val sample_items_per_artist = 10

  val rating_lower_threshold = 25
  val CF_utils: CollabFilteringUtils = new CollabFilteringUtils(
    "user_id",
    "artist_id",
    ratingCol)
  val RandomGrid_samples = 10
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
        .appName("CollabFiltering User-Artist")
        .getOrCreate()

    val sc = spark.sparkContext

    val arangoDBHandler = new ArangoDBHandler(spark)
    val user_artist_interactions = arangoDBHandler.getUserToArtistEdges()
    val user_rec_interactions = arangoDBHandler.getUserToRecordingEdges()
    val artist_rec_interactions = arangoDBHandler.getArtistToRecordingEdges()


    // CrossValidation for hyperparameter tuning
    //    hyperparameterTuning(user_recs_interactions, users, recs, spark)

    testRecommendationsGeneration(
      user_artist_interactions,
      user_rec_interactions,
      artist_rec_interactions,
      spark)

    // Stop the underlying SparkContext
    sc.stop
    System.exit(0)
  }

  def testRecommendationsGeneration(user_artist_interactions: Dataset[Row],
                                    user_rec_interactions: Dataset[Row],
                                    artist_rec_interactions: Dataset[Row],
                                    sparkSession: SparkSession): Unit = {
    println("Generatinng recommendations for test users.")
    experiment_years.foreach(year => {
      println("year " + year.toString)
      val hyperparameters = CF_selected_hyperparms.CF_selected_hyperparms.getOrElse(year, Map()).asInstanceOf[Map[String, Any]]
      val train_user_ids = sparkSession.read
        .orc(out_dir + s"holdout/users/year-${year}-train.orc")

      val train_interactions = CF_utils.preprocessEdges(
        user_artist_interactions, train_user_ids, year, rating_lower_threshold)

      testRecommendationsForOneYear("RS", year,
        user_rec_interactions, user_artist_interactions, artist_rec_interactions,
        train_interactions,
        hyperparameters, sparkSession)

      testRecommendationsForOneYear("EA", year,
        user_rec_interactions, user_artist_interactions, artist_rec_interactions,
        train_interactions,
        hyperparameters, sparkSession)

    })

  }

  def testRecommendationsForOneYear(RS_or_EA: String,
                                    year: Int,
                                    user_rec_interactions: DataFrame,
                                    user_artist_interactions: DataFrame,
                                    artist_rec_interactions: DataFrame,
                                    train_interactions: DataFrame,
                                    hyperparameters: Map[String, Any],
                                    sparkSession: SparkSession): Unit = {

    val test_user_ids = sparkSession.read
      .orc(out_dir + s"holdout/users/year-${year}-test_${RS_or_EA}.orc")
    Array(1, 2, 3).foreach(set => {
      println("set " + set.toString)

      val test_train_user_rec_interaction_keys = sparkSession.read
        .orc(out_dir + s"holdout/interactions/year_${year}-test_RS-train_interactions-set_${set}.orc")


      //  filter user-artist interactions which are indicated by user-item interactions.
      //  use artist-recording edges for the same.
      val test_train_user_artist_ids = user_rec_interactions
        .select("user_id", "rec_id", "_key")
        .join(test_train_user_rec_interaction_keys, Seq("_key"), "inner")
        .select("user_id", "rec_id")
        .join(artist_rec_interactions.select("artist_id", "rec_id"), Seq("rec_id"), "inner")
        .select("user_id", "artist_id")

      val test_train_interactions = CF_utils.preprocessEdges(
        user_artist_interactions.join(
          test_train_user_artist_ids,
          Seq("user_id", "artist_id"), "inner"),
        test_user_ids, year, rating_lower_threshold)

      val model = CF_utils.getCFModel(train_interactions.union(test_train_interactions),
        hyperparameters.getOrElse("latentFactors", -1).asInstanceOf[Int],
        hyperparameters.getOrElse("maxItr", -1).asInstanceOf[Int],
        hyperparameters.getOrElse("regularizingParam", -1.0).asInstanceOf[Double],
        hyperparameters.getOrElse("alpha", -1.0).asInstanceOf[Double])

      // Get recommendations :)
      val raw_recommendations = CF_utils.getRecommendations(model, test_user_ids, items_to_recommend)
      val recommendations = CF_utils.postprocessRecommendations(raw_recommendations)

      postprocessExpansionArtistToRecordings(recommendations, artist_rec_interactions, user_rec_interactions, year)
        .write
        .mode(SaveMode.Overwrite)
        .orc(out_dir + s"output-${RS_or_EA}/CF-user_artist/year_${year}-CF-user_artist-set_${set}.orc")
    })
  }

  def postprocessExpansionArtistToRecordings(recommendations: Dataset[Row],
                                             artist_rec_interactions: Dataset[Row],
                                             user_rec_interactions: Dataset[Row],
                                             year: Int): Dataset[Row] = {
    // for every user_id artist_id pair select only 10 recordings at most
    // join recommendations and artist_rec_interactions by artist_id

    var df = recommendations
      .join(
        artist_rec_interactions.select("artist_id", "rec_id")
          .dropDuplicates(),
        Seq("artist_id"), "inner")
      // filter to keep only those records which exist at the time of recommendation
      .join(
        user_rec_interactions
          .select("rec_id", "years.*")
          .filter("yr_" + year.toString + " > 0")
          .select("rec_id"),
        Seq("rec_id"), "inner")

    // subset items per user-artist pair in a random manner
    val win_1 = Window.partitionBy("user_id", "artist_id").orderBy(rand())
    df = df
      .withColumn("row_1", row_number.over(win_1))
      .filter("row_1 <= " + sample_items_per_artist.toString)
      // drop duplicate entries for user_id and rec_id pairs
      .dropDuplicates(Seq("user_id", "rec_id"))
      .select("user_id", "rec_id", "row_1")
      .orderBy("user_id")

    // select first 100 for each user_id
    val win_2 = Window.partitionBy("user_id").orderBy("row_1")
    df = df
      .withColumn("row_2", row_number.over(win_2))
      .filter("row_2 <= " + items_to_recommend)
      .withColumnRenamed("row_2", "rank")
      .select("user_id", "rank", "rec_id")

    df
  }

  def hyperparameterTuning(): Unit = {
    // TODO copy progress from CollaborativeFiltering_UserRecord.hyperparameterTuning() after completing those TODOs
  }
}
