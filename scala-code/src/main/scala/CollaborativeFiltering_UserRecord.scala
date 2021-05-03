import DatasetDivision.out_dir
import breeze.stats.distributions._
import io.circe._
import io.circe.syntax._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SaveMode, SparkSession}
import utils.{CF_selected_hyperparms, CollabFilteringUtils, RandomGridGenerator}

import java.io.{BufferedWriter, File, FileWriter}
import scala.collection.mutable.ArrayBuffer

object CollaborativeFiltering_UserRecord {

  val ratingCol = "rating"
  val items_to_recommend = 100
  val experiment_years: Array[Int] = Array(2005, 2008, 2012)

  val rating_lower_threshold = 25
  val CF_utils: CollabFilteringUtils = new CollabFilteringUtils(
    "user_id",
    "rec_id",
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
        .appName("CollabFiltering User-Record")
        .getOrCreate()

    val sc = spark.sparkContext

    val arangoDBHandler = new ArangoDBHandler(spark)
    val user_recs_interactions = arangoDBHandler.getUserToRecordingEdges()

    // CrossValidation for hyperparameter tuning
//    hyperparameterTuning(user_rec_interactions, spark)

    testRecommendationsGeneration(user_recs_interactions, spark)

    // Stop the underlying SparkContext
    sc.stop
    System.exit(0)
  }

  def hyperparameterTuning(user_recs_interactions: Dataset[Row], sparkSession: SparkSession): Unit = {
    experiment_years.foreach(year => {
      println("year " + year.toString)
      val randGrid = new RandomGridGenerator(RandomGrid_samples)
        .addDistr("latentFactors", (5 to 30).toArray)
        .addDistr("maxItr", (3 to 25).toArray)
        .addDistr("regularizingParam", Uniform(0.001, 2))
        .addDistr("alpha", Uniform(0.1, 3))
        .getSamples()
      var year_hyperparameter_with_error_array = collection.mutable.ArrayBuffer[Json]()

      randGrid.foreach(hyperparameters => {
        val train_user_ids = sparkSession.read
          .orc(out_dir + s"holdout/users/year-${year}-train.orc")

        val train_interactions = CF_utils.preprocessEdges(user_recs_interactions, train_user_ids, year, rating_lower_threshold)
        val Array(rand_train, rand_validation) = train_interactions.randomSplit(Array(0.7, 0.3), 4242)

        //  Get predictions
        // using getOrElse("param", -1) because need to use match instead, will give error if -1 is selected... workaround :)
        val model = CF_utils.getCFModel(rand_train,
          hyperparameters.getOrElse("latentFactors", -1).asInstanceOf[Int],
          hyperparameters.getOrElse("maxItr", -1).asInstanceOf[Int],
          hyperparameters.getOrElse("regularizingParam", -1.0).asInstanceOf[Double],
          hyperparameters.getOrElse("alpha", -1.0).asInstanceOf[Double],
          num_blocks =hyperparameters.getOrElse("num_blocks", 10).asInstanceOf[Int])

        val predictions = model.transform(rand_validation)
        val rmse = CF_utils.getRMSE(predictions)

        //        package hyperparameters with error as json
        val json = JsonObject()
          .add("latentFactors", hyperparameters.getOrElse("latentFactors", -1).asInstanceOf[Int].asJson)
          .add("maxItr", hyperparameters.getOrElse("maxItr", -1).asInstanceOf[Int].asJson)
          .add("regularizingParam", hyperparameters.getOrElse("regularizingParam", -1.0).asInstanceOf[Double].asJson)
          .add("alpha", hyperparameters.getOrElse("alpha", -1.0).asInstanceOf[Double].asJson)
          .add("error", rmse.asJson)

        year_hyperparameter_with_error_array += json.asJson
        val bw = new BufferedWriter(
          new FileWriter(
            new File(out_dir.substring(9) + s"hyperparameter-tuning/CF-user_rec-year_${year}.json")
          )
        )
        bw.write(year_hyperparameter_with_error_array.toArray.asJson.spaces2SortKeys)
        bw.close()
      })

    })
  }

  def testRecommendationsGeneration(user_recs_interactions: Dataset[Row], sparkSession: SparkSession): Unit = {
    println("Generatinng recommendations for test users.")
    experiment_years.foreach(year => {
      println("year " + year.toString)
      val hyperparameters = CF_selected_hyperparms.CF_selected_hyperparms.getOrElse(year, Map()).asInstanceOf[Map[String, Any]]
      val train_user_ids = sparkSession.read
        .orc(out_dir + s"holdout/users/year-${year}-train.orc")

      val train_interactions = CF_utils.preprocessEdges(user_recs_interactions, train_user_ids, year, rating_lower_threshold)

      testRecommendationsForOneYear("RS", year,
        user_recs_interactions, train_interactions,
        hyperparameters, sparkSession)

      testRecommendationsForOneYear("EA", year,
        user_recs_interactions, train_interactions,
        hyperparameters, sparkSession)
    })

  }

  def testRecommendationsForOneYear(RS_or_EA: String,
                                    year: Int,
                                    user_recs_interactions: DataFrame,
                                    train_interactions: DataFrame,
                                    hyperparameters: Map[String, Any],
                                    sparkSession: SparkSession): Unit = {

    val test_user_ids = sparkSession.read
      .orc(out_dir + s"holdout/users/year-${year}-test_${RS_or_EA}.orc")
    Array(1, 2, 3).foreach(set => {
      println(s"set $set")

      val test_train_interaction_keys = sparkSession.read
        .orc(out_dir + s"holdout/interactions/year_${year}-test_${RS_or_EA}-train_interactions-set_${set}.orc")

      val test_train_interactions = CF_utils.preprocessEdges(
        user_recs_interactions.join(test_train_interaction_keys, Seq("_key"), "inner"),
        test_user_ids, year, rating_lower_threshold)
//      test_train_interactions.printSchema()
//      println(s"test_train_interactions size = ${test_train_interactions.count()}")

      val model = CF_utils.getCFModel(train_interactions.union(test_train_interactions),
        hyperparameters.getOrElse("latentFactors", -1).asInstanceOf[Int],
        hyperparameters.getOrElse("maxItr", -1).asInstanceOf[Int],
        hyperparameters.getOrElse("regularizingParam", -1.0).asInstanceOf[Double],
        hyperparameters.getOrElse("alpha", -1.0).asInstanceOf[Double],
        num_blocks = hyperparameters.getOrElse("num_blocks", 10).asInstanceOf[Int])

      // Get recommendations :)
      val raw_recommendations = CF_utils.getRecommendations(model, test_user_ids, items_to_recommend)
      val recommendations = CF_utils.postprocessRecommendations(raw_recommendations)

      recommendations
        .coalesce(1)
        .write
        .mode(SaveMode.Overwrite)
        .orc(out_dir + s"output-${RS_or_EA}/CF-user_rec/year_${year}-CF-user_rec-set_${set}.orc")
    })
  }
}
