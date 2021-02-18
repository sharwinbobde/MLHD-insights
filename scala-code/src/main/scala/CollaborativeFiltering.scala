import breeze.stats.distributions._
import io.circe._
import io.circe.syntax._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Dataset, Row, SaveMode, SparkSession}
import utils.{CF_selected_hyperparms, CollabFilteringUtils, RandomGridGenerator}

import java.io.{BufferedWriter, File, FileWriter}
import scala.collection.mutable.ArrayBuffer

object CollaborativeFiltering {

  val ratingCol = "rating"
  val items_to_recommend = 10
  val experiment_years: Array[Int] = Array(2005, 2008, 2012)

  val rating_lower_threshold = 25

  var out_dir = ""

  val CF_utils: CollabFilteringUtils = new CollabFilteringUtils(
    "user_id",
    "rec_id",
    ratingCol)

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
        .appName("CollabFiltering")
        .getOrCreate()

    val sc = spark.sparkContext

    val arangoDBHandler = new ArangoDBHandler(spark)
    val users = arangoDBHandler.getUsers
    val recs = arangoDBHandler.getRecordings
    val user_recs_interactions = arangoDBHandler.getUserToRecordingEdges

    // CrossValidation for hyperparameter tuning
    hyperparameterTuning(user_recs_interactions, users, recs, spark)

    testRecoomendationsGeneration(user_recs_interactions, users, recs, spark)

    // Stop the underlying SparkContext
    sc.stop
    System.exit(0)
  }

  def testRecoomendationsGeneration(user_recs_interactions: Dataset[Row], users: Dataset[Row], recs: Dataset[Row], sparkSession: SparkSession): Unit = {
    println("Generatinng recommendations for test users.")
    experiment_years.foreach(year => {
      println("year " + year.toString)
      val hyperparameters = CF_selected_hyperparms.CF_selected_hyperparms.getOrElse(year, Map()).asInstanceOf[Map[String, Any]]
      val train_user_ids = sparkSession.read
        .option("header", "true")
        .csv(out_dir + "holdout/year_" + year.toString + "_train.csv")

      val test_user_ids = sparkSession.read
        .option("header", "true")
        .csv(out_dir + "holdout/year_" + year.toString + "_test.csv")

      val train_interactions = CF_utils.preprocessEdges(user_recs_interactions, train_user_ids, year, rating_lower_threshold)
      Array(1,2,3).foreach(set => {
        println("set " + set.toString)

        val test_train_interaction_keys = sparkSession.read
          .option("header", "true")
          .csv(out_dir + "holdout/year_" + year.toString + "_test_train_interactions_set_" + set.toString + ".csv")

        val test_test_interaction_keys = sparkSession.read
          .option("header", "true")
          .csv(out_dir + "holdout/year_" + year.toString + "_test_test_interactions_set_" + set.toString + ".csv")
        val test_train_interactions = CF_utils.preprocessEdges(
          user_recs_interactions.join(test_train_interaction_keys, Seq("_key"), "inner"),
          test_user_ids, year, rating_lower_threshold)

//        val test_test_interactions = preprocessEdges(
//          user_recs_interactions.join(test_test_interaction_keys, Seq("_key"), "inner"),
//          users, recs, year)

        val model = CF_utils.getCFModel(train_interactions.union(test_train_interactions),
          hyperparameters.getOrElse("latentFactors", -1).asInstanceOf[Int],
          hyperparameters.getOrElse("maxItr", -1).asInstanceOf[Int],
          hyperparameters.getOrElse("regularizingParam", -1.0).asInstanceOf[Double],
          hyperparameters.getOrElse("alpha", -1.0).asInstanceOf[Double])

        // Get recommendations :)
        val raw_recommendations = CF_utils.getRecommendations(model, test_user_ids, 100)
        val recommendations = CF_utils.postprocessRecommendations(raw_recommendations)

        recommendations
          .coalesce(1)
          .write
          .mode(SaveMode.Overwrite)
          .option("header", "true")
          .csv(out_dir + "output/year_" + year.toString + "_CF_set_" + set.toString + ".csv")
      })
    })

  }

  def hyperparameterTuning(user_recs_interactions: Dataset[Row], users: Dataset[Row], recs: Dataset[Row], sparkSession: SparkSession): Unit = {
    experiment_years.foreach(year => {
      println("year " + year.toString)
      val randGrid = new RandomGridGenerator(1)
        .addDistr("latentFactors", (5 to 20).toArray)
        .addDistr("maxItr", (5 to 25).toArray)
        .addDistr("regularizingParam", Uniform(0.001, 2))
        .addDistr("alpha", Uniform(0.1, 3))
        .getSamples()

      // TODO Create df with hyper params


      // TODO read and store crossval-folds


      // TODO create udf to find mean_RMSE error and add to df


      // TODO save df csv with hyperparams and mesn RMSE

      var year_hyperparameter_with_error_array = collection.mutable.ArrayBuffer[Json]()

      randGrid.foreach(hyperparameters => {

        val hyperparameter_rmses = ArrayBuffer[Double](5)
        //  for each fold
        (1 to 5).foreach(fold_num => {
          val train_user_ids = sparkSession.read
            .option("header", "true")
            .csv(out_dir + "crossval/year_" + year.toString + "_cv_fold_" + fold_num.toString + "_train.csv")

          val train_interactions = CF_utils.preprocessEdges(user_recs_interactions, train_user_ids, year, rating_lower_threshold)
          val Array(rand_train, rand_validation) = train_interactions.randomSplit(Array(0.7, 0.3), 4242)

          //  Get predictions
          // using getOrElse("param", -1) because need to use match instead, will give error if -1 is selected... workaround :)
          val model = CF_utils.getCFModel(rand_train,
            hyperparameters.getOrElse("latentFactors", -1).asInstanceOf[Int],
            hyperparameters.getOrElse("maxItr", -1).asInstanceOf[Int],
            hyperparameters.getOrElse("regularizingParam", -1.0).asInstanceOf[Double],
            hyperparameters.getOrElse("alpha", -1.0).asInstanceOf[Double])

          val predictions = model.transform(rand_validation)
          hyperparameter_rmses += CF_utils.getRMSE(predictions)
        })
        val mean_rmse = hyperparameter_rmses.sum / hyperparameter_rmses.size

        //        package hyperparameters with error as json
        val json = JsonObject()
          .add("latentFactors", hyperparameters.getOrElse("latentFactors", -1).asInstanceOf[Int].asJson)
          .add("maxItr", hyperparameters.getOrElse("maxItr", -1).asInstanceOf[Int].asJson)
          .add("regularizingParam", hyperparameters.getOrElse("regularizingParam", -1.0).asInstanceOf[Double].asJson)
          .add("alpha", hyperparameters.getOrElse("alpha", -1.0).asInstanceOf[Double].asJson)
          .add("error", mean_rmse.asJson)

        year_hyperparameter_with_error_array += json.asJson
      })

      val bw = new BufferedWriter(new FileWriter(new File(out_dir + "hyperparameter-tuning/CollabFiltering_year_" + year.toString + ".json")))
      bw.write(year_hyperparameter_with_error_array.toArray.asJson.spaces2SortKeys)
      bw.close()
    })
  }
}
