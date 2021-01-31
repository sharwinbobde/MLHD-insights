import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.functions.{col, isnan, max, monotonically_increasing_id, posexplode, udf}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel
import ArangoDBHandler._
import utils.RandomGridGenerator
import breeze.stats.distributions._

import java.io.{BufferedWriter, File, FileWriter}
import scala.collection.mutable.ArrayBuffer
import io.circe._
import io.circe.generic.auto._
import io.circe.parser._
import io.circe.syntax._

object CollaborativeFiltering {

  val ratingCol = "rating_01"
  val items_to_recommend = 10
  val selected_year = "2006"


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
        .appName("CollabFiltering")
        .getOrCreate()

    val sc = spark.sparkContext

    val users = getUsers(sc, spark)
    val recs = getRecordings(sc, spark)
    val user_recs_interactions = getUserToRecordingEdges(sc, spark)

    //    join dataframes to make users and recs numeric
    //     Used for train-test
    //    val interactions_preprocessed = preprocessEdges(user_recs_interactions, users, recs)
    //    interactions_preprocessed.printSchema()

    //    Crossvalidation for hyperparameter tuning
    hyperparameterTuning(user_recs_interactions, users, recs, spark)



    //    val Array(train, test) = interactions_preprocessed.randomSplit(Array(0.8, 0.2))

    //    // Get predictions
    //    val model = getCFModel(train)
    //    var predictions = getPredictions(model, test)
    //    predictions = postprocessPredictions(predictions, users, recs)
    //    predictions.printSchema()
    //    predictions.show(50)
    //
    //    //    Get training error
    //    val rmse = getRMSE(model, predictions)
    //    println(s"Root-mean-square error = $rmse")


    ////    Get recommendations
    //    val model_recommend = getCFModel(interactions_preprocessed)
    //    var recommendations = getRecommendations(model_recommend, test, items_to_recommend)
    //    recommendations = postprocessRecommendations(recommendations, users, recs)
    //
    //    recommendations.printSchema()
    //    recommendations.show(50)

    // Stop the underlying SparkContext
    sc.stop
    System.exit(0)
  }

  def testSetActions(): Unit = {
    // TODO
  }

  def hyperparameterTuning(user_recs_interactions: Dataset[Row], users: Dataset[Row], recs: Dataset[Row], sparkSession: SparkSession): Unit = {
    (2005 until 2007).foreach(year => {
      println("year " + year.toString)
      val randGrid = new RandomGridGenerator(1)
        .addDistr("latentFactors", (5 to 20).toArray)
        .addDistr("maxItr", (5 to 25).toArray)
        .addDistr("regularizingParam", Uniform(0.001, 2))
        .addDistr("alpha", Uniform(1, 3))
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
          val train_user_uuids = sparkSession.read
            .option("header", "true")
            .csv(out_dir + "crossval/year_" + year.toString + "_cv_fold_" + fold_num.toString + "_train.csv")

          val train_users = preprocessFolds(train_user_uuids, users)
          val train_interactions = preprocessEdges(user_recs_interactions, train_users, recs, year)
          val Array(rand_train, rand_validation) = train_interactions.randomSplit(Array(0.7, 0.3), 4242)

          //  Get predictions
          // using getOrElse("param", -1) because need to use match instead, will give error if -1 is selected... workaround :)
          val model = getCFModel(rand_train,
            hyperparameters.getOrElse("latentFactors", -1).asInstanceOf[Int],
            hyperparameters.getOrElse("maxItr", -1).asInstanceOf[Int],
            hyperparameters.getOrElse("regularizingParam", -1.0).asInstanceOf[Double],
            hyperparameters.getOrElse("alpha", -1.0).asInstanceOf[Double])

          val predictions = model.transform(rand_validation)
          hyperparameter_rmses += getRMSE(predictions)
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

      val bw = new BufferedWriter(new FileWriter(new File("out_data/hyperparameter-tuning/CollabFiltering_year_" + year.toString + ".json")))
      bw.write(year_hyperparameter_with_error_array.toArray.asJson.spaces2SortKeys)
      bw.close()
    })
  }

  def preprocessEdges(interactions: Dataset[Row], users: Dataset[Row], recs: Dataset[Row], selected_year: Int): Dataset[Row] = {
    val quantizer = udf((x: Int) => {
      //      normalise according to problem
      if (x <= 30) {
        0
      } else if (x >= 256) {
        1
      }
      else {
        (x - 30) / (256 - 30)
      }
    }, IntegerType)

    interactions
      .select("_from", "_to", "years.*")
      .withColumnRenamed("yr_" + selected_year.toString, "rating")
      .select("_from", "_to", "rating")
      .filter("rating > 30")
      .persist(StorageLevel.DISK_ONLY)
      .join(users.select("_id", "user_id")
        .withColumnRenamed("_id", "_from"),
        Seq("_from"), joinType = "inner")
      .join(recs.select("_id", "rec_id")
        withColumnRenamed("_id", "_to"),
        Seq("_to"), joinType = "inner")
      .select("user_id", "rec_id", "rating")
      .withColumn("rating_01", quantizer(col("rating")))
  }

  def preprocessFolds(fold_users_uuids: Dataset[Row], all_users: Dataset[Row]): Dataset[Row] = {
    //    append "users/" to all users in the fold to make same as _id
    val appender = udf((s: String) => "users/" + s)
    val fold_ids = fold_users_uuids.withColumnRenamed("users", "_id")
      .withColumn("_id", appender(col("_id")))
    all_users.join(fold_ids, Seq("_id"), "left_semi")
  }

  def getCFModel(train: Dataset[Row], latentFactors: Int, maxItr: Int, regularizingParam: Double, alpha: Double): ALSModel = {
    //  Collaborative Filtering
    val als = new ALS()
      .setRank(latentFactors)
      .setMaxIter(maxItr)
      .setRegParam(regularizingParam)
      .setAlpha(alpha)
      .setImplicitPrefs(true)
      //      .setNonnegative(true)
      .setUserCol("user_id")
      .setItemCol("rec_id")
      .setRatingCol(ratingCol)
      .setColdStartStrategy("drop")
      .setSeed(69)

    als.fit(train)
  }

  def getRMSE(predictions: Dataset[Row]): Double = {
    new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol(ratingCol)
      .setPredictionCol("prediction")
      .evaluate(predictions)

  }

  def getRecommendations(model: ALSModel, users: Dataset[Row], num_items: Int): Dataset[Row] = {
    model.recommendForUserSubset(users.select("user_id"), num_items)
  }

  def getPredictions(model: ALSModel, test: Dataset[Row]): Dataset[Row] = {
    model.transform(test)
  }

  def postprocessRecommendations(recommendations: Dataset[Row], users: Dataset[Row], recs: Dataset[Row]): Dataset[Row] = {
    recommendations.select(col("user_id"), posexplode(col("recommendations")))
      .select("user_id", "pos", "col.*")
      .withColumn("rank", col("pos") + 1)
      .join(recs.select(col("rec_id"), col("_key").alias("item")), "rec_id")
      .join(users.select(col("user_id"), col("_key").alias("user")), "user_id")
      .select("user", "rank", "item", "rating")
      .orderBy("user", "rank")
  }

  def postprocessPredictions(predictions: Dataset[Row], users: Dataset[Row], recs: Dataset[Row]): Dataset[Row] = {
    predictions
      .orderBy("user_id", "rec_id")
  }
}
