import com.arangodb.spark.{ArangoSpark, ReadOptions}
import document_schemas.{Node, UserToRecordingOrArtistRelation}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.functions.{col, isnan, monotonically_increasing_id, udf, posexplode}
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel
import ArangoDBHandler._

object CollaborativeFiltering {

  val ratingCol = "rating_01"
  val items_to_recommend = 10
  val selected_year: String = "2005"


  def main(args: Array[String]) {
    // Turn off copious logging
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val out_dir = args(0)

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
    val interactions_preprocessed = preprocessEdges(user_recs_interactions, users, recs)
    interactions_preprocessed.printSchema()

    //    Train-Test split
    //    TODO get known crossvalidation sets later
    val Array(train, test) = interactions_preprocessed.randomSplit(Array(0.8, 0.2))


    // Get predictions
    val model = getCFModel(train)
    var predictions = getPredictions(model, test)
    predictions = postprocessPredictions(predictions, users, recs)
    predictions.printSchema()
    predictions.show(50)

    //    Get training error
    val rmse = getRMSE(model, predictions)
    println(s"Root-mean-square error = $rmse")

    ////    Get recommendations
    //    val model_recommend = getCFModel(interactions_preprocessed)
    //    var recommendations = getRecommendations(model_recommend, test, items_to_recommend)
    //    recommendations = postprocessRecommendations(recommendations, users, recs)
    //
    //    recommendations.printSchema()
    //    recommendations.show(50)

    // Stop the underlying SparkContext
    sc.stop

  }

  def preprocessEdges(interactions: Dataset[Row], users: Dataset[Row], recs: Dataset[Row]): Dataset[Row] = {
    val quantizer = udf((x: Int) => {
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
      .withColumnRenamed("yr_" + selected_year, "rating")
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

  def getCFModel(train: Dataset[Row]): ALSModel = {
    //  Collaborative Filtering
    val als = new ALS()
      .setRank(10)
      .setMaxIter(5)
      .setRegParam(1)
      .setAlpha(1)
      //      .setImplicitPrefs(true)
      //      .setNonnegative(true)
      .setUserCol("user_id")
      .setItemCol("rec_id")
      .setRatingCol(ratingCol)
      .setColdStartStrategy("drop")
      .setSeed(69)

    als.fit(train)
  }

  def getRMSE(model: ALSModel, predictions: Dataset[Row]): Double = {
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
