package utils

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.functions.{col, posexplode}
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.storage.StorageLevel

class CollabFilteringUtils(val user_colname: String, val item_colname: String, val rating_colname: String) {


  def getCFModel(train: Dataset[Row], latentFactors: Int, maxItr: Int, regularizingParam: Double, alpha: Double): ALSModel = {
    //  Collaborative Filtering
    val als = new ALS()
      .setRank(latentFactors)
      .setMaxIter(maxItr)
      .setRegParam(regularizingParam)
      .setAlpha(alpha)
      .setImplicitPrefs(true)
      .setNonnegative(true)
      .setUserCol(user_colname)
      .setItemCol(item_colname)
      .setRatingCol(rating_colname)
      .setColdStartStrategy("drop")
      .setSeed(69)

    als.fit(train)
  }

  def getRMSE(predictions: Dataset[Row]): Double = {
    new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol(rating_colname)
      .setPredictionCol("prediction")
      .evaluate(predictions)

  }

  def getRecommendations(model: ALSModel, user_ids: Dataset[Row], num_items: Int): Dataset[Row] = {
    model.recommendForUserSubset(user_ids, num_items)
  }

  def getPredictions(model: ALSModel, test: Dataset[Row]): Dataset[Row] = {
    model.transform(test)
  }

  def postprocessRecommendations(recommendations: Dataset[Row]): Dataset[Row] = {
    recommendations.select(col("user_id"), posexplode(col("recommendations")))
      .select("user_id", "pos", "col.*")
      .withColumn("rank", col("pos") + 1)
      .select("user_id", "rank", "rec_id", "rating")
      .orderBy("user_id", "rank")
  }

  def postprocessPredictions(predictions: Dataset[Row]): Dataset[Row] = {
    predictions
      .orderBy("user_id", "rec_id")
  }

  def preprocessEdges(interactions: Dataset[Row], user_ids: Dataset[Row], selected_year: Int, rating_lower_threshold: Int): Dataset[Row] = {
    interactions
      .select(user_colname, item_colname, "years.*")
      .withColumnRenamed("yr_" + selected_year.toString, rating_colname)
      .filter(rating_colname + " > " + rating_lower_threshold.toString)
      .join(user_ids, Seq(user_colname), joinType = "inner")
      .select(user_colname, item_colname, rating_colname)
      .persist(StorageLevel.DISK_ONLY)
  }
}
