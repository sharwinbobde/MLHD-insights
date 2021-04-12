import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{LongType, StringType, StructType}
import org.apache.spark.storage.StorageLevel

import scala.collection.{breakOut, mutable}

object DatasetDivision {
  val experiment_years: Array[Int] = (2005 to 2012).toArray
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
        .appName("DatasetDivision")
        .getOrCreate()

    val sc = spark.sparkContext

    val arangoDBHandler = new ArangoDBHandler(spark)
    val users = arangoDBHandler.getUsers()

    val user_rec_interactions = arangoDBHandler.getUserToRecordingEdges()


    experiment_years.foreach(year => {
      val subscribed_users = spark.read
        .option("header", "true")
        .schema(
          new StructType()
            .add("user_id", LongType, nullable = false)
        )
        .csv(out_dir + "year_" + year.toString + "_subscribers.csv")

      val Array(train, test_RS, test_EA) = subscribed_users.randomSplit(Array(0.6, 0.2, 0.2), 424356)
      println("\nyear " + year.toString)
      println("train size = " + train.count().toString)
      println("test_RS size= " + test_RS.count().toString)
      println("test_EA size= " + test_EA.count().toString)

      train
        .write
        .mode(SaveMode.Overwrite)
        .orc(out_dir + s"holdout/users/year-${year}-train.orc")
      test_RS
        .write
        .mode(SaveMode.Overwrite)
        .orc(out_dir + s"holdout/users/year-${year}-test_RS.orc")
      test_EA
        .write
        .mode(SaveMode.Overwrite)
        .orc(out_dir + s"holdout/users/year-${year}-test_EA.orc")

//      //      Create Simple train and validation folds
//      createSimpleTrainValForYear(train, year)
//
//      //      Create Crossvalidation train and validation folds
//      val kFolds = MLUtils.kFold(train.rdd, 5, 4242)
//
//      var fold_num = 1
//      kFolds.foreach((fold: (RDD[Row], RDD[Row])) => {
//        createCrossvalTrainValForYear(fold, fold_num, year, spark)
//        fold_num += 1
//      })

      // Save 3 sets of test *interactions* for each test test which are randomly sampled for each user.
      create3SetsOfInteractions(test_RS, "test_RS", year, user_rec_interactions)
      create3SetsOfInteractions(test_EA, "test_EA", year, user_rec_interactions)
    })


    // Stop the underlying SparkContext
    sc.stop
    System.exit(0)
  }

  def create3SetsOfInteractions(df: DataFrame, name: String, year:Int, user_rec_interactions: DataFrame):Unit={
    val test_interactions = user_rec_interactions
      .select("user_id", "_key")
      .join(df,
        Seq("user_id"),
        "inner")

    (1 to 3).foreach(set => {
    val (test_train, test_test) = splitTestInteractionsUniformlyAcrossUsers(test_interactions)

    test_train
      .write
      .mode(SaveMode.Overwrite)
      .orc(out_dir + s"holdout/interactions/year_${year}-${name}-train_interactions-set_${set}.orc")

    test_test
      .write
      .mode(SaveMode.Overwrite)
      .orc(out_dir + s"holdout/interactions/year_${year}-${name}_test-interactions-set_${set}.orc")

    val test_test_user_item = user_rec_interactions
      .join(test_test, Seq("_key"), "inner")
      .select("user_id", "rec_id")

    test_test_user_item
      .write
      .mode(SaveMode.Overwrite)
      .orc(out_dir + s"holdout/interactions/year_${year}-${name}_test-interactions-user_item-set_${set}.orc")
    })
  }

  def splitTestInteractionsUniformlyAcrossUsers(test_interactions: DataFrame): (DataFrame, DataFrame) = {
    val splitting_udf = udf((keys: mutable.WrappedArray[String]) => {
      val (train_indexes, test_indexes) = scala.util.Random.shuffle(keys.indices.toList).splitAt((keys.length / 2).ceil.toInt)
      Array(train_indexes.map(keys)(breakOut), test_indexes.map(keys)(breakOut))
    })

    val test_data_train_test_split = test_interactions
      .select("user_id", "_key")
      .groupBy("user_id")
      .agg(collect_list("_key").alias("keys"))
      .withColumn("split", splitting_udf(col("keys")))
      .withColumn("train", element_at(col("split"), 1))
      .withColumn("test", element_at(col("split"), 2))
      .select("train", "test")
      .persist(StorageLevel.DISK_ONLY)

    val train = test_data_train_test_split
      .select(explode(col("train")).as("_key"))

    val test = test_data_train_test_split
      .select(explode(col("test")).as("_key"))

    test_data_train_test_split.unpersist()

    (train, test)
  }

//  def createCrossvalTrainValForYear(fold: (RDD[Row], RDD[Row]), fold_num: Int, year: Int, spark: SparkSession): Unit = {
//    val fold_train = spark.createDataFrame(rowRDD = fold._1, new StructType().add("user_id", LongType, nullable = false))
//    val fold_validation = spark.createDataFrame(rowRDD = fold._2, new StructType().add("user_id", LongType, nullable = false))
//
//    fold_train.coalesce(1)
//      .write
//      .mode(SaveMode.Overwrite)
//      .option("header", "true")
//      .csv(out_dir + "crossval/year_" + year.toString + "_cv_fold_" + fold_num.toString + "_train.csv")
//
//    fold_validation.coalesce(1)
//      .write
//      .mode(SaveMode.Overwrite)
//      .option("header", "true")
//      .csv(out_dir + "crossval/year_" + year.toString + "_cv_fold_" + fold_num.toString + "_validation.csv")
//
//  }

//  def createSimpleTrainValForYear(train: Dataset[Row], year: Int): Unit = {
//    val Array(fold_train, fold_validation) = train.randomSplit(Array(0.7, 0.3), 424356)
//
//    fold_train.coalesce(1)
//      .write
//      .mode(SaveMode.Overwrite)
//      .option("header", "true")
//      .csv(out_dir + "simple-train-val/year_" + year.toString + "_train.csv")
//
//    fold_validation.coalesce(1)
//      .write
//      .mode(SaveMode.Overwrite)
//      .option("header", "true")
//      .csv(out_dir + "simple-train-val/year_" + year.toString + "_validation.csv")
//
//  }

}
