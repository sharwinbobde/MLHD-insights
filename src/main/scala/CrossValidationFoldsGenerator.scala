import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{Dataset, Row, SaveMode, SparkSession}
import ArangoDBHandler._
import com.arangodb.spark.{ArangoSpark, ReadOptions}
import document_schemas.Node
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, monotonically_increasing_id}
import org.apache.spark.sql.types.{StringType, StructType}
import org.apache.spark.storage.StorageLevel

object CrossValidationFoldsGenerator {

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
        .appName("CrossValidationFoldsGenerator")
        .getOrCreate()

    val sc = spark.sparkContext

    val users = getUsers(sc, spark)
      .select(col("_key").alias("user"))


    (2005 until 2013).foreach(year => {
      val subscribed_users = spark.read
        .option("header", "true")
        .csv(out_dir + "year_" + year.toString + "_subscribers.csv")

      val Array(train, test) = subscribed_users.randomSplit(Array(0.7, 0.3), 424356)
      println("\nyear " + year.toString)
      println("trainSize = " + train.count().toString)
      println("testSize = " + test.count().toString)

      train.coalesce(1)
        .write
        .mode(SaveMode.Overwrite)
        .option("header", "true")
        .csv(out_dir + "train-test/year_" + year.toString + "_train.csv")

      test.coalesce(1)
        .write
        .mode(SaveMode.Overwrite)
        .option("header", "true")
        .csv(out_dir + "train-test/year_" + year.toString + "_test.csv")

      //      Create Simple train and validation folds
      createSimpleTrainValForYear(train, year)

      //      Create Crossvalidation train and validation folds
      val kFolds = MLUtils.kFold(train.rdd, 5, 4242)

      var fold_num = 1
      kFolds.foreach((fold: (RDD[Row], RDD[Row])) => {
        createCrossvalTrainValForYear(fold, fold_num, year, spark)
        fold_num += 1
      })
    })


    // Stop the underlying SparkContext
    sc.stop
  }

  def createCrossvalTrainValForYear(fold: (RDD[Row], RDD[Row]), fold_num: Int, year: Int, spark: SparkSession): Unit = {
    val fold_train = spark.createDataFrame(rowRDD = fold._1, new StructType().add("users", StringType, nullable = false))
    val fold_validation = spark.createDataFrame(rowRDD = fold._2, new StructType().add("users", StringType, nullable = false))

    fold_train.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(out_dir + "crossval/year_" + year.toString + "_cv_fold_" + fold_num.toString + "_train.csv")

    fold_validation.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(out_dir + "crossval/year_" + year.toString + "_cv_fold_" + fold_num.toString + "_validation.csv")

  }

  def createSimpleTrainValForYear(train: Dataset[Row], year: Int): Unit = {
    val Array(fold_train, fold_validation) = train.randomSplit(Array(0.7, 0.3), 424356)

    fold_train.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(out_dir + "simple-train-val/year_" + year.toString + "_train.csv")

    fold_validation.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(out_dir + "simple-train-val/year_" + year.toString + "_validation.csv")

  }

}
