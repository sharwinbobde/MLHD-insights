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
      .select(col("_key").alias("user"))

    var Array(train, test) = users.randomSplit(Array(0.8, 0.2), 424356)
    println("\ntrainSize = " + train.count().toString)
    println("\ntestSize = " + test.count().toString)
    test.printSchema()
    test.show(30)


    train.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(out_dir + "train.csv")

    test.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(out_dir + "test.csv")


    val kFolds = MLUtils.kFold(train.rdd, 7, 4242)

    var fold_num = 1
    kFolds.foreach((fold: (RDD[Row],RDD[Row]))=>{
      val fold_train = spark.createDataFrame(rowRDD = fold._1, new StructType().add("user", StringType, nullable = false))
      val fold_validation = spark.createDataFrame(rowRDD = fold._2, new StructType().add("user", StringType, nullable = false))
//      println("\nfold = " + fold_num.toString)
//      println("trainSize = " + fold_train.count().toString)
//      println("valSize = " + fold_validation.count().toString)

      fold_train.coalesce(1)
        .write
        .mode(SaveMode.Overwrite)
        .option("header", "true")
        .csv(out_dir + "cv_fold_" + fold_num.toString + "_train.csv")

      fold_validation.coalesce(1)
        .write
        .mode(SaveMode.Overwrite)
        .option("header", "true")
        .csv(out_dir + "cv_fold_" + fold_num.toString + "_validation.csv")

      fold_num += 1
    })


    // Stop the underlying SparkContext
    sc.stop
  }

}
