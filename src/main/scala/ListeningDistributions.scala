import com.arangodb.spark.{ArangoSpark, ReadOptions}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession, types}
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import utils.PercentileApprox._
import document_schemas.UserToRecordingOrArtistRelation


object ListeningDistributions {


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
        .appName("AppName")
        .getOrCreate()

    val sc = spark.sparkContext
    val rdd = ArangoSpark.load[UserToRecordingOrArtistRelation](sc, "users_to_recordings", ReadOptions("MLHD_processing", collection = "users_to_recordings"))

    val schema = new UserToRecordingOrArtistRelation().getSchema

    val rows: RDD[Row] = rdd.map(x => x.getAsRow)
    val df = spark.createDataFrame(rowRDD = rows, schema)
    df.printSchema()

    //    from 2006 to 2018
    (2006 until 2007).foreach(i => {
      val year_str = i.toString
      storeStatsForYear(year_str, df, out_dir)
    })

    // Stop the underlying SparContext
    sc.stop
  }


  def storeStatsForYear(year_str: String, df: DataFrame, out_dir: String): Unit = {
    val next_year_str = (Integer.valueOf(year_str) + 1).toString
    val df1 = df
      .groupBy("years.yr_" + year_str)
      .agg(
        count("years.yr_" + year_str).alias("count")
      )
      .sort("yr_" + year_str)
      .withColumnRenamed("yr_" + year_str, "listens")

    //    Save data
    df1.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(out_dir + "yr_" + year_str + ".csv")

    val df2 = df.groupBy("years.yr_" + year_str)
      .agg(
        percentile_approx(col("years.yr_" + next_year_str), typedLit(Seq(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))).alias("percentiles")
      )
    val df3 = df2
      .select(
        (-1 until 11).map(i => {
          if (i == -1) {
            col("yr_" + year_str)
          } else {
            col("percentiles").getItem(i).as(s"percentile_$i")
          }
        }): _*
      )
    //        .withColumn("listens", df2.col("yr_2008"))
    df3.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(out_dir + "yr_" + year_str + "_next_yr_percentiles.csv")
  }
}

