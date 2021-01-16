import com.arangodb.spark.{ArangoSpark, ReadOptions}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession, types}
import org.apache.spark.sql.functions._
import org.apache.spark.sql._
import utils.PercentileApprox._
import document_schemas.UserToRecordingOrArtistRelation
import ArangoDBHandler._
import org.apache.spark.storage.StorageLevel
import io.circe._, io.circe.generic.auto._, io.circe.parser._, io.circe.syntax._

import java.io.{BufferedWriter, File, FileWriter}
import scala.util.parsing.json.JSONObject

object ListeningDistributions {
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
        .appName("ListeningDistributions")
        .getOrCreate()

    val sc = spark.sparkContext
    val user_rec_interactions = getUserToRecordingEdges(sc, spark)
    user_rec_interactions.printSchema()

    //    from 2006 to 2018
    //    (2006 until 2007).foreach(i => {
    //      val year_str = i.toString
    //      storeStatsForYear(year_str, user_rec_interactions, out_dir)
    //    })

    //  Look at user's total listens every year
    //    user_rec_interactions
    //      .select(col("_from").alias("user"), col("years.*"))
    //      .groupBy("user")
    //      .agg(
    //        sum("yr_2005"),
    //        sum("yr_2006"),
    //        sum("yr_2007"),
    //        sum("yr_2008"),
    //        sum("yr_2009"),
    //        sum("yr_2010"),
    //        sum("yr_2011"),
    //        sum("yr_2012"),
    //        sum("yr_2013"),
    //        sum("yr_2014"),
    //      )
    //      .coalesce(1)
    //      .write
    //      .mode(SaveMode.Overwrite)
    //      .option("header", "true")
    //      .csv(out_dir + "user_listens_per_year.csv")


    // Look at percentile distributions of listens for every year
    percentileUserListensByYear(user_rec_interactions)


    // Stop the underlying SparContext
    sc.stop
  }

  def percentileUserListensByYear(user_rec_interactions: Dataset[Row]): Unit = {
    val percentiles_to_extract = Array(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    val df = user_rec_interactions
      .select(col("_from").alias("user"), col("years.*"))
      .groupBy("user")
      .agg(
        sum(col("yr_2005")),
        sum(col("yr_2006")),
        sum(col("yr_2007")),
        sum(col("yr_2008")),
        sum(col("yr_2009")),
        sum(col("yr_2010")),
        sum(col("yr_2011")),
        sum(col("yr_2012")),
        sum(col("yr_2013")),
        sum(col("yr_2014")),
      )
      .cache()
    var m = Map[String, Json]()
    (2005 until 2015).foreach(yr => {
      m += (yr.toString -> df.stat.
        approxQuantile("sum(yr_" + yr.toString + ")",
          percentiles_to_extract, 0.0).asJson)

    })

    val file = new File("out_data/user_rec_listen_percentiles_by_year.json")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(m.asJson.spaces2SortKeys)
    bw.close()
  }


  def storeStatsForYear(year_str: String, df: DataFrame): Unit = {
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

