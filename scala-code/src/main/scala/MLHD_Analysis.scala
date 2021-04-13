import com.arangodb.spark.{ArangoSpark, WriteOptions}
import io.circe._
import io.circe.syntax._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{BooleanType, IntegerType, LongType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession, _}
import org.apache.spark.storage.StorageLevel
import utils.PercentileApprox.percentile_approx

import java.io.{BufferedWriter, File, FileWriter}

object MLHD_Analysis {
  val experiment_years: Array[Int] = (2005 to 2012).toArray
  val visualization_years: Array[Int] = (2005 to 2013).toArray
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
        .appName("MLHD_Analysis")
        .getOrCreate()

    val sc = spark.sparkContext

    val arangoDBHandler = new ArangoDBHandler(spark)

    val user_rec_interactions = arangoDBHandler.getUserToRecordingEdges()

    visualization_years.foreach(yr => storeStatsForYear(yr, user_rec_interactions))

    //      Look at user's total listens every year
    val user_listens_per_year = user_rec_interactions
      .select(col("user_id"), col("years.*"))
      .groupBy("user_id")
      .agg(
        sum("yr_2005"),
        sum("yr_2006"),
        sum("yr_2007"),
        sum("yr_2008"),
        sum("yr_2009"),
        sum("yr_2010"),
        sum("yr_2011"),
        sum("yr_2012"),
        sum("yr_2013"),
        sum("yr_2014"),
      )

    user_listens_per_year
      .write
      .mode(SaveMode.Overwrite)
      .orc(out_dir + "user_listens_per_year.orc")

    // Look at percentile distributions of listens for every year
    percentileUserListensByYear(user_listens_per_year)


    ////        get Subscribed uses by years
    saveSubscribedUsersByYear(user_listens_per_year)

    //         get percentile distributions of listens for every year normalised by the subscribers
    subscriberNormalisedPercentileUserListensByYear(user_listens_per_year, spark)

    val item_listens_per_year = user_rec_interactions
      .select(col("rec_id"), col("years.*"))
      .groupBy("rec_id")
      .agg(
        sum("yr_2005").as("sum_2005"),
        sum("yr_2006").as("sum_2006"),
        sum("yr_2007").as("sum_2007"),
        sum("yr_2008").as("sum_2008"),
        sum("yr_2009").as("sum_2009"),
        sum("yr_2010").as("sum_2010"),
        sum("yr_2011").as("sum_2011"),
        sum("yr_2012").as("sum_2012"),
        sum("yr_2013").as("sum_2013"),
      )

    item_listens_per_year
      .write
      .mode(SaveMode.Overwrite)
      .orc(out_dir + "item_listens_per_year.orc")

    // Stop the underlying SparContext
    sc.stop
    System.exit(0)
  }


  def percentileUserListensByYear(user_listens_per_year: DataFrame): Unit = {
    val percentiles_to_extract = Array(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    var m = Map[String, Json]()
    visualization_years.foreach(yr => {
      m += (yr.toString -> user_listens_per_year.stat
        .approxQuantile("sum(yr_" + yr.toString + ")",
          percentiles_to_extract, 0.0).asJson)

    })
    val bw = new BufferedWriter(new FileWriter(new File(out_dir.substring(9) + "user_rec_listen_percentiles_by_year.json")))
    bw.write(m.asJson.spaces2SortKeys)
    bw.close()
  }

  def subscriberNormalisedPercentileUserListensByYear(user_listens_per_year: DataFrame, sparkSession: SparkSession): Unit = {
    val percentiles_to_extract = Array(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    var m = Map[String, Json]()
    experiment_years.foreach(year => {
      val year_subscribers_list = sparkSession.read
        .orc(out_dir + s"subscribed_users/year_${year}_subscribers.orc")

      val percentiles = user_listens_per_year
        .join(year_subscribers_list, Seq("user_id"), "inner")
        .stat.approxQuantile("sum(yr_" + year.toString + ")",
        percentiles_to_extract, 0.0)

      m += (year.toString -> percentiles.asJson)

    })


    val bw = new BufferedWriter(new FileWriter(new File(out_dir.substring(9) + "normalised_user_rec_listen_percentiles_by_year.json")))
    bw.write(m.asJson.spaces2SortKeys)
    bw.close()
  }

  def saveSubscribedUsersByYear(user_listens_per_year: DataFrame): Unit = {
    val subscription_checker = udf((sum_for_user: Long) => {
      sum_for_user > 0
    }, BooleanType)

    experiment_years.foreach(year => {
      user_listens_per_year
        .filter(
          subscription_checker(col("sum(yr_" + year.toString + ")"))
        )
        .select("user_id")
        .persist(StorageLevel.DISK_ONLY)
        .write
        .mode(SaveMode.Overwrite)
        .orc(out_dir + s"subscribed_users/year_${year}_subscribers.orc")

    })
  }


  def storeStatsForYear(year: Int, user_rec_interactions: DataFrame): Unit = {
    val df1 = user_rec_interactions
      .groupBy(s"years.yr_${year}")
      .agg(
        count(s"years.yr_${year}").alias("count")
      )
      .sort(s"yr_${year}")
      .withColumnRenamed(s"yr_${year}", "listens")

    //    Save data
    df1.write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .orc(out_dir + s"listen_count_per_year_frequency/yr_${year}.orc")

    val next_year = year + 1
    val df2 = user_rec_interactions
      .groupBy(s"years.yr_${year}")
      .agg(
        percentile_approx(
          col(s"years.yr_${next_year}"),
          typedLit(Seq(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)))
          .alias("percentiles")
      )
    val df3 = df2
      .select(
        (-1 until 11).map(i => {
          if (i == -1) {
            col(s"yr_${year}")
          } else {
            col("percentiles").getItem(i).as(s"percentile_$i")
          }
        }): _*
      )
    df3.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .orc(out_dir + s"next_year_percentiles/yr_${year}_next_yr_percentiles.orc")
  }
}

