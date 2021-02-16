import com.arangodb.spark.{ArangoSpark, WriteOptions}
import io.circe._
import io.circe.syntax._
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{BooleanType, LongType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession, _}
import org.apache.spark.storage.StorageLevel
import utils.PercentileApprox._

import java.io.{BufferedWriter, File, FileWriter}

object SetupExperiments {
  val experiment_years = (2005 to 2012).toArray
  val visualization_years = (2005 to 2014).toArray
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
        .appName("SetupExperiments")
        .getOrCreate()

    val sc = spark.sparkContext

    val arangoDBHandler = new ArangoDBHandler(spark)
    val user_rec_interactions = arangoDBHandler.getUserToRecordingEdges

    addIDsToNodes(arangoDBHandler.getRecordings, arangoDBHandler.getArtists, arangoDBHandler.getRecordings)

    //        from 2006 to 2018
    visualization_years.foreach(i => {
      val year_str = i.toString
      storeStatsForYear(year_str, user_rec_interactions)
    })

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
      .coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(out_dir + "user_listens_per_year.csv")

    // Look at percentile distributions of listens for every year
    percentileUserListensByYear(user_listens_per_year)


    ////        get Subscribed uses by years
    saveSubscribedUsersByYear(user_listens_per_year)

    //         get percentile distributions of listens for every year normalised by the subscribers
    subscriberNormalisedPercentileUserListensByYear(user_listens_per_year, spark)

    itemFrequenciesForMetrics(user_rec_interactions)


    val item_listens_per_year = user_rec_interactions
      .select(col("rec_id"), col("years.*"))
      .groupBy("rec_id")
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

    item_listens_per_year
      .coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(out_dir + "item_listens_per_year.csv")


    // Stop the underlying SparContext
    sc.stop
    System.exit(0)
  }

  def addIDsToNodes(users: Dataset[Row], artists: Dataset[Row], recordings: Dataset[Row]): Unit = {
    val users_with_id = users.withColumn("node_id", monotonically_increasing_id() * 10 + 1)
    val artists_with_id = artists.withColumn("node_id", monotonically_increasing_id() * 10 + 2)
    val recs_with_id = recordings.withColumn("node_id", monotonically_increasing_id() * 10 + 3)

    ArangoSpark.saveDF(users_with_id, "users", WriteOptions(database = "MLHD_processing", method = WriteOptions.UPDATE))
    ArangoSpark.saveDF(artists_with_id, "artists", WriteOptions(database = "MLHD_processing", method = WriteOptions.UPDATE))
    ArangoSpark.saveDF(recs_with_id, "recordings", WriteOptions(database = "MLHD_processing", method = WriteOptions.UPDATE))
  }

  def itemFrequenciesForMetrics(user_rec_interactions: Dataset[Row]): Unit = {
    user_rec_interactions
      .select("rec_id", "years.*")
      .groupBy("rec_id")
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
      .coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(out_dir + "item_frequencies.csv")
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
      val year_subscribers_list = sparkSession.read.option("header", "true")
        .schema(
          new StructType()
            .add("user_id", LongType, nullable = false))
        .csv(out_dir + "year_" + year.toString + "_subscribers.csv")

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
        .coalesce(1)
        .write
        .mode(SaveMode.Overwrite)
        .option("header", "true")
        .csv(out_dir + "year_" + year.toString + "_subscribers.csv")
    })
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

