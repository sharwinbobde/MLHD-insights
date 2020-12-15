import com.arangodb.spark.{ArangoSpark, ReadOptions}
import com.arangodb.velocypack.module.scala.VPackScalaModule
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{DataFrame, SaveMode, SparkSession, types}
import  org.apache.spark.sql.functions._

import scala.beans.BeanProperty

object Test {

  case class TestBean(@BeanProperty years:Years){
    def this() = this(null)
  }

  case class Years(@BeanProperty yr_2006: Int,
                   @BeanProperty yr_2007: Int,
                   @BeanProperty yr_2008: Int,
                   @BeanProperty yr_2009: Int,
                   @BeanProperty yr_2010: Int,
                   @BeanProperty yr_2011: Int,
                   @BeanProperty yr_2012: Int,
                   @BeanProperty yr_2013: Int,
                   @BeanProperty yr_2014: Int,
                   @BeanProperty yr_2015: Int,
                   @BeanProperty yr_2016: Int,
                   @BeanProperty yr_2017: Int,
                   @BeanProperty yr_2018: Int,
                   @BeanProperty yr_2019: Int,
                  ){
    def this() = this(0,0,0,0,0,0,0,0,0,0,0,0,0,0)
  }

  def main(args: Array[String]) {
    // Turn off copious logging
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)

    val spark: SparkSession =
      SparkSession
        .builder()
        .config("arangodb.hosts", "plb.sharwinbobde.com:8529") // system ip as docker ip won't be loopback
        .config("arangodb.user", "root")
        .config("arangodb.password", "Happy2Help!")
        .appName("AppName")
        .getOrCreate()

    val sc = spark.sparkContext
    val rdd = ArangoSpark.load[TestBean](sc, "users_to_recordings", ReadOptions("MLHD_processing", collection = "users_to_recordings"))
    val df = spark.createDataFrame(rdd)
    df.printSchema()

    val df1 = df
      .groupBy("years.yr_2008")
      .agg(
        count("years.yr_2008").alias("count"),
        mean("years.yr_2009").alias("next_yr_mean"),
        stddev("years.yr_2009").alias("next_yr_std"),
      )
      .sort("yr_2008")
      .withColumnRenamed("yr_2008", "yr_2008_listens")

//    Save data
    df1.printSchema()
    df1.coalesce(1)
      .write
      .mode(SaveMode.Overwrite)
      .option("header", "true")
      .csv(args(0)+"yr_2008.csv")

    // Stop the underlying SparContext
    sc.stop
  }
}
