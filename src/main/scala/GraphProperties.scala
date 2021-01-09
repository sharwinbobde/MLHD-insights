import com.arangodb.spark.{ArangoSpark, ReadOptions}
import document_schemas.{Node, UserToRecordingOrArtistRelation}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions.{col, monotonically_increasing_id, sum}
import org.apache.spark.storage.StorageLevel
import ArangoDBHandler._

import scala.util.parsing.json._
import java.io.{BufferedWriter, File, FileWriter}

object GraphProperties {
  val max_parts = 144

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
        .appName("GraphProperties")
        .getOrCreate()

    val sc = spark.sparkContext

    val users = getUsers(sc, spark)
    val recs = getRecordings(sc, spark)
    val artists = getArtists(sc, spark)
    val users_to_recs = getUserToRecordingEdges(sc, spark)
    val users_to_artists = getUserToArtistEdges(sc, spark)
    val artists_to_recs = getArtistToRecordingEdges(sc, spark)

    var m = Map[String, Any]()
    m += ("users" -> users.count())
    m += ("recs" -> recs.count())
    m += ("artists" -> artists.count())

    m += ("users_to_recs" -> users_to_recs.count())
    m += ("users_to_artists" -> users_to_artists.count())
    m += ("artists_to_recs" -> artists_to_recs.count())

    val arr1 = getEdgeSums(users_to_recs, max_parts)
    m += ("users_to_recs_sums" -> JSONObject(arr1(0)))

    val arr2 = getEdgeSums(users_to_artists, max_parts)
    m += ("users_to_artists_sums" -> JSONObject(arr2(0)))

    (10 to max_parts by 20).foreach(upto_part=> {
      val arr3 = getEdgeSums(users_to_recs, upto_part)
      m += ("users_to_recs_upto_part_" + upto_part.toString + "_sums" -> JSONObject(arr3(0)))
    })

    (10 to max_parts by 20).foreach(upto_part=> {
      val arr4 = getEdgeSums(users_to_artists, upto_part)
      m += ("users_to_artists_upto_part_" + upto_part.toString + "_sums" -> JSONObject(arr4(0)))
    })

    // Stop the underlying SparkContext
    sc.stop

    val file = new File("graph_properties.json")
    val bw = new BufferedWriter(new FileWriter(file))
    bw.write(JSONObject(m).toString())
    bw.close()
  }

  def getEdgeSums(df: DataFrame, parts: Int): Array[Map[String, Any]] ={
    var df_sums = df.select("years.*")
      .filter(col("part") <= parts)
      .agg(
        sum("yr_2004").alias("sum_2004"),
        sum("yr_2005").alias("sum_2005"),
        sum("yr_2006").alias("sum_2006"),
        sum("yr_2007").alias("sum_2007"),
        sum("yr_2008").alias("sum_2008"),
        sum("yr_2009").alias("sum_2009"),
        sum("yr_2010").alias("sum_2010"),
        sum("yr_2011").alias("sum_2011"),
        sum("yr_2012").alias("sum_2012"),
        sum("yr_2013").alias("sum_2013"),
        sum("yr_2014").alias("sum_2014"),
        sum("yr_2015").alias("sum_2015"),
        sum("yr_2016").alias("sum_2016"),
        sum("yr_2017").alias("sum_2017"),
        sum("yr_2018").alias("sum_2018"),
        sum("yr_2019").alias("sum_2019"),
      )
    df_sums.collect.map(r => Map(df_sums.columns.zip(r.toSeq): _*))
  }
}
