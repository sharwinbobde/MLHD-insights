import com.arangodb.spark.{ArangoSpark, ReadOptions}
import document_schemas.{ArtistToRecordingRelation, Node, UserToRecordingOrArtistRelation}
import org.apache.spark.SparkContext
import org.apache.spark.sql.functions.monotonically_increasing_id
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

object ArangoDBHandler {

  def getUsers(sc: SparkContext, spark: SparkSession): Dataset[Row] = {
    val users_rdd = ArangoSpark.load[Node](sc, "users", ReadOptions("MLHD_processing", collection = "users"))
    spark.createDataFrame(rowRDD = users_rdd.map(x => x.getAsRow), new Node().getSchema)
      .withColumn("user_id", monotonically_increasing_id())
      .persist(StorageLevel.DISK_ONLY)
  }

  def getArtists(sc: SparkContext, spark: SparkSession): Dataset[Row] = {
    val artists_rdd = ArangoSpark.load[Node](sc, "artists", ReadOptions("MLHD_processing", collection = "artists"))
    spark.createDataFrame(rowRDD = artists_rdd.map(x => x.getAsRow), new Node().getSchema)
      .withColumn("artist_id", monotonically_increasing_id())
      .persist(StorageLevel.DISK_ONLY)
  }

  def getRecordings(sc: SparkContext, spark: SparkSession): Dataset[Row] = {
    val recs_rdd = ArangoSpark.load[Node](sc, "recordings", ReadOptions("MLHD_processing", collection = "recordings"))
    spark.createDataFrame(rowRDD = recs_rdd.map(x => x.getAsRow), new Node().getSchema)
      .withColumn("rec_id", monotonically_increasing_id())
      .persist(StorageLevel.DISK_ONLY)
  }

  def getUserToRecordingEdges(sc: SparkContext, spark: SparkSession): Dataset[Row] = {
    val user_to_recs_rdd = ArangoSpark
      .load[UserToRecordingOrArtistRelation](sc,
        "users_to_recordings",
        ReadOptions("MLHD_processing", collection = "users_to_recordings"))
    spark.createDataFrame(rowRDD = user_to_recs_rdd.map(x => x.getAsRow), new UserToRecordingOrArtistRelation().getSchema)
      .persist(StorageLevel.DISK_ONLY)
  }

  def getUserToArtistEdges(sc: SparkContext, spark: SparkSession): Dataset[Row] = {
    val user_to_recs_rdd = ArangoSpark
      .load[UserToRecordingOrArtistRelation](sc,
        "users_to_artists",
        ReadOptions("MLHD_processing", collection = "users_to_artists"))
    spark.createDataFrame(rowRDD = user_to_recs_rdd.map(x => x.getAsRow), new UserToRecordingOrArtistRelation().getSchema)
      .persist(StorageLevel.DISK_ONLY)
  }

  def getArtistToRecordingEdges(sc: SparkContext, spark: SparkSession): Dataset[Row] = {
    val user_to_recs_rdd = ArangoSpark
      .load[ArtistToRecordingRelation](sc,
        "artists_to_recordings",
        ReadOptions("MLHD_processing", collection = "artists_to_recordings"))
    spark.createDataFrame(rowRDD = user_to_recs_rdd.map(x => x.getAsRow), new ArtistToRecordingRelation().getSchema)
      .persist(StorageLevel.DISK_ONLY)
  }
}
