import com.arangodb.spark.{ArangoSpark, ReadOptions}
import document_schemas.{ArtistToRecordingRelation, Node, UserToRecordingOrArtistRelation}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel

class ArangoDBHandler(val spark: SparkSession) {
  private var users: Dataset[Row] = null
  private var artists: Dataset[Row] = null
  private var recs: Dataset[Row] = null

  private var users_to_recs: Dataset[Row] = null
  private var users_to_artists: Dataset[Row] = null
  private var artists_to_recs: Dataset[Row] = null

  // ====================================== Edges ======================================
  def getUserToRecordingEdges(): Dataset[Row] = {
    if (users_to_recs == null) {
      val edge_rdd = ArangoSpark
        .load[UserToRecordingOrArtistRelation](spark.sparkContext,
          "users_to_recordings",
          ReadOptions("MLHD_processing", collection = "users_to_recordings"))
      users_to_recs = spark.createDataFrame(rowRDD = edge_rdd.map(x => x.getAsRow), new UserToRecordingOrArtistRelation().getSchema)
        .join(
          getUsers().select(col("_id").as("_from"), col("node_id").as("user_id")),
          Seq("_from"), "inner"
        )
        .join(
          getRecordings().select(col("_id").as("_to"), col("node_id").as("rec_id")),
          Seq("_to"), "inner"
        )
        .persist(StorageLevel.DISK_ONLY)
    }
    users_to_recs
  }

  // ====================================== Nodes ======================================
  def getUsers(): Dataset[Row] = {
    if (users == null) {
      val users_rdd = ArangoSpark.load[Node](spark.sparkContext,
        "users",
        ReadOptions("MLHD_processing", collection = "users"))
      users = spark.createDataFrame(rowRDD = users_rdd.map(x => x.getAsRow), new Node().getSchema)
        .persist(StorageLevel.DISK_ONLY)
    }
    users
  }

  def getRecordings(): Dataset[Row] = {
    if (recs == null) {
      val recs_rdd = ArangoSpark.load[Node](spark.sparkContext,
        "recordings",
        ReadOptions("MLHD_processing", collection = "recordings"))
      recs = spark.createDataFrame(rowRDD = recs_rdd.map(x => x.getAsRow), new Node().getSchema)
        .persist(StorageLevel.DISK_ONLY)
    }
    recs
  }

  def getUserToArtistEdges(): Dataset[Row] = {
    if (users_to_artists == null) {
      val edge_rdd = ArangoSpark
        .load[UserToRecordingOrArtistRelation](spark.sparkContext,
          "users_to_artists",
          ReadOptions("MLHD_processing", collection = "users_to_artists"))
      users_to_artists = spark.createDataFrame(rowRDD = edge_rdd.map(x => x.getAsRow), new UserToRecordingOrArtistRelation().getSchema)
        .join(
          getUsers().select(col("_id").as("_from"), col("node_id").as("user_id")),
          Seq("_from"), "inner"
        )
        .join(
          getArtists().select(col("_id").as("_to"), col("node_id").as("artist_id")),
          Seq("_to"), "inner"
        )
        .persist(StorageLevel.DISK_ONLY)
    }
    users_to_artists
  }

  def getArtists(): Dataset[Row] = {
    if (artists == null) {
      val artists_rdd = ArangoSpark.load[Node](spark.sparkContext,
        "artists",
        ReadOptions("MLHD_processing", collection = "artists"))
      artists = spark.createDataFrame(rowRDD = artists_rdd.map(x => x.getAsRow), new Node().getSchema)
        .persist(StorageLevel.DISK_ONLY)
    }
    artists
  }

  def getArtistToRecordingEdges(): Dataset[Row] = {
    if (artists_to_recs == null) {
      val edge_rdd = ArangoSpark
        .load[ArtistToRecordingRelation](spark.sparkContext,
          "artists_to_recordings",
          ReadOptions("MLHD_processing", collection = "artists_to_recordings"))
      artists_to_recs = spark.createDataFrame(rowRDD = edge_rdd.map(x => x.getAsRow), new ArtistToRecordingRelation().getSchema)
        .join(
          getArtists().select(col("_id").as("_from"), col("node_id").as("artist_id")),
          Seq("_from"), "inner"
        )
        .join(
          getRecordings().select(col("_id").as("_to"), col("node_id").as("rec_id")),
          Seq("_to"), "inner"
        )
        .persist(StorageLevel.DISK_ONLY)
    }
    artists_to_recs
  }
}
