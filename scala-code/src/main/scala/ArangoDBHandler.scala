import com.arangodb.spark.{ArangoSpark, ReadOptions}
import document_schemas.{ArtistToRecordingRelation, Node, UserToRecordingOrArtistRelation}
import org.apache.spark.sql.{Dataset, Row, SparkSession}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.functions._

class ArangoDBHandler(val spark: SparkSession) {
  private var users: Dataset[Row] = null
  private var artists: Dataset[Row] = null
  private var recs: Dataset[Row] = null

  private var users_to_recs: Dataset[Row] = null
  private var users_to_artists: Dataset[Row] = null
  private var artists_to_recs: Dataset[Row] = null

  // ====================================== Nodes ======================================
  def getUsers(force_reread: Boolean = false): Dataset[Row] = {
    if (force_reread || users == null) {
      val users_rdd = ArangoSpark.load[Node](spark.sparkContext,
        "users",
        ReadOptions("MLHD_processing", collection = "users"))
      users = spark.createDataFrame(rowRDD = users_rdd.map(x => x.getAsRow), new Node().getSchema)
        .persist(StorageLevel.DISK_ONLY)
      users
    } else {
      users
    }
  }

  def getArtists(force_reread: Boolean = false): Dataset[Row] = {
    if (force_reread || artists == null) {
      val artists_rdd = ArangoSpark.load[Node](spark.sparkContext,
        "artists",
        ReadOptions("MLHD_processing", collection = "artists"))
      artists = spark.createDataFrame(rowRDD = artists_rdd.map(x => x.getAsRow), new Node().getSchema)
        .persist(StorageLevel.DISK_ONLY)
      artists
    } else {
      artists
    }
  }

  def getRecordings(force_reread: Boolean = false): Dataset[Row] = {
    if (force_reread || recs == null) {
      val recs_rdd = ArangoSpark.load[Node](spark.sparkContext,
        "recordings",
        ReadOptions("MLHD_processing", collection = "recordings"))
      recs = spark.createDataFrame(rowRDD = recs_rdd.map(x => x.getAsRow), new Node().getSchema)
        .persist(StorageLevel.DISK_ONLY)
      recs
    } else {
      recs
    }
  }


  // ====================================== Edges ======================================
  def getUserToRecordingEdges(force_reread: Boolean = false): Dataset[Row] = {
    if (force_reread || users_to_recs == null) {
      val edge_rdd = ArangoSpark
        .load[UserToRecordingOrArtistRelation](spark.sparkContext,
          "users_to_recordings",
          ReadOptions("MLHD_processing", collection = "users_to_recordings"))
      users_to_recs = spark.createDataFrame(rowRDD = edge_rdd.map(x => x.getAsRow), new UserToRecordingOrArtistRelation().getSchema)
        .join(
          getUsers(force_reread).select(col("_id").as("_from"), col("node_id").as("user_id")),
          Seq("_from"), "inner"
        )
        .join(
          getRecordings(force_reread).select(col("_id").as("_to"), col("node_id").as("rec_id")),
          Seq("_to"), "inner"
        )
        .persist(StorageLevel.DISK_ONLY)
      users_to_recs
    } else {
      users_to_recs
    }
  }

  def getUserToArtistEdges(force_reread: Boolean = false): Dataset[Row] = {
    if (force_reread || users_to_artists == null) {
      val edge_rdd = ArangoSpark
        .load[UserToRecordingOrArtistRelation](spark.sparkContext,
          "users_to_artists",
          ReadOptions("MLHD_processing", collection = "users_to_artists"))
      users_to_artists = spark.createDataFrame(rowRDD = edge_rdd.map(x => x.getAsRow), new UserToRecordingOrArtistRelation().getSchema)
        .join(
          getUsers(force_reread).select(col("_id").as("_from"), col("node_id").as("user_id")),
          Seq("_from"), "inner"
        )
        .join(
          getArtists(force_reread).select(col("_id").as("_to"), col("node_id").as("artist_id")),
          Seq("_to"), "inner"
        )
        .persist(StorageLevel.DISK_ONLY)
      users_to_artists
    } else {
      users_to_artists
    }
  }

  def getArtistToRecordingEdges(force_reread: Boolean = false): Dataset[Row] = {
    if(force_reread || artists_to_recs==null) {
      val edge_rdd = ArangoSpark
        .load[ArtistToRecordingRelation](spark.sparkContext,
          "artists_to_recordings",
          ReadOptions("MLHD_processing", collection = "artists_to_recordings"))
      artists_to_recs = spark.createDataFrame(rowRDD = edge_rdd.map(x => x.getAsRow), new ArtistToRecordingRelation().getSchema)
        .join(
          getArtists(force_reread).select(col("_id").as("_from"), col("node_id").as("artist_id")),
          Seq("_from"), "inner"
        )
        .join(
          getRecordings(force_reread).select(col("_id").as("_to"), col("node_id").as("rec_id")),
          Seq("_to"), "inner"
        )
        .persist(StorageLevel.DISK_ONLY)
      artists_to_recs
    }else{
      artists_to_recs
    }
  }
}
