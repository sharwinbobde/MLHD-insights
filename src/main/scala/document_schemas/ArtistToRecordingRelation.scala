package document_schemas

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{ShortType, StringType, StructType}

class ArtistToRecordingRelation (var _from: String,
                                 var _to: String,
                                 var part: Int) extends Serializable {
  def this() = this(_from = "", _to = "", -1)

  def getAsRow: Row = Row(_from, _to, part)

  def getSchema: StructType = {
    new StructType()
      .add("_from", StringType, nullable = false)
      .add("_to", StringType, nullable = false)
      .add("part", ShortType, nullable = false)
  }
}
