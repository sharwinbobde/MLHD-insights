package document_schemas

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{IntegerType, StringType, StructType}


class UserToRecordingOrArtistRelation(var _from: String,
                                      var _to: String,
                                      var years: Years,
                                      var part: Int) extends Serializable {
  def this() = this(_from = "", _to = "", years = new Years(), -1)

  def getAsRow: Row = Row(_from, _to, years.getAsRow, part)

  def getSchema: StructType = {
    new StructType()
      .add("_from", StringType, nullable = false)
      .add("_to", StringType, nullable = false)
      .add("years", new Years().getSchema, nullable = false)
      .add("part", IntegerType, nullable = false)
  }
}