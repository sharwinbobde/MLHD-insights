package document_schemas

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{ShortType, StringType, StructType}

class Node(var _id: String,
           var _key: String) extends Serializable {
  def this() = this(_id = "", _key = "")

  def getAsRow: Row = Row(_id, _key)

  def getSchema: StructType = {
    new StructType()
      .add("_id", StringType, nullable = false)
      .add("_key", StringType, nullable = false)
  }
}