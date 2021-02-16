package document_schemas

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{LongType, ShortType, StringType, StructType}

class Node(var _id: String,
           var _key: String,
           var node_id: Long) extends Serializable {
  def this() = this(_id = "", _key = "", -1)

  def getAsRow: Row = Row(_id, _key, node_id)

  def getSchema: StructType = {
    new StructType()
      .add("_id", StringType, nullable = false)
      .add("_key", StringType, nullable = false)
      .add("node_id", LongType, nullable = false)
  }
}