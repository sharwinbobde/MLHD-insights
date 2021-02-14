package document_schemas

import org.apache.spark.sql.Row
import org.apache.spark.sql.types.{IntegerType, StructType}

class Years(
             var yr_2004: Int,
             var yr_2005: Int,
             var yr_2006: Int,
             var yr_2007: Int,
             var yr_2008: Int,
             var yr_2009: Int,
             var yr_2010: Int,
             var yr_2011: Int,
             var yr_2012: Int,
             var yr_2013: Int,
             var yr_2014: Int,
             var yr_2015: Int,
             var yr_2016: Int,
             var yr_2017: Int,
             var yr_2018: Int,
             var yr_2019: Int,
           ) extends Serializable {

  def this() = this(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

  def getAsRow: Row = Row(yr_2004, yr_2005, yr_2006, yr_2007, yr_2008, yr_2009, yr_2010, yr_2011, yr_2012, yr_2013, yr_2014, yr_2015, yr_2016, yr_2017, yr_2018, yr_2019)

  def getSchema: StructType = {
    new StructType()
      .add("yr_2004", IntegerType, nullable = false)
      .add("yr_2005", IntegerType, nullable = false)
      .add("yr_2006", IntegerType, nullable = false)
      .add("yr_2007", IntegerType, nullable = false)
      .add("yr_2008", IntegerType, nullable = false)
      .add("yr_2009", IntegerType, nullable = false)
      .add("yr_2010", IntegerType, nullable = false)
      .add("yr_2011", IntegerType, nullable = false)
      .add("yr_2012", IntegerType, nullable = false)
      .add("yr_2013", IntegerType, nullable = false)
      .add("yr_2014", IntegerType, nullable = false)
      .add("yr_2015", IntegerType, nullable = false)
      .add("yr_2016", IntegerType, nullable = false)
      .add("yr_2017", IntegerType, nullable = false)
      .add("yr_2018", IntegerType, nullable = false)
      .add("yr_2019", IntegerType, nullable = false)
  }
}
