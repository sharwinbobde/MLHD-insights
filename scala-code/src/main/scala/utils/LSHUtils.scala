package utils

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf

import scala.collection.mutable

object LSHUtils {

  // UDF for finding hamming distance
  val hammingDistUDF: UserDefinedFunction = udf((a: mutable.WrappedArray[Int], b: mutable.WrappedArray[Int]) => {
    var dist = 0
    for (i <- 0 until 256) {
      val xor = a(i) ^ b(i)
      dist += Integer.bitCount(xor)
    }
    dist
  })


}
