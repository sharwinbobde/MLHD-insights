package utils

import org.apache.spark.sql.Column
import org.apache.spark.sql.catalyst.expressions.aggregate.ApproximatePercentile
import org.apache.spark.sql.functions.lit

object PercentileApprox {

  def percentile_approx(col: Column, percentage: Column, accuracy: Column): Column = {
    val expr = new ApproximatePercentile(
      col.expr, percentage.expr, accuracy.expr
    ).toAggregateExpression
    new Column(expr)
  }

  def percentile_approx(col: Column, percentage: Column): Column = percentile_approx(
    col, percentage, lit(ApproximatePercentile.DEFAULT_PERCENTILE_ACCURACY)
  )
}
