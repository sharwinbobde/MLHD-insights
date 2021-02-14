package utils

import breeze.stats.distributions.Rand
import scala.collection.mutable

// inspired from Benoit Descamps https://towardsdatascience.com/hyperparameters-part-ii-random-search-on-spark-77667e68b606
class RandomGridGenerator(n: Int) {

  private val paramDistributions = mutable.Map.empty[String, Any]

  def addDistr[T](param: String, distr: Any): this.type = distr match {
    case _: Rand[_] => {
      paramDistributions.put(param, distr)
      this
    }
    case _: Array[_] => {
      paramDistributions.put(param, distr)
      this
    }
    case _ => throw new NotImplementedError("Distribution should be of type breeze.stats.distributions.Rand or an Array")
  }

  def getSamples(): Array[mutable.Map[String, Any]] = {
    val paramMaps = (1 to n).map(_ => mutable.Map.empty[String, Any])

    paramDistributions.foreach {
      case (param: String, distribution: Any) =>
        distribution match {
          case d: Rand[_] =>
            paramMaps.map(_.put(param, d.sample()))
          case d: Array[_] =>
            val r = scala.util.Random
            paramMaps.map(_.put(param, d(r.nextInt(d.length))))
        }
    }
    paramMaps.toArray
  }

}
