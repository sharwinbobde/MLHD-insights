import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

from src.utils.MetricsEvaluator import MetricsEvaluator
from src.utils.RecommendationUtils import RecommendationUtils
from datetime import datetime

data_stem = "../../scala-code/data/processed/"


class RecSysProblem(FloatProblem):

    def __init__(self, RS_or_EA: str, models: list[str], metrics: list[str], year: int, set_:int, k: int, K: int):
        super(RecSysProblem, self).__init__()
        self.number_of_variables = len(models)
        self.number_of_objectives = len(metrics)
        self.metrics = metrics

        self.year = year
        self.set_ = set_
        self.k = k
        self.K = K
        self.models = models

        self.obj_directions = [self.MAXIMIZE, self.MAXIMIZE]
        self.obj_labels = metrics

        self.metrics_evaluator = MetricsEvaluator(RS_or_EA=RS_or_EA,
                                                  year=year, k=k,
                                                  models=["CF-user_rec", "CF-user_artist"],
                                                  archive_size=100)

        self.lower_bound = [0.0 for _ in range(self.number_of_variables)]
        self.upper_bound = [1.0 for _ in range(self.number_of_variables)]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        normalised_weights = solution.variables / np.sum(solution.variables)

        recs = RecommendationUtils \
            .get_recommendations_dict_from_many_df(models=self.models,
                                                   set_num=self.set_,
                                                   model_recs_df=self.metrics_evaluator.recommendation_dfs,
                                                   reranking_weights=normalised_weights,
                                                   K=self.K)

        m = self.metrics_evaluator.get_metrics(self.metrics, recs, self.set_, self.K)
        for i in range(self.number_of_objectives):
            solution.objectives[i] = m[self.obj_labels[i]]
        return solution

    def get_name(self):
        return "RecommenderSystemFusion"
