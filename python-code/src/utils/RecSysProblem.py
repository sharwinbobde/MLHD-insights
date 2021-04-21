import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

from src.utils.MetricsEvaluator import MetricsEvaluator
from src.utils.RecommendationUtils import RecommendationUtils


data_stem = "../../scala-code/data/processed/"


class RecSysProblem(FloatProblem):

    def __init__(self, test_set_type: str, models: list[str], metrics: list[str], year: int, k: int, K: int):
        super(RecSysProblem, self).__init__()
        self.number_of_variables = 2
        self.number_of_objectives = len(metrics)

        # self.rec_utils = RecommendationUtils(data_stem, test_set_type=test_set_type)
        # self.number_of_constraints = 1

        self.year = year
        self.k = k
        self.K = K
        self.models = models

        self.obj_directions = [self.MAXIMIZE, self.MAXIMIZE]
        self.obj_labels = metrics

        self.metrics_evaluator = MetricsEvaluator(year=year, k=k)

        self.lower_bound = [0.0 for _ in range(self.number_of_variables)]
        self.upper_bound = [1.0 for _ in range(self.number_of_variables)]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # TODO average over 3 sets
        set_ = 2
        normalised_weights = solution.variables / np.sum(solution.variables)
        # recs = self.metrics_evaluator.rec_utils \
        #     .get_recommendations_dict_many_model(self.year, self.models, set_, self.k, self.K, normalised_weights)

        recs = RecommendationUtils \
            .get_recommendations_dict_from_many_df(models=self.models,
                                                   set_num=set_,
                                                   model_recs_df=self.metrics_evaluator.recommendation_dfs,
                                                   reranking_weights=normalised_weights,
                                                   K=self.K)

        m = self.metrics_evaluator.get_all_metrics(recs, set_, self.K)
        for i in range(self.number_of_objectives):
            solution.objectives[i] = m[self.obj_labels[i]]
        # self.__evaluate_constraints(solution)
        return solution

    # def __evaluate_constraints(self, solution: FloatSolution) -> None:
    #     # all weights sum up to 1.0
    #     solution.constraints[0] = np.sum(solution.variables) - 1.0

    def get_name(self):
        return "RecommenderSystemFusion"
