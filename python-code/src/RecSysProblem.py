from jmetal.core.problem import Problem, FloatProblem
from jmetal.core.solution import FloatSolution
import numpy as np

class RecSysProblem(FloatProblem):

    def __init__(self):
        super(RecSysProblem, self).__init__()
        self.number_of_variables = 2
        self.number_of_objectives = 3
        self.number_of_constraints = 1

        self.obj_directions = [self.MAXIMIZE, self.MAXIMIZE]
        self.obj_labels = [
            'MAP',
            'MAR',
            'Cov',
            # 'Pers'
        ]

        self.lower_bound = [0.0 for _ in range(self.number_of_variables)]
        self.upper_bound = [1.0 for _ in range(self.number_of_variables)]

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1 = solution.variables[0]
        x2 = solution.variables[1]

        solution.objectives[0] = x1
        solution.objectives[1] = x2
        solution.objectives[2] = x1+x2
        # solution.objectives[3] = x1*0.1-x2*0.2
        self.__evaluate_constraints(solution)
        return solution

    def __evaluate_constraints(self, solution: FloatSolution) -> None:
        # all weights sum up to 1.0
        solution.constraints[0] = np.sum(solution.variables) - 1.0

    def get_name(self):
        return "RecommenderSystemFusion"
