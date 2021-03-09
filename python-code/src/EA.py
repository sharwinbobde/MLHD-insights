from jmetal.algorithm.multiobjective import NSGAII
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.lab.visualization import Plot, InteractivePlot
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.problem import ZDT1
from jmetal.util.evaluator import SparkEvaluator
from jmetal.util.observer import VisualizerObserver, ProgressBarObserver
from jmetal.util.termination_criterion import StoppingByEvaluations

from src.RecSysProblem import RecSysProblem

problem = RecSysProblem(models=["CF_user-rec", "CF_user-artist"],
                        metrics=['MAR', 'Cov', 'Pers', 'Nov'],
                        year=2005,
                        k=93,
                        K=93)

max_evaluations = 100*10

algorithm = NSGAIII(
    problem=problem,
    population_size=100,
    reference_directions=UniformReferenceDirectionFactory(problem.number_of_objectives, n_points=50),
    mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables, distribution_index=20),
    crossover=SBXCrossover(probability=1.0, distribution_index=20),
    population_evaluator=SparkEvaluator(),
    termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
)
# algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
# algorithm.observable.register(
#     observer=VisualizerObserver(reference_front=problem.reference_front, display_frequency=100))

algorithm.run()
front = algorithm.get_result()
print(front)
plot_front = Plot(title='Pareto front approximation',
                  reference_front=problem.reference_front,
                  axis_labels=problem.obj_labels)
plot_front.plot(front,
                label='NSGAIII-' + problem.get_name(),
                filename='../images/' + algorithm.get_name() + "-" + problem.get_name(),
                format='png')

# Plot interactive front
plot_front = InteractivePlot(title='Pareto front approximation. Problem: ' + problem.get_name(),
                             reference_front=problem.reference_front,
                             axis_labels=problem.obj_labels)
plot_front.plot(front,
                label='NSGAIII-' + problem.get_name(),
                filename='../images/' + algorithm.get_name() + "-" + problem.get_name(), )
