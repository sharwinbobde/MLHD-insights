from jmetal.algorithm.multiobjective import MOEAD, SPEA2, HYPE
from jmetal.algorithm.multiobjective.gde3 import GDE3
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.core.quality_indicator import *
from jmetal.core.solution import FloatSolution
from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem import ZDT1, ZDT2, ZDT3
from jmetal.util.aggregative_function import Tschebycheff
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.evaluator import SparkEvaluator, MultiprocessEvaluator, DaskEvaluator
from jmetal.util.termination_criterion import StoppingByEvaluations

import config
from src.utils.RecSysProblem import RecSysProblem


def configure_experiment(problems: dict, n_run: int):
    jobs = []
    num_processes = 5
    population = 10
    generations = 2
    max_evaluations = population * generations

    for run in range(n_run):
        for problem_tag, problem in problems.items():
            reference_point = FloatSolution([0, 0], [1, 1], problem.number_of_objectives, )
            reference_point.objectives = np.repeat(1, problem.number_of_objectives).tolist()
            jobs.append(
                Job(
                    algorithm=NSGAII(
                        problem=problem,
                        population_size=population,
                        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
                                                    distribution_index=20),
                        crossover=SBXCrossover(probability=1.0, distribution_index=20),
                        population_evaluator=MultiprocessEvaluator(processes=num_processes),
                        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),

                        offspring_population_size=population,
                    ),
                    algorithm_tag='NSGAII',
                    problem_tag=problem_tag,

                    run=run,
                )
            )
            jobs.append(
                Job(
                    algorithm=GDE3(
                        problem=problem,
                        population_size=population,
                        population_evaluator=MultiprocessEvaluator(processes=num_processes),
                        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),

                        cr=0.5,
                        f=0.5,
                    ),
                    algorithm_tag='GDE3',
                    problem_tag=problem_tag,
                    run=run,
                )
            )

            jobs.append(
                Job(
                    algorithm=SPEA2(
                        problem=problem,
                        population_size=population,
                        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
                                                    distribution_index=20),
                        crossover=SBXCrossover(probability=1.0, distribution_index=20),
                        population_evaluator=MultiprocessEvaluator(processes=num_processes),
                        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),

                        offspring_population_size=population,
                    ),
                    algorithm_tag='SPEA2',
                    problem_tag=problem_tag,

                    run=run,
                )
            )
            # jobs.append(
            #     Job(
            #         algorithm=HYPE(
            #             problem=problem,
            #             population_size=population,
            #             mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
            #                                         distribution_index=20),
            #             crossover=SBXCrossover(probability=1.0, distribution_index=20),
            #             population_evaluator=MultiprocessEvaluator(processes=num_processes),
            #             termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            #
            #             offspring_population_size=population,
            #             reference_point=reference_point
            #         ),
            #         algorithm_tag='HYPE',
            #         problem_tag=problem_tag,
            #
            #         run=run,
            #     )
            # )

    return jobs


if __name__ == '__main__':
    RS_or_EA = "RS"
    problems = {}
    for set_ in range(1, 4):
        problems[f"RecSys_{set_}"] = RecSysProblem(RS_or_EA=RS_or_EA,
                                                   models=["CF-user_rec", "CF-user_artist"],
                                                   metrics=['MAR', 'Cov', 'Pers', 'Nov'],
                                                   year=2008,
                                                   set_=set_,
                                                   k=15,
                                                   K=15)

    # Configure the experiments
    jobs = configure_experiment(problems=problems, n_run=1)

    # Run the study
    output_directory = config.EA_experiment_output_dir + RS_or_EA + '/'
    experiment = Experiment(output_dir=output_directory, jobs=jobs, m_workers=1)
    experiment.run()

    # Generate summary file
    generate_summary_from_experiment(
        input_dir=output_directory,
        quality_indicators=[HyperVolume([1.0, 1.0])]
    )
