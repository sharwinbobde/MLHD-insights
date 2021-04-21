from jmetal.algorithm.multiobjective.gde3 import GDE3
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.core.quality_indicator import *
from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem import ZDT1, ZDT2, ZDT3
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.evaluator import SparkEvaluator, MultiprocessEvaluator
from jmetal.util.termination_criterion import StoppingByEvaluations

import config
from src.utils.RecSysProblem import RecSysProblem


def configure_experiment(problems: dict, n_run: int):
    jobs = []
    max_evaluations = 25000

    for run in range(n_run):
        for problem_tag, problem in problems.items():
            jobs.append(
                Job(
                    algorithm=NSGAII(
                        problem=problem,
                        population_size=100,
                        offspring_population_size=100,
                        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
                                                    distribution_index=20),
                        crossover=SBXCrossover(probability=1.0, distribution_index=20),
                        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
                        population_evaluator=SparkEvaluator(processes=config.num_cpu)
                    ),
                    algorithm_tag='NSGAII',
                    problem_tag=problem_tag,
                    run=run,
                )
            )
            # jobs.append(
            #     Job(
            #         algorithm=GDE3(
            #             problem=problem,
            #             population_size=100,
            #             cr=0.5,
            #             f=0.5,
            #             termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            #         ),
            #         algorithm_tag='GDE3',
            #         problem_tag=problem_tag,
            #         run=run,
            #     )
            # )
            # jobs.append(
            #     Job(
            #         algorithm=SMPSO(
            #             problem=problem,
            #             swarm_size=100,
            #             mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
            #                                         distribution_index=20),
            #             leaders=CrowdingDistanceArchive(100),
            #             termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            #         ),
            #         algorithm_tag='SMPSO',
            #         problem_tag=problem_tag,
            #         run=run,
            #     )
            # )

    return jobs


if __name__ == '__main__':

    problem = RecSysProblem(test_set_type="RS",
                            models=["CF-user_rec", "CF-user_artist"],
                            metrics=['MAR', 'Cov', 'Pers', 'Nov'],
                            year=2008,
                            k=10,
                            K=10)

    # Configure the experiments
    jobs = configure_experiment(problems={'RecSys': problem}, n_run=10)
    # Run the study
    output_directory = config.EA_experiment_output_dir
    experiment = Experiment(output_dir=output_directory, jobs=jobs)
    experiment.run()

    # Generate summary file
    generate_summary_from_experiment(
        input_dir=output_directory,
        quality_indicators=[HyperVolume([1.0, 1.0])]
    )
