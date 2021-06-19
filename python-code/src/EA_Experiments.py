import inquirer
import sys, glob
from jmetal.algorithm.multiobjective import SPEA2
from jmetal.algorithm.multiobjective.gde3 import GDE3
from jmetal.algorithm.multiobjective.nsgaii import NSGAII
from jmetal.algorithm.multiobjective.nsgaiii import NSGAIII, UniformReferenceDirectionFactory
from jmetal.core.quality_indicator import *
from jmetal.core.solution import FloatSolution
from jmetal.lab.experiment import Experiment, Job, generate_summary_from_experiment, compute_wilcoxon
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.util.evaluator import MultiprocessEvaluator
from jmetal.util.termination_criterion import StoppingByEvaluations
import pandas as pd
import config
from src.utils.RecSysProblem import RecSysProblem
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import re
from paretoset import paretoset
import plotly.express as px
from pandas.plotting import parallel_coordinates

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Calibri"],
})
mpl.rcParams['figure.dpi'] = 300


def configure_experiment(problems: dict, n_run: int):
    jobs = []
    num_processes = config.num_cpu
    population = 10
    generations = 10
    max_evaluations = population * generations

    for run in range(n_run):
        for problem_tag, problem in problems.items():
            reference_point = FloatSolution([0, 0], [1, 1], problem.number_of_objectives, )
            reference_point.objectives = np.repeat(1, problem.number_of_objectives).tolist()
            jobs.append(
                Job(
                    algorithm=NSGAIII(
                        problem=problem,
                        population_size=population,
                        mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,
                                                    distribution_index=20),
                        crossover=SBXCrossover(probability=1.0, distribution_index=20),
                        population_evaluator=MultiprocessEvaluator(processes=num_processes),
                        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
                        reference_directions=UniformReferenceDirectionFactory(4, n_points=100),
                    ),
                    algorithm_tag='NSGAIII',
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

    return jobs


def plot_quality_indicators():
    data = pd.read_csv("QualityIndicatorSummary_modified.csv") \
        .rename(columns={'IndicatorValue': 'value',
                         'IndicatorName': 'Indicator',
                         'year': "Year"
                         })

    data_RS = data[data['set_type'] == 'RS']
    # fig = plt.figure()
    g = sns.FacetGrid(data_RS, height=3, aspect=1, row="Year", col="Indicator", sharey=False)
    g.fig.suptitle("EA observations Swarmplot for $RS$ test-sets\n(\\textit{lower} is always better)")
    g.map(sns.swarmplot, "Algorithm", "value", order=["NSGAIII", "GDE3", "SPEA2"], palette='Set2')
    plt.savefig('../images/EA-analysis.svg')
    plt.show()


def create_reference_fronts():
    for yr in [2005, 2008, 2012]:
        files = []
        out_files = []
        for file in glob.glob(config.EA_experiment_output_dir + "/**/FUN*.tsv", recursive=True):
            if re.match(r'.*RecSys-RS-yr_' + str(yr) + '.*', file):
                files.append(file)
                problem = re.findall(r'RecSys[^/]*', file)[0]
                out_files.append(problem + '.pf')
        df = pd.DataFrame()
        for file in files:
            temp_df = pd.read_csv(file, sep='\s+', header=None)
            df = df.append(temp_df)
        mask = paretoset(df, sense=["max", "max", "max", "max"])
        df = df[mask]

        for out_file in out_files:
            df.to_csv(config.EA_experiment_reference_fronts + out_file, index=False, sep=' ', header=False)
    pass


def plot_fronts():
    for yr in [2005, 2008, 2012]:
        df = pd.DataFrame()
        for file in glob.glob(config.EA_experiment_output_dir + "/**/FUN*.tsv", recursive=True):
            if re.match(r'.*RecSys-RS-yr_' + str(yr) + '.*', file):
                problem = re.findall(r'RecSys[^/]*', file)[0]
                algo = re.search(r'(GDE3)|(NSGAIII)|(SPEA2)', file).group()
                temp_df = pd.read_csv(file, sep='\s+', header=None)
                temp_df['algo'] = algo
                temp_df['problem'] = problem
                temp_df['col_value'] = np.random.uniform(0, 1, size=temp_df.shape[0])
                df = df.append(temp_df)
        df.columns = ['MAR', 'Cov', 'modPers', 'Nov', 'algo', 'problem', 'col_value']
        # print(df)
        fig = px.parallel_coordinates(df, color="col_value", dimensions=['MAR', 'Cov', 'modPers', 'Nov'],
                                      color_continuous_scale=px.colors.qualitative.Set2)
        fig.update(layout_coloraxis_showscale=False)
        fig.write_image(f"../images/front_objectives/front_obj_yr_{yr}.svg")
        return


def plot_variables():
    for yr in [2005, 2008, 2012]:
        df = pd.DataFrame()
        for file in glob.glob(config.EA_experiment_output_dir + "/**/VAR*.tsv", recursive=True):
            if re.match(r'.*RecSys-RS-yr_' + str(yr) + '.*', file):
                problem = re.findall(r'RecSys[^/]*', file)[0]
                algo = re.search(r'(GDE3)|(NSGAIII)|(SPEA2)', file).group()
                temp_df = pd.read_csv(file, sep='\s+', header=None)
                temp_df['algo'] = algo
                temp_df['problem'] = problem
                temp_df['col_value'] = np.random.uniform(0, 1, size=temp_df.shape[0])
                df = df.append(temp_df)
        df.columns = ["CF-user_rec", "CF-user_artist", "Tailored", 'algo', 'problem', 'col_value']
        df['sum'] = df[["CF-user_rec", "CF-user_artist", "Tailored"]].sum(axis=1)
        for colname in ["CF-user_rec", "CF-user_artist", "Tailored"]:
            df[colname] = df[colname] / (df['sum'] + 1e-6)
        # print(df)
        fig = px.parallel_coordinates(df, color="col_value", dimensions=["CF-user_rec", "CF-user_artist", "Tailored"],
                                      color_continuous_scale=px.colors.qualitative.Set1, )
        fig.update(layout_coloraxis_showscale=False)
        fig.write_image(f"../images/front_variables/front_var_yr_{yr}.pdf")

        df = df.rename(columns={"CF-user_rec": 'CF-user-rec', "CF-user_artist": "CF-user-artist"})

        fig = plt.figure(figsize=(8, 6))
        pd.plotting.parallel_coordinates(df, "algo",
                                         cols=["CF-user-rec", "CF-user-artist", "Tailored"],
                                         color=["lime", "tomato", "dodgerblue"],
                                         alpha=0.2,
                                         axvlines_kwds={"color": "black"})
        plt.autoscale()
        plt.legend(facecolor='white', framealpha=0.9)
        plt.savefig(f"../images/front_variables/front_var_yr_{yr}.svg")


if __name__ == '__main__':
    choices = {
        0: ".",
        1: 'Run EA Experiments',
        2: 'Aggregate Results',
    }
    output_directory = config.EA_experiment_output_dir
    if len(sys.argv) == 1:
        # No parametric input
        questions = [
            inquirer.List('choice',
                          message="What do you want to do?",
                          carousel=True,
                          choices=list(choices.values()),
                          ),
        ]
        answers = inquirer.prompt(questions)
        choice = answers["choice"]
        RS_or_EA = "RS"
        year = 2005
        set_ = 1
        data_stem = "../../scala-code/data/processed/"
    else:
        if len(sys.argv) != 7:
            raise Exception("Needs 6 arguments if arguments are provided.")
        choice = choices[int(sys.argv[1])]
        data_stem = sys.argv[2]
        output_directory = sys.argv[3]

        RS_or_EA = sys.argv[4]
        year = int(sys.argv[5])
        set_ = int(sys.argv[6])
        # algos = [sys.argv[7]]
        # reps = int(sys.argv[8])
    print(choice)
    if choice == "Run EA Experiments":
        # TODO take this as input

        problems = {}
        # for set_ in range(1, 4):
        problems[f"RecSys-{RS_or_EA}-yr_{year}-set_{set_}"] = \
            RecSysProblem(RS_or_EA=RS_or_EA,
                          models=["CF-user_rec", "CF-user_artist", "Tailored"],
                          metrics=['MAR', 'Cov', 'modPers', 'Nov'],
                          year=year,
                          set_=set_,
                          k=10,
                          K=10,
                          data_stem=data_stem)

        # Configure the experiments
        jobs = configure_experiment(problems=problems, n_run=3)

        # Run the study
        experiment = Experiment(output_dir=output_directory, jobs=jobs, m_workers=2)
        experiment.run()

    elif choice == 'Aggregate Results':
        # create_reference_fronts()
        # generate_summary_from_experiment(
        #     input_dir=output_directory,
        #     reference_fronts=config.EA_experiment_reference_fronts,
        #     quality_indicators=[GenerationalDistance(), InvertedGenerationalDistance(), EpsilonIndicator(),
        #                         HyperVolume([1.0, 1.0, 1.0, 1.0])]
        # )
        #
        df = pd.read_csv("QualityIndicatorSummary.csv")
        df = df.rename(columns={"ExecutionId": "rep"})
        df['year'] = df['Problem'].str.extract(r"(\d{4})")
        df['set_num'] = df['Problem'].str.extract(r"((?<=set_)\d{1})")
        df['set_type'] = df['Problem'].str.extract(r"((?<=RecSys-)\w{2})")
        df = df.pivot(index=['year', 'set_num', 'set_type', 'Algorithm', 'rep'],
                      columns="IndicatorName", values="IndicatorValue") \
            .reset_index()
        print(df)
        df.to_csv("QualityIndicatorSummary_modified.csv", index=False)

        # plot_quality_indicators()
        # plot_fronts()
        # plot_variables()
