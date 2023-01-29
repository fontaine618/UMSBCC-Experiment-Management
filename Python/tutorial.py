from pypet import Parameter, Trajectory, Environment, cartesian_product
import numpy as np
import pandas as pd
import math
from scipy.stats import ttest_ind
import seaborn as sns

# Obviously, you need to install all the above
# One thing I noticed is that recent versions of numpy do not work with
# the latest version of pypet (0.6.0). I had to downgrade to numpy==1.22.0.
# It appears that numpy dropped some types, but pypet relied on them.
# You can fix it pretty easily by changing a single file in pypet;
# contact me if you need help with that.


# This is the equivalent of defining a problem in batchtools
# Here we do not depend on data, but we could
# This depends on data-generating parameters included in the trajectory
def two_groups(traj: Trajectory) -> tuple[np.ndarray, np.ndarray]:
    # extract the data parameter group from the trajectory
    data_parms = traj.data
    np.random.seed(data_parms.seed)
    x = np.random.normal(0, 1, data_parms.n_per_group)
    y = np.random.normal(
        data_parms.mean_difference,
        math.sqrt(data_parms.var_ratio),
        data_parms.n_per_group
    )
    if data_parms.log_normal:
        x = np.exp(x)
        y = np.exp(y)
    return x, y


# This is the equivalent of defining an algorithm, which depends on
# data and parameters
def t_test(instance, equal_var=True):
    x, y = instance
    _, p = ttest_ind(x, y, equal_var=equal_var)
    return p


# This is normally done internally in batchtools
# In pypet, there is not really compartmentalization of problems and
# algorithms, so we have to do it ourselves
# This function simply executes whatever codes that needs to be run given parameters
def run_t_test(traj: Trajectory) -> float:
    # This creates the instance
    instance = two_groups(traj)
    # This is calling the algorithm on the instance with the algorithm parameters from
    # the trajectory
    p = t_test(instance, equal_var=traj.algo.equal_var)
    traj.f_add_result("$.p_value", p_value=p, comment="the p-value of the hypothesis test")
    # print(f"Run {traj.v_idx} finished, p-value={traj.test.p_value}")
    return p


# This is a helper function to get the parameters from the trajectory
# and create a pandas dataframe index by run index.
# This only works for nested parameters up to two levels
# The column names will be (parameter group, parameter name)
def parameter_df(runs: list[int], traj: Trajectory) -> pd.DataFrame:
    parameters = dict()
    for group_name, group in traj.par.f_get_groups().items():
        for par_name, par in group.f_get_children().items():
            try:
                values = par.f_get_range()
            except TypeError:  # patch for default
                values = [par.f_get_default()] * len(runs)
            parameters[(group_name, par_name)] = values
    df = pd.DataFrame(parameters, index=runs)
    return df


# In pypet, I find it really useful to use postprocessing to get the results
# Otherwise getting the results from each run is a bit tedious
# This function is called once all jobs are completed
def post_processing(traj: Trajectory, result_list: tuple[list[int], list[float]]) -> None:
    # result_list is a tuple of two lists of (hopefully) the same length
    # the first list contains the run indices
    # the second list contains the results, i.e., the output of run_t_test
    runs = [res[0] for res in result_list]
    p_values = [res[1] for res in result_list]
    results = pd.DataFrame({("test", "p_value"): p_values}, index=runs)
    parameters = parameter_df(runs, traj)
    # Store the results in the trajectory
    traj.f_add_result("summary", results=results, parameters=parameters)


def main():
    # make the registry
    env = Environment(
        trajectory="two_groups",
        filename="tutorial.hdf5",
        overwrite_file=True,
        multiproc=True,
        ncores=10,
    )
    traj = env.trajectory

    # we now parameters
    # this implicitly creates a pypet.Parameter
    # the naming convention is that dots are used to separate groups
    # the groups can be nested: group1.group2.parameter, etc.
    # the second argument is the default value
    # the comment argument is the description
    # pypet is a bit capricious with types, so it is best to be explicit
    # in particular, the type should match with the exploration below
    traj.f_add_parameter("data.n_per_group", np.int64(), comment="Number of samples per group")
    traj.f_add_parameter("data.mean_difference", np.float64(), comment="Mean difference between groups")
    traj.f_add_parameter("data.var_ratio", np.float64(), comment="Ratio of variances between groups")
    traj.f_add_parameter("data.log_normal", np.bool_(), comment="Log-normal distribution")
    traj.f_add_parameter("data.seed", np.int64(), comment="Random seed")
    traj.f_add_parameter("algo.equal_var", np.bool_(), comment="Equal variances")
    # parameter space
    traj.f_explore(cartesian_product({
        "data.n_per_group": np.array([10, 30], dtype=int),
        "data.mean_difference": np.linspace(0, 1, 6, dtype=float),
        "data.var_ratio": np.array([1.0, 0.1], dtype=float),
        "data.log_normal": np.array([False]),
        "data.seed": np.arange(20, dtype=int),
        "algo.equal_var": np.array([True, False])
    }))
    # here we add the postprocessing function, so the results will be summarized directly
    # for quick access
    env.add_postprocessing(post_processing)
    # the main call: this executes run_t_test for each parameter combination
    # defined above
    _ = env.run(run_t_test)
    env.disable_logging()
    # get results (notice that in postprocessing we added results under the summary group)
    traj.results.summary.results.head()
    traj.results.summary.parameters.head()
    # merge the two using run index
    results = traj.results.summary.results.merge(
        traj.results.summary.parameters,
        left_index=True,
        right_index=True
    )
    results.head()
    # plot
    sns.set_style("whitegrid")
    df = results.groupby([
        ('data', 'n_per_group'),
        ('data', 'mean_difference'),
        ('data', 'var_ratio'),
        ('data', 'log_normal'),
        ('algo', 'equal_var')
    ]).agg({("test", "p_value"): lambda x: np.mean(x < 0.05)})
    df.reset_index(inplace=True)
    grid = sns.FacetGrid(
        df,
        col=("data", "n_per_group"),
        row=("data", "var_ratio"),
        hue=("algo", "equal_var"),
        margin_titles=True
    )
    grid.map(sns.lineplot, ("data", "mean_difference"), ("test", "p_value"))
    grid.add_legend()
    grid.refline(y=0.05, linestyle=":", color="black")
    grid.savefig("results.pdf")

    # One thing I found really confusing with pypet is the Trajectory object.
    # It behaves differently depending on when/where it is interacted with
    # Inside a run, traj only contains the parameters and results of the current run
    # Outside a run, traj contains all the parameters and results of all runs

    # you can get all values of a parameter using: (indexed following run index)
    traj.f_get("data.seed").f_get_range()
    # for results per run, it's a bit harder
    traj.f_load(load_results=2) # this will load run-level results, otherwise, we only have access to summary
    print(traj.results.r_1.p_value) # this is the p-value of the first run
    print([x.p_value for x in traj.f_get_from_runs(name="p_value").values()]) # the list of all p-values
    # you can also iterate through all runs
    for run in traj.f_iter_runs():
        # set what is the "current" run
        traj.f_set_crun(run)
        # get the p-value of the current run; crun is a token that is replaced by the current run index
        print(traj.results.crun.p_value)


if __name__ == "__main__":
    main()

