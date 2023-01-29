# ==============================================================================
# BATCHTOOLS TUTORIAL - PART I
# presented at UMSBCC (Jan 30th, 2023)
# by Simon Fontaine
#
# This first part works locally (all computation on your machine)
# ------------------------------------------------------------------------------



# ==============================================================================
# Required packages
library(batchtools)
library(tidyverse)
library(magrittr)
library(data.table)
# ------------------------------------------------------------------------------



# ==============================================================================
# Setup batchtools registry
registry = makeExperimentRegistry(
  # will do everything in a temporary folder, change this if you want to  
  # access your results at some later time point
  file.dir=NA, 
  # replicability
  seed=1
)
# Overwrite cluster.functions to utilize multiple cores
registry$cluster.functions = makeClusterFunctionsSocket(ncpus = 11)
# ------------------------------------------------------------------------------



# ==============================================================================
# Define a problem 
# here: a function that generates the relevant data given some parameters
two_groups = function(
    # required arguments, but not used here
    data, job, 
    # parameters used to generate data
    n_per_group, mean_difference, var_ratio, log_normal, seed,
    # to ensure compatibility if we have other problems
    ...
){
  set.seed(seed)
  mean_x = 0
  mean_y = -mean_difference
  var_x = 1
  var_y = var_ratio
  x = rnorm(n_per_group, mean_x, sqrt(var_x))
  y = rnorm(n_per_group, mean_y, sqrt(var_y))
  if(log_normal){
    x = exp(x)
    y = exp(x)
  }
  return(list(x=x, y=y))
}

addProblem(
  name="two_groups", 
  fun=two_groups,
  data=NULL
)
# ------------------------------------------------------------------------------



# ==============================================================================
# Define an algorithm
t_test = function(
    # required arguments, but not used here
    data, job, 
    # the instance, which is the output of "two_groups()" above
    instance, 
    # parameters for the method
    var.equal, 
    # to ensure compatibility if we have other algorithms
    ...
){
  dt = system.time({
    x = instance$x
    y = instance$y
    htest = t.test(x, y, var.equal=var.equal)
  })
  return(c(
    p_value=htest$p.value,
    time=dt[1])
  )
}

addAlgorithm(
  name="t_test",
  fun=t_test
)
# ------------------------------------------------------------------------------



# ==============================================================================
# Define the parameter exploration
# NB: the "CJ()" function creates data.tables from cartesian products

# List of problems to solve
problems = list(
  two_groups = CJ(
    n_per_group=c(10, 30),
    mean_difference=seq(0, 1, 0.2),
    var_ratio=c(1., 0.1),
    log_normal=c(F),
    seed=seq(200)
  )
)

# List of methods to apply
algorithms = list(
  t_test=CJ(var.equal=c(T, F))
)


addExperiments(
  prob.designs=problems,
  algo.designs=algorithms,
  repls=1
)
# NB: setting repls > 1 will repeat the instance x algo multiple times
# with a different seed. However, you cannot control the seed, so you cannot
# be sure you get the same instance. I prefer to pass seed as a parameter
# to instance directly instead.
# The docs says seed = original_seed + job_id, where original seed is set
# when initiating the registry.
# ------------------------------------------------------------------------------



# ==============================================================================
# Get current status
summarizeExperiments()
getStatus()

# send one job to see if we coded things correctly
testJob(1)

# send the first 100 jobs
submitJobs(1:100)
getStatus()

# chunking jobs can be more efficient
chunk_df = data.table(job.id=1:9600, chunk=1:110)
head(chunk_df)
submitJobs(chunk_df)
getStatus()

# ------------------------------------------------------------------------------



# ==============================================================================
# Process results

# Define a reduce function; here, it is just identity, 
# but it can be more complicated if your output is more complicated
reduce = function(result) result
results = reduceResultsDataTable(fun = reduce) %>% unwrap()

# get the job parameters (algo parameters and problem parameters)
parameters = getJobPars() %>% unwrap()

# merge together
results %<>% left_join(parameters, by="job.id")
head(results)
# ------------------------------------------------------------------------------



# ==============================================================================
# Plot results

agg_results = results %>% 
  group_by(
    problem, algorithm,
    n_per_group, mean_difference, var_ratio, log_normal,
    var.equal
  ) %>%
  summarise(
    prop_rejected=mean(p_value<0.05),
    mean_time=mean(time.user.self),
    n=n()
  )


ggplot(
  data=agg_results %>% mutate(var_ratio=as.character(var_ratio)),
  mapping=aes(
    x=mean_difference,
    y=prop_rejected,
    color=var.equal
  )
) + 
  geom_line() + 
  facet_wrap(
    ~n_per_group + var_ratio,
    labeller=labeller(
      n_per_group=~paste0("N=", .),
      var_ratio=~paste0("var=(1,", ., ")"),
      .multi_line=F
    ),
    nrow=2
  ) + 
  geom_hline(yintercept=0.05, linetype="dashed") + 
  theme_minimal()

# ------------------------------------------------------------------------------



# ==============================================================================
# BATCHTOOLS TUTORIAL - PART II
# This second part is to run on GreatLakes using the Slurm scheduler
# ------------------------------------------------------------------------------



# ==============================================================================
# Preparation
#
# You need to add two files to your home directory
# "/home/uniqname/"
#
# The first file tells batchtools to use the Slurm scheduler
# filename: .batchtools.conf.R
#
# The second file is a template for batchtools to create batch files
# filename: .batchtools.slurm.tmpl
# 
# Open R Session on the cluster (SSH or other)
# $ ssh uniqname@greatlakes.arc-ts.umich.edu
#
# Resolve authentification
# 
# Load R
# $ module load R
# 
# Start R Session
# $ R
# 
# Install packages if necessary
install.packages("batchtools")
install.packages("tidyverse")
# ------------------------------------------------------------------------------



# ==============================================================================
# Required packages
library(batchtools)
library(tidyverse)
library(magrittr)
library(data.table)
# ------------------------------------------------------------------------------



# ==============================================================================
# Setup batchtools registry (this will make use of the two files we just added)
registry = makeExperimentRegistry(
  file.dir="batchtools/test2",
  seed=1
)
# ------------------------------------------------------------------------------



# ==============================================================================
# Here we run all the same experiment setup as above until submitting
# ...
# ------------------------------------------------------------------------------



# ==============================================================================
# Get current status
summarizeExperiments()
getStatus()

# Prepare parameters
resources = list(
  account="stats_dept1",
  partition="standard",
  memory="1000m", # this is per cpu
  ncpus=1,
  walltime="10:00",
  chunks.as.arrayjobs=FALSE,
  job_name="two_groups"
)
# Now the rest is the same, but we need to pass resources as an argument

# try one job to see if we set things up correctly
testJob(1, external=T)

# send the first 2 jobs, 
# this will queue 2 slorm jobs
submitJobs(1:2, resources)
getStatus()

# send 10 chunks, each with 50 jobs
# since chunks.as.arrayjobs=FALSE, this will queue 10 jobs, each with 50 experiments
chunk_df = data.table(job.id=1:500, chunk=1:10)
head(chunk_df)
submitJobs(chunk_df, resources)
getStatus()

# send 10 chunks, each with 50 jobs
# since chunks.as.arrayjobs=TRUE, this will queue 100 jobs
resources$chunks.as.arrayjobs = TRUE
chunk_df = data.table(job.id=501:600, chunk=1:10)
submitJobs(chunk_df, resources)
getStatus()

# you can check your jobs by running this in the terminal:
# $ squeue -u uniqname
# This command will unroll job arrays:
# $ squeue -u uniqname -r

# ------------------------------------------------------------------------------













