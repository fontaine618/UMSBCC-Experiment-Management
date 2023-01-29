# only this first line is necessary
cluster.functions = makeClusterFunctionsSlurm()
# but you can also specify some default parameters
default.resources = list(
  partition="standard",
  account="stats_dept1",
  chunks.as.arrayjobs=TRUE,
  job_name="R"
)