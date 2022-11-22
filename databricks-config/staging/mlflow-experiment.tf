resource "databricks_mlflow_experiment" "experiment" {
  name        = "${local.mlflow_experiment_parent_dir}/${local.env_prefix}mlops-stack-aws-seek-experiment"
  description = "MLflow Experiment used to track runs for mlops-stack-aws-seek project."
}
