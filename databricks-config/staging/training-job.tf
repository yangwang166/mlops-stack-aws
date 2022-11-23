resource "databricks_job" "model_training_job" {
  name = "${local.env_prefix}mlops-stack-aws-seek-model-training-job"

  # Optional validation: we include it here for convenience, to help ensure that the job references a notebook
  # that exists in the current repo. Note that Terraform >= 1.2 is required to use these validations
  lifecycle {
    postcondition {
      condition     = alltrue([for task in self.task : fileexists("../../${task.notebook_task[0].notebook_path}.py")])
      error_message = "Databricks job must reference a notebook at a relative path from the root of the repo, with file extension omitted. Could not find one or more notebooks in repo"
    }
  }

  task {
    task_key = "Train"

    notebook_task {
      notebook_path = "notebooks/TrainWithFeatureStore"
      base_parameters = {
        env                = local.env
        training_data_path = "/databricks-datasets/nyctaxi-with-zipcodes/subsampled"
        experiment_name    = databricks_mlflow_experiment.experiment.name
        model_name         = "${local.env_prefix}mlops-stack-aws-seek-model"
      }
    }

    new_cluster {
      num_workers   = 3
      spark_version = "11.3.x-cpu-ml-scala2.12"
      node_type_id  = "i3.xlarge"
      custom_tags   = { "clusterSource" = "mlops-stack/0.0" }
    }
  }

  task {
    task_key = "TriggerModelDeploy"
    depends_on {
      task_key = "Train"
    }

    notebook_task {
      notebook_path = "notebooks/TriggerModelDeploy"
      base_parameters = {
        env = local.env
      }
    }

    new_cluster {
      num_workers   = 3
      spark_version = "11.3.x-cpu-ml-scala2.12"
      node_type_id  = "i3.xlarge"
      # We set the job cluster to single user mode to enable your training job to access
      # the Unity Catalog.
      single_user_name   = data.databricks_current_user.service_principal.user_name
      data_security_mode = "SINGLE_USER"
      custom_tags        = { "clusterSource" = "mlops-stack/0.0" }
    }
  }

  git_source {
    url      = var.git_repo_url
    provider = "gitHub"
    branch   = "main"
  }

  schedule {
    quartz_cron_expression = "0 0 16 * * ?" # daily at 16pm
    timezone_id            = "UTC"
  }

  # If you want to turn on notifications for this job, please uncomment the below code,
  # and provide a list of emails to the on_failure argument.
  #
  #  email_notifications {
  #    on_failure: []
  #  }
}

