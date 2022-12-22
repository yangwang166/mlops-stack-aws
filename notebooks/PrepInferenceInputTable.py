# Databricks notebook source
##################################################################################
# Model Training Notebook using Databricks Feature Store
#
# This notebook shows an example of a Model Training pipeline using Databricks Feature Store tables.
# It is configured and can be executed as a training-job.tf job defined under ``databricks-config``
#
# Parameters:
#
# * env (optional)  - String name of the current environment (dev, staging, or prod). Defaults to "dev".
# * training_data_path (required)   - Path to the training data.
# * experiment_name (required)      - MLflow experiment name for the training runs. Will be created if it doesn't exist.
# * model_name (required)           - MLflow registered model name to use for the trained model. Will be created if it
# *                                   doesn't exist.
##################################################################################


# COMMAND ----------

# DBTITLE 1, Notebook arguments
# List of input args needed to run this notebook as a job.
# Provide them via DB widgets or notebook arguments.
#
# Name of the current environment.
dbutils.widgets.dropdown("env", "dev", ["dev", "staging", "prod", "test"], "Environment Name")

# Path to the Hive-registered Delta table containing the training data.
dbutils.widgets.text("training_data_path", "/databricks-datasets/nyctaxi-with-zipcodes/subsampled", label="Path to the training data")

# MLflow experiment name.
dbutils.widgets.text("experiment_name", "mlops-ajmal-cuj-experiment-test", label="MLflow experiment name")

# MLflow registered model name to use for the trained mode..
dbutils.widgets.text("model_name", "mlops-ajmal-cuj-model-test", label="Model Name")

# COMMAND ----------

# DBTITLE 1,Define input and output variables
env = dbutils.widgets.get("env")
input_table_path = dbutils.widgets.get("training_data_path")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")

# COMMAND ----------

# DBTITLE 1, Set experiment
import mlflow

mlflow.set_experiment(experiment_name)

# COMMAND ----------

# DBTITLE 1, Load raw data
raw_data = spark.read.format("delta").load(input_table_path)
display(raw_data)

# COMMAND ----------

# DBTITLE 1, Helper functions
from pyspark.sql import *
from pyspark.sql.functions import current_timestamp, to_timestamp, lit
from pyspark.sql.types import IntegerType, TimestampType
import math
from datetime import timedelta, timezone
import mlflow.pyfunc


def rounded_unix_timestamp(dt, num_minutes=15):
    """
    Ceilings datetime dt to interval num_minutes, then returns the unix timestamp.
    """
    nsecs = dt.minute * 60 + dt.second + dt.microsecond * 1e-6
    delta = math.ceil(nsecs / (60 * num_minutes)) * (60 * num_minutes) - nsecs
    return int((dt + timedelta(seconds=delta)).replace(tzinfo=timezone.utc).timestamp())

rounded_unix_timestamp_udf = udf(rounded_unix_timestamp, IntegerType())


def rounded_taxi_data(taxi_data_df):
    # Round the taxi data timestamp to 15 and 30 minute intervals so we can join with the pickup and dropoff features
    # respectively.
    taxi_data_df = (
        taxi_data_df.withColumn(
            "rounded_pickup_datetime",
            to_timestamp(rounded_unix_timestamp_udf(taxi_data_df["tpep_pickup_datetime"], lit(15))),
        )
            .withColumn(
            "rounded_dropoff_datetime",
            to_timestamp(rounded_unix_timestamp_udf(taxi_data_df["tpep_dropoff_datetime"], lit(30))),
        )
            .drop("tpep_pickup_datetime")
            .drop("tpep_dropoff_datetime")
    )
    taxi_data_df.createOrReplaceTempView("taxi_data")
    return taxi_data_df


def get_latest_model_version(model_name):
    latest_version = 1
    mlflow_client = MlflowClient()
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


# COMMAND ----------

# DBTITLE 1, Read taxi data for training
taxi_data = rounded_taxi_data(raw_data)
display(taxi_data)

# COMMAND ----------

# DBTITLE 1, Create FeatureLookups
from databricks.feature_store import FeatureLookup
import mlflow

pickup_features_table = "mlops_stack_yang_seek.trip_pickup_features_" + env
dropoff_features_table = "mlops_stack_yang_seek.trip_dropoff_features_" + env

pickup_feature_lookups = [
    FeatureLookup(
        table_name = pickup_features_table,
        feature_names = ["mean_fare_window_1h_pickup_zip", "count_trips_window_1h_pickup_zip"],
        lookup_key = ["pickup_zip"],
        timestamp_lookup_key = ["rounded_pickup_datetime"]
    ),
]

dropoff_feature_lookups = [
    FeatureLookup(
        table_name = dropoff_features_table,
        feature_names = ["count_trips_window_30m_dropoff_zip", "dropoff_is_weekend"],
        lookup_key = ["dropoff_zip"],
        timestamp_lookup_key = ["rounded_dropoff_datetime"]
    ),
]

# COMMAND ----------

# DBTITLE 1, Create Training Dataset
from databricks import feature_store

# End any existing runs (in the case this notebook is being run for a second time)
mlflow.end_run()

# Start an mlflow run, which is needed for the feature store to log the model
mlflow.start_run()

# Since the rounded timestamp columns would likely cause the model to overfit the data 
# unless additional feature engineering was performed, exclude them to avoid training on them.
exclude_columns = ["rounded_pickup_datetime", "rounded_dropoff_datetime"]

fs = feature_store.FeatureStoreClient()

# Create the training set that includes the raw input data merged with corresponding features from both feature tables
training_set = fs.create_training_set(
    taxi_data,
    feature_lookups = pickup_feature_lookups + dropoff_feature_lookups,
    label = "fare_amount",
    exclude_columns = exclude_columns
)

# Load the TrainingSet into a dataframe which can be passed into sklearn for training a model
training_df = training_set.load_df()

# COMMAND ----------

# Display the training dataframe, and note that it contains both the raw input data and the features from the Feature Store, like `dropoff_is_weekend`
display(training_df)

# COMMAND ----------



# COMMAND ----------

# DBTITLE 1,Prep Inference Input Table
training_df.drop("fare_amount").write.format("delta").mode("overwrite").saveAsTable("jetstar_feature_store_taxi_example.inference_data")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC Train a LightGBM model on the data returned by `TrainingSet.to_df`, then log the model with `FeatureStoreClient.log_model`. The model will be packaged with feature metadata.

# COMMAND ----------

# DBTITLE 1,Define the search space for LightGBM
from hyperopt import fmin, tpe, hp, SparkTrials, Trials, STATUS_OK
from hyperopt.pyll import scope
from hyperopt.pyll.stochastic import sample
import numpy as np

search_space = {
  'class_weight': hp.choice('class_weight', [None, 'balanced']),
  'num_leaves': scope.int(hp.quniform('num_leaves', 30, 150, 1)),
  'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
  'subsample_for_bin': scope.int(hp.quniform('subsample_for_bin', 20000, 300000, 20000)),
  'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
  'bagging_fraction': hp.uniform('bagging_fraction', 0.5, 1),
  'min_data_in_leaf': scope.int(hp.qloguniform('min_data_in_leaf', 0, 6, 1)),
  'lambda_l1': scope.int(hp.choice('lambda_l1', [0, hp.loguniform('lambda_l1_positive', -16, 2)])),
  'lambda_l2': scope.int(hp.choice('lambda_l2', [0, hp.loguniform('lambda_l2_positive', -16, 2)])),
  'verbose': -1,
  'subsample': None,
  'reg_alpha': None,
  'reg_lambda': None,
  'min_sum_hessian_in_leaf': None,
  'min_child_samples': None,
  'colsample_bytree': None,
  'min_child_weight': scope.int(hp.loguniform('min_child_weight', -16, 5))
}

# COMMAND ----------

# DBTITLE 1,Create training and testing data sets
from sklearn.model_selection import train_test_split

features_and_label = training_df.columns

# Collect data into a Pandas array for training
data = training_df.toPandas()[features_and_label]

train, test = train_test_split(data, random_state=123)
X_train = train.drop(["fare_amount"], axis=1)
X_test = test.drop(["fare_amount"], axis=1)
y_train = train.fare_amount
y_test = test.fare_amount

# COMMAND ----------

# DBTITLE 1,Define the objective function for HyperOpt
from mlflow.models.signature import infer_signature
from sklearn.metrics import r2_score
import lightgbm as lgb
import mlflow.lightgbm


def objective_function(trial_params):
  with mlflow.start_run(nested=True):
    mlflow.lightgbm.autolog()
    
    # Redefine our pipeline with the new params from the search algo    
    train_lgb_dataset = lgb.Dataset(X_train, label=y_train.values)
    test_lgb_dataset = lgb.Dataset(X_test, label=y_test.values)
    
    # Fit a model with the trial's parameters
    num_rounds = 50
    model = lgb.train(trial_params, train_lgb_dataset, num_rounds)
    
    # Fit, predict, score
    predictions_valid = model.predict(X_test)
    r2_score_calc = r2_score(y_test.values, predictions_valid)
    
    # Log :)
    mlflow.log_metric('r2_score', r2_score_calc)

    # Set the loss to r2_score_calc so fmin maximizes the auc_score
    return {'status': STATUS_OK, 'loss': r2_score_calc}

# COMMAND ----------

# DBTITLE 1,Identify best parameters
# Greater parallelism will lead to speedups, but a less optimal hyperparameter sweep. 
spark_trials = SparkTrials(parallelism=2)

# Run fmin within an MLflow run context so that each hyperparameter configuration is logged as a child run of a parent
best_params = fmin(
  fn=objective_function, 
  space=search_space, 
  algo=tpe.suggest, 
  max_evals=2,
  trials=spark_trials, 
  return_argmin=False
)

# COMMAND ----------

# Retrain the model with the best parameters
train_lgb_dataset = lgb.Dataset(X_train, label=y_train.values)
test_lgb_dataset = lgb.Dataset(X_test, label=y_test.values)

# Fit a model with the trial's parameters
model = lgb.train(best_params, train_lgb_dataset, 100)

# COMMAND ----------

# DBTITLE 1, Log model and return output.
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# Log the trained model with MLflow and package it with feature lookup information.
fs.log_model(
    model,
    artifact_path="model_packaged",
    flavor=mlflow.lightgbm,
    training_set=training_set,
    registered_model_name=model_name
)

# Build out the MLflow model registry URL for this model version.
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
model_version = get_latest_model_version(model_name)
model_registry_url = "https://{workspace_url}/#mlflow/models/{model_name}/versions/{model_version}"\
    .format(workspace_url=workspace_url, model_name=model_name, model_version=model_version)

# The returned model URI is needed by the model deployment notebook.
model_uri = f"models:/{model_name}/{model_version}"
dbutils.jobs.taskValues.set("model_uri", model_uri)
dbutils.notebook.exit(model_uri)

# COMMAND ----------


