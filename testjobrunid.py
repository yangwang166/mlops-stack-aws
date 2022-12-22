# Databricks notebook source
# MAGIC %sql
# MAGIC create database if not exists yang

# COMMAND ----------

df = spark.read.load("/databricks-datasets/learning-spark-v2/people/people-10m.delta")

# Write the data to a table.
table_name = "yang.people_10m"
df.write.mode("append").saveAsTable(table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC desc history yang.people_10m

# COMMAND ----------


