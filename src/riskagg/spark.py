"""Spark session factory."""

from pyspark.sql import SparkSession


def get_spark(app_name: str = "riskagg", **conf: str) -> SparkSession:
    builder = (
        SparkSession.builder.master("local[*]")
        .appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.session.timeZone", "UTC")
    )
    for key, value in conf.items():
        builder = builder.config(key.replace("__", "."), str(value))
    return builder.getOrCreate()
