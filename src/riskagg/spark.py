"""Spark session factory."""

from pyspark.sql import SparkSession

# Non-default settings every riskagg session asks for; the run manifest records their values.
SESSION_CONF: dict[str, str] = {
    "spark.driver.memory": "4g",  # local mode: the driver is the executor; 1g OOMs on demo
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.shuffle.partitions": "8",
    "spark.sql.session.timeZone": "UTC",
}


def get_spark(app_name: str = "riskagg", **conf: str) -> SparkSession:
    builder = SparkSession.builder.master("local[*]").appName(app_name)
    for key, value in SESSION_CONF.items():
        builder = builder.config(key, value)
    for key, value in conf.items():
        builder = builder.config(key.replace("__", "."), str(value))
    return builder.getOrCreate()
