from pyspark.sql import SparkSession


def get_spark() -> SparkSession:
    """Get the spark session

    Modified to use local Spark session instead of Connect server for simplicity.
    This approach is more reliable for development and testing.

    Returns
    -------
    SparkSession
        An active spark session
    """
    # Use local Spark session - much simpler and more reliable than Connect server
    return SparkSession.builder \
        .appName("PySpark Skills Assessment") \
        .master("local[*]") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()
