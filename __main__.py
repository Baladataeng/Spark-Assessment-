from pyspark.sql import functions as sf
from pyspark.testing import assertDataFrameEqual

from assessment.helpers import get_spark
from assessment.transform import compute

if __name__ == "__main__":

    spark = get_spark()

    edges = spark.read.json(
        "data/edges.jsonl",
        """
        parent STRING,
        child STRING
        """,
    )

    output = compute(edges)

    output.write.mode("overwrite").json("output.jsonl")
    assertDataFrameEqual(
        output,
        spark.read.json(
            "data/solution.jsonl",
            """
            node STRING,
            path STRING,
            depth LONG,
            descendants LONG,
            is_root BOOLEAN,
            is_leaf BOOLEAN
            """,
        ),
        ignoreNullable=True,
    )

    print(f"Success!")

    # solution.write.json("data/solution.jsonl")
