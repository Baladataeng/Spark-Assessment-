# PySpark Skills Assessment

## Pre-requisites

### Spark

This assessment assumes you have the ability to spin up or access a Spark cluster on your machine.

The data size is trivial, so a standalone local cluster is more than enough compute to complete the exercise.

The code as-is is designed to function with a Spark Connect server running on `localhost`. It was tested with Spark 4.0.0, but you are free to use another version of Spark.

> For version >= 3.5, the only modifications that might be required are to `assessment.helpers:get_spark` which provides the `SparkSession` for running the script.

> Earlier versions < 3.5 might need to replace the call to `assertDataFrameEqual` in `assessment.__main__` with another equality check.

### Python

The assessment also requires a python installation. The only python dependency is `pyspark`, as noted in the `requirements.txt` file.

The code was written with `python==3.13.5` and `pyspark==4.0.0`, so using other versions might require slight modifications.

## Data

### Input

The input data is stored in `data/edges.jsonl`, and represents edges in a hierarchical structure. There are no loops, and a single root node.

schema:

```
parent STRING,
child STRING
```

### Output

The expected output data is stored in `data/solution.jsonl`, and represents a dataset with one record per node, with associated information about the nodes relationship to the remainder of the structure.

schema:

```
node STRING, # the unique string value identifying each parent/child node
path STRING, # the concatenation of all node strings starting at the root
depth LONG, # the number of steps removed from the root node
descendants LONG, # the total number of nodes below a node in the hierarchy
is_root BOOLEAN, # indicates whether the node is the sole root node
is_leaf BOOLEAN # indicates whether the node is a leaf (i.e. no children)
```

## Logic

All logic should be writen in `assessment.transform:compute`, such that the return value for the function is a dataframe matching the description above.

The problem is not overly complex, but should test your ability to work within the Spark framework. Although the data is small, your code should be written to handle arbitrarily large datasets.

> Note: Assume this means arbitrary number of nodes, and an _unknown but limited_ maximum depth of the hierarchy - e.g. on the order of 10.

As the solution for the input data is provided, the approach and implementation are what is being evaluated. Demonstrating knowledge of Spark and how it might guide or constrain your approach to the solution is more important than arriving at the solution itself. To that effect, annotations and notes are encouraged (within reason) to help highlight tradeoffs or considerations made.

## Testing

Assuming you have set up your environment as described, testing should be as simple as starting up a Spark Connect server locally, and calling the module as a script - i.e. `python -m assessment`.

If your code runs successfully, the dataset will be written back to disk and checked against the solution.

When submitting the response, please include the output data and code in an archive. Avoid including any virtual environments in that archive, please :)
