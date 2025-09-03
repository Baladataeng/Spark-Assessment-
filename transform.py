from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, LongType, BooleanType


def compute(edges: DataFrame) -> DataFrame:
    """The assessment logic

    This function is where you will define the logic for the assessment. The
    function will receive a dataframe of edges representing a hierarchical structure,
    and return a dataframe of nodes containing structural information describing the
    hierarchy.

    Consider the tree structure below:
    A
    |-- B
    |   `-- C
    |-- D
    `-- E
        |-- F
        `-- G


    Parameters
    ----------
    edges: DataFrame
        A dataframe of edges representing a hierarchical structure. For the example
        above, the dataframe would be:

        | parent | child |
        |--------|-------|
        | A      | B     |
        | B      | C     |
        | A      | D     |
        | A      | E     |
        | E      | F     |
        | E      | G     |

    Returns
    -------
    DataFrame
        A dataframe of nodes with the following columns:
            - node: The node of the hierarchy
            - path: The path to the node from the root
            - depth: The depth of the node in the hierarchy
            - descendents: The number of descendents of the node
            - is_root: Whether the node is the root of the hierarchy
            - is_leaf: Whether the node is a leaf of the hierarchy

        The output dataframe for the example above would look like:

        | node | path  | depth | descendents | is_root | is_leaf |
        |------|-------|-------|-------------|---------|---------|
        | A    | A     | 0     | 6           | true    | false   |
        | B    | A.B   | 1     | 1           | false   | false   |
        | C    | A.B.C | 2     | 0           | false   | true    |
        | D    | A.D   | 1     | 0           | false   | true    |
        | E    | A.E   | 1     | 2           | false   | false   |
        | F    | A.E.F | 2     | 0           | false   | true    |
        | G    | A.E.G | 2     | 0           | false   | true    |

    """
    
    # ============================================================================
    # ALGORITHM IMPLEMENTATION: Hierarchical Data Processing
    # ============================================================================
    # This implementation uses an iterative breadth-first approach to build
    # hierarchical information from edge data. The algorithm is designed for
    # scalability with Spark's distributed processing model.
    # ============================================================================
    
    # Step 1: Root Node Discovery
    # ---------------------------------------------------------------------------  
    # Root nodes appear as parents but never as children in the edge list.
    # We use set difference (subtract) to efficiently identify the single root.
    # This approach works for any tree structure and scales well with large datasets.
    
    parents = edges.select("parent").distinct()
    children = edges.select(F.col("child").alias("parent")).distinct()  
    root_nodes = parents.subtract(children)
    root_node = root_nodes.collect()[0]["parent"]  # Assumes single root (as per problem)
    
    # Step 2: Iterative Path Construction (Breadth-First Traversal)
    # ---------------------------------------------------------------------------
    # Build complete paths from root to all nodes level by level.
    # This approach leverages Spark's batch processing model and avoids recursion.
    
    spark = edges.sparkSession
    
    # Initialize with root node (depth=0, path=node_name)
    paths_df = spark.createDataFrame([(root_node, root_node, 0)], ["node", "path", "depth"])
    
    # Process each depth level iteratively
    max_depth = 10  # Safety limit for deep hierarchies (per problem statement ~10 max depth)
    for depth in range(max_depth):
        # Find all children of nodes at current depth level
        next_level = paths_df.filter(F.col("depth") == depth).alias("p").join(
            edges.alias("e"), 
            F.col("p.node") == F.col("e.parent"),  # Match parent nodes with their children
            "inner"
        ).select(
            F.col("e.child").alias("node"),  # Child becomes the new node
            # Build full path: parent_path + "." + child_name
            F.concat(F.col("p.path"), F.lit("."), F.col("e.child")).alias("path"),
            F.lit(depth + 1).alias("depth")  # Increment depth
        )
        
        # Early termination: if no children found at this level, hierarchy is complete
        if next_level.count() == 0:
            break
            
        # Combine new level with existing paths
        paths_df = paths_df.union(next_level)
    
    # Step 3: Descendant Count Calculation
    # ---------------------------------------------------------------------------
    # For each node, count all nodes whose path starts with this node's path + "."
    # This string-based approach efficiently identifies descendant relationships
    # and scales well with Spark's string processing capabilities.
    
    descendants_counts = paths_df.alias("parent").join(
        paths_df.alias("child"),
        # Match condition: child path starts with "parent.path."
        F.col("child.path").startswith(F.concat(F.col("parent.path"), F.lit("."))),
        "left"  # Include nodes with no descendants (count will be 0)
    ).groupBy("parent.node", "parent.path", "parent.depth").agg(
        F.count("child.node").alias("descendants")  # Count matching descendants
    )
    
    # Step 4: Leaf Node Identification
    # ---------------------------------------------------------------------------
    # Leaf nodes are those that don't appear as parents in the original edge data.
    # We use a left join pattern to identify these nodes efficiently.
    
    parent_nodes = edges.select("parent").distinct().withColumn("is_leaf", F.lit(False))
    
    # Step 5: Final Result Assembly
    # ---------------------------------------------------------------------------
    # Combine all computed information into the required output schema.
    # Use coalesce to handle null values from left joins gracefully.
    
    result = descendants_counts.alias("d").join(
        parent_nodes.select(F.col("parent").alias("node"), "is_leaf"),
        "node",
        "left"  # Left join to preserve all nodes from paths
    ).select(
        "node",                                   # Unique node identifier
        "path",                                   # Full path from root
        "depth",                                  # Distance from root  
        "descendants",                            # Count of all descendants
        # Root identification: only the discovered root node gets True
        F.when(F.col("node") == root_node, True).otherwise(False).alias("is_root"),
        # Leaf identification: nodes not in parent_nodes are leaves
        F.coalesce("is_leaf", F.lit(True)).alias("is_leaf")
    )
    
    return result
