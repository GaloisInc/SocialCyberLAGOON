import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from graphframes import GraphFrame

from lagoon.db.connection import get_session
from lagoon.db import schema as sch
from lagoon.ml.config import *

# Create spark session
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setCheckpointDir(f'{os.path.dirname(os.path.dirname(__file__))}/.spark_checkpoints/')


def save_spark_dfs():
    with get_session() as sess:
        entities = sess.query(sch.FusedEntity).all()
        observations = sess.query(sch.FusedObservation).all()

        nodes = spark.createDataFrame(
            data=[(entities[i].id,) for i in range(len(entities))],
            schema = ['id']
            )
        edges = spark.createDataFrame(
            data=[(observations[i].src_id, observations[i].dst_id) for i in range(len(observations))],
            schema = ['src','dst']
            )
    
    foldername = os.path.join(RESULTS_FOLDER, 'graph_ops')
    os.makedirs(foldername, exist_ok=True)
    nodes.toPandas().to_csv(os.path.join(foldername, 'nodes.csv'), index=False)
    edges.toPandas().to_csv(os.path.join(foldername, 'edges.csv'), index=False)


def create_graph():
    """
    Should be run after save_spark_dfs
    """
    nodes = spark.read.csv(os.path.join(os.path.join(RESULTS_FOLDER, 'graph_ops'), 'nodes.csv'), header=True)
    edges = spark.read.csv(os.path.join(os.path.join(RESULTS_FOLDER, 'graph_ops'), 'edges.csv'), header=True)
    return GraphFrame(nodes, edges)

def conncomp():
    g = create_graph()
    result = g.connectedComponents()
    result = result.groupby("component").agg(F.collect_list("id"), F.count("id")).sort("count(id)", ascending=False).drop("component")
    
    foldername = os.path.join(RESULTS_FOLDER, 'graph_ops')
    os.makedirs(foldername, exist_ok=True)
    result.toPandas().to_csv(os.path.join(foldername, 'conncomp.csv'), index_label="component")


if __name__ == "__main__":
    pass