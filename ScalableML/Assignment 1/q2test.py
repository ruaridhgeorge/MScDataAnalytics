### 2C: k means ###############################################################################

import matplotlib.pyplot as plt
import numpy as np
import pyspark
import tempfile
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row, DataFrameReader, SparkSession
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator, RegressionEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.tuning import CrossValidator, CrossValidatorModel, ParamGridBuilder
from pyspark.sql.functions import rand

spark = SparkSession.builder \
    .master("local[4]") \
    .appName("COM6012") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

ratings = spark.read.csv("ScalableML/Data/ml-25m/ratings.csv", header="true", inferSchema="true")
#.cache()
ratings = ratings.na.drop()
(ratings, rest) = ratings.randomSplit([0.001, 0.999])
ratings=ratings.cache()

def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:])]).toDF(['features'])

feat_vecs = transData(ratings)
feat_vecs.show()

k=25
kmeans = KMeans().setK(k)
kmodel = kmeans.fit(feat_vecs)
pred = kmodel.transform(feat_vecs)
pred.show()
centers = kmodel.clusterCenters()
clusterSizes = kmodel.summary.clusterSizes

# LARGEST CLUSTERS?

genome_tags = spark.read.csv("ScalableML/Data/ml-25m/genome-tags.csv", header="true", inferSchema="true")
genome_scores = spark.read.csv("ScalableML/Data/ml-25m/genome-scores.csv", header="true", inferSchema="true")

alsK = ALS(maxIter=10, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
#model = als.fit(ratings)
#ALSModel.itemFactor
# Vectors.dense
#temp_path = tempfile.mkdtemp()
#kmeans_path = temp_path + "/kmeans"
#kmeans.save(kmeans_path)


# 27 tags

spark.stop()

#    .config("spark.local.dir","/fastdata/acq19rg") \