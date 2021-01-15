# Question 2

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
    .master("local[8]") \
    .appName("COM6012") \
    .config("spark.local.dir","/fastdata/acq19rg") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")

# Monkey patch the cross validator function to output each evaluator and the averages. Reference: https://stackoverflow.com/questions/38874546/spark-crossvalidatormodel-access-other-models-than-the-bestmodel (Mack)
class CrossValidatorEdit(CrossValidator):
#
    def _fit(self, dataset):
        est = self.getOrDefault(self.estimator)
        epm = self.getOrDefault(self.estimatorParamMaps)
        numModels = len(epm)
#
        eva = self.getOrDefault(self.evaluator)
        metricName = eva.getMetricName()
#
        nFolds = self.getOrDefault(self.numFolds)
        seed = self.getOrDefault(self.seed)
        h = 1.0 / nFolds
#
        randCol = self.uid + "_rand"
        df = dataset.select("*", rand(seed).alias(randCol))
        metrics = [0.0] * numModels
#
        for i in range(nFolds):
            foldNum = i + 1
            print("Comparing models on fold %d" % foldNum)
#
            validateLB = i * h
            validateUB = (i + 1) * h
            condition = (df[randCol] >= validateLB) & (df[randCol] < validateUB)
            validation = df.filter(condition)
            train = df.filter(~condition)
#
            for j in range(numModels):
                paramMap = epm[j]
                model = est.fit(train, paramMap)
                # TODO: duplicate evaluator to take extra params from input
                metric = eva.evaluate(model.transform(validation, paramMap))
                metrics[j] += metric
#
                avgSoFar = metrics[j] / foldNum
                print("params: %s\t%s: %f\tavg: %f" % (
                    {param.name: val for (param, val) in paramMap.items()},
                    metricName, metric, avgSoFar))
#
        if eva.isLargerBetter():
            bestIndex = np.argmax(metrics)
        else:
            bestIndex = np.argmin(metrics)
#
        bestParams = epm[bestIndex]
        bestModel = est.fit(dataset, bestParams)
        avgMetrics = [m / nFolds for m in metrics]
        bestAvg = avgMetrics[bestIndex]
        print("Best model:\nparams: %s\t%s: %f" % (
            {param.name: val for (param, val) in bestParams.items()},
            metricName, bestAvg))
#
        return self._copyValues(CrossValidatorModel(bestModel, avgMetrics))

### 2A: 3-fold cross-validation ##################################################################
print('Question 2A')
ratings = spark.read.csv("ScalableML/Data/ml-25m/ratings.csv", header="true", inferSchema="true").cache()
ratings = ratings.na.drop()
#(ratings, rest) = ratings.randomSplit([0.001, 0.999]). Smaller data set used for testing

# Build a parameter grid so we can fit at least 3 ALS models
als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
paramGrid = ParamGridBuilder() \
    .addGrid(als.maxIter, [5, 15]) \
    .addGrid(als.regParam, [0.1, 1]) \
    .build()

# Defining our RMSE and MAE evaluators
eval_rmse = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
eval_mae = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction")

# Defining our Cross Validator functions that will output the RMSE and MAE values for each ALS model and each fold.
cv_rmse = CrossValidatorEdit(estimator=als, estimatorParamMaps=paramGrid, evaluator=eval_rmse, numFolds=3)
cv_mae = CrossValidatorEdit(estimator=als, estimatorParamMaps=paramGrid, evaluator=eval_mae, numFolds=3)

# Fit the models
cv_rmse_model = cv_rmse.fit(ratings)
cv_mae_model = cv_mae.fit(ratings)

# As found from output

rmse_mean = [0.814032,0.805254,1.322023]
mae_mean = [0.630661,0.621696,1.152988]
rmse_sd = [0.0002577,0.0001113,0.0002874]
mae_sd = [0.0003699,0.0001892,0.0002495]

mean_labels = ['rmse_5_0.1', 'rmse_15_0.1', 'rmse_15_1', 'mae_5_0.1', 'mae_15_0.1', 'mae_15_1'] 
x = np.arange(len(mean_labels))

plt.bar(x,rmse_mean+mae_mean)
plt.xticks(x, mean_labels)
plt.title('Mean of MAE and RMSE metrics for 3 ALS models')
plt.savefig("/home/acq19rg/ScalableML/means_bar.png")

sd_labels = ['rmse_5_0.1', 'rmse_15_0.1', 'rmse_15_1', 'mae_5_0.1', 'mae_15_0.1', 'mae_15_1'] 
y = np.arange(len(sd_labels))

plt.bar(y,rmse_sd+mae_sd)
plt.xticks(y, sd_labels)
plt.title('SD of MAE and RMSE metrics for 3 ALS models')
plt.savefig("/home/acq19rg/ScalableML/sd_bar.png")



# mean and sd of rmse and mae for each version of als in one figure

### 2C: k means ###############################################################################
print('Question 2C')

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
clusterSizes.sort()[:3]
print('Top 3 Cluster Sizes')

# LARGEST CLUSTERS?

genome_tags = spark.read.csv("ScalableML/Data/ml-25m/genome-tags.csv", header="true", inferSchema="true")
genome_scores = spark.read.csv("ScalableML/Data/ml-25m/genome-scores.csv", header="true", inferSchema="true")

alsK = ALS(maxIter=10, regParam=0.1, userCol="userId", itemCol="movieId", ratingCol="rating",
          coldStartStrategy="drop")
modelK = alsK.fit(ratings)
dfItemFactors = modelK.itemFactors

spark.stop()




