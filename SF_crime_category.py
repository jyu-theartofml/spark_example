from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import atexit
import pandas as pd
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml.pipeline import Pipeline

if __name__ == "__main__":
	#these can be before the if __name__ line
	conf=SparkConf()
	sc = SparkContext(conf=conf.setMaster('local[*]'))
	sqlContext=SQLContext(sc)
	sc.setLogLevel("WARN")
    # Create a SparkSession (Note, the config section is only for Windows!)
	spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/Temp").appName("tutorial").getOrCreate()
    
	
	raw_data=spark.read.csv("C:/Users/yuy/Desktop/kaggle/SF crime/train.csv", header="true", inferSchema="true",mode="DROPMALFORMED")
	column_vec_in = ['DayOfWeek', 'PdDistrict']
	column_vec_out = ['DW_catVec','PD_catVec']
	
	indexers = [StringIndexer(inputCol=x, outputCol=x+'_tmp') 
	for x in column_vec_in ]
	encoders = [OneHotEncoder(dropLast=False, inputCol=x+"_tmp", outputCol=y)
	for x,y in zip(column_vec_in, column_vec_out)]
		
	#list	
	tmp = [[i,j] for i,j in zip(indexers, encoders)]
	tmp = [i for sublist in tmp for i in sublist]
	
	#list col names to include in features
	col_new=['DW_catVec','PD_catVec','X','Y']
	
	#if a number in 'value' column is zero, it will not count towards the vectorized features
	assembler_features = VectorAssembler(inputCols=col_new, outputCol='features')
	labelIndexer = StringIndexer(inputCol='Category', outputCol="label")
	tmp += [assembler_features, labelIndexer]
	pipeline = Pipeline(stages=tmp)
	allData = pipeline.fit(raw_data).transform(raw_data)
	##allData.select('features','label').show(10)

	#split data into training and test set
	(trainingData, testData) = allData.randomSplit([0.8, 0.2])
	
	# Train a RandomForest model. Note-> only RF and Decision supports multiclass label as of September 2016
	#set maxDepth=22
	rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=20, featureSubsetStrategy="sqrt",maxDepth=22, maxMemoryInMB=400,
	seed=123)
	model=rf.fit(trainingData)
	
	#make prediction on test set
	predictions=model.transform(testData)
	#the .transform spits out a 'prediction' column
	predictions.select("prediction","label").show(5)
	
	
	# Select (prediction, true label) and compute test error
	evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
	accuracy = evaluator.evaluate(predictions)
	print("Test Error = %g" % (1.0 - accuracy))
    # Stop the session, this worked!!!!
	
	spark.stop()
