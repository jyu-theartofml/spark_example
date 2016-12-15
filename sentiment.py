from pyspark import SparkConf 


from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size
from pyspark.sql.types import *
import atexit
import pandas as pd
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes, OneVsRest, MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorIndexer, VectorAssembler
from pyspark.ml.feature import HashingTF, IDF, Tokenizer,CountVectorizer, Word2Vec
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover
from pyspark.ml.pipeline import Pipeline

if __name__ == "__main__":
	#these can be before the if __name__ line
	conf=SparkConf()
	sc = SparkContext(conf=conf.setMaster('local[*]'))
	sqlContext=SQLContext(sc)
	sc.setLogLevel("WARN")
    # Create a SparkSession (Note, the config section is only for Windows!)
	spark = SparkSession.builder.config("spark.sql.warehouse.dir", "file:///C:/Temp").appName("tutorial").getOrCreate()
    
	#for tsv files, sqlcontext.read.load format recognizes header columns, either one of these below works
	
	#raw_data=sqlContext.read.format('com.databricks.spark.csv').options( header="true", delimiter="\t", inferSchema="true",mode="DROPMALFORMED").load("C:/Users/yuy/Desktop/kaggle/movie/train.tsv")
	raw_data=sqlContext.read.load("C:/Users/yuy/Desktop/kaggle/movie/train.tsv", format='com.databricks.spark.csv', header="true", delimiter="\t", inferSchema="true",mode="DROPMALFORMED")
	tokenizer=Tokenizer(inputCol="Phrase", outputCol="words")
	wordsDF=tokenizer.transform(raw_data)
	
	
	#dont remove stop words because in test file there's stop words to be predicted
	
	#count vectorized the word tokens
	cv=CountVectorizer(inputCol="words", outputCol="features")
	word_vec=cv.fit(wordsDF).transform(wordsDF)
	#word_vec.select("features").show(3)
	#IDF
	idf = IDF(inputCol="features", outputCol="IDF_features")
	idfModel = idf.fit(word_vec).transform(word_vec)
	
	labelIndexer = StringIndexer(inputCol='Sentiment', outputCol="label")
	dataset=labelIndexer.fit(idfModel).transform(idfModel)
	dataset.show(3)
	dataset2=dataset.select("IDF_features","label")
	dataset2.show(3)
	(trainingData, testData) = dataset2.randomSplit([0.8, 0.2], 24)
	
	# create the trainer and set its parameters
	#use multinomial when using tf-idf, bernoulli requires binary values 0,1
	nb = NaiveBayes(smoothing=1.0, modelType="multinomial", featuresCol="IDF_features")
	model_nb = nb.fit(trainingData)
	# compute accuracy on the test set
	result = model_nb.transform(testData)
	predictionAndLabels = result.select("prediction", "label")
	evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
	print("Naive Bayes Accuracy: " + str(evaluator.evaluate(predictionAndLabels)))
	
	#try one-vs-rest using LR as base classifier
	lr = LogisticRegression(maxIter=300, tol=1E-3, fitIntercept=True)
	ovr=OneVsRest(classifier=lr, featuresCol="IDF_features")
	model_ovr= ovr.fit(trainingData)
	result_ovr = model_ovr.transform(testData)
	predictionAndLabels_ovr = result_ovr.select("prediction", "label")
	evaluator_ovr = MulticlassClassificationEvaluator(metricName="accuracy")
	print("Logistic Regression Accuracy: " + str(evaluator_ovr.evaluate(predictionAndLabels_ovr)))
	
	#try word2vec
	##dataset_wv=labelIndexer.fit(wordsDF).transform(wordsDF)
	##w2v=Word2Vec(vectorSize=5, minCount=0, inputCol="words",outputCol="transformed_vector")
	
	##model_w2v=w2v.fit(dataset_wv)
	#transform into feature vector for learning algorithm downstream
	##w2v=model_w2v.transform(dataset_wv)
	##dataset3=w2v.select("transformed_vector", "label")
	##(train, test)=dataset3.randomSplit([0.8, 0.2], 24)
	#naive bayes requires non-negative feature values, so use RF
	##rf = RandomForestClassifier(labelCol="label", featuresCol="transformed_vector", numTrees=40, featureSubsetStrategy="sqrt",maxDepth=12, maxMemoryInMB=1500,
	##seed=123)
	##model_rf = rf.fit(train)
	##result_rf = model_rf.transform(test)
	##predictionAndLabels_rf = result_rf.select("prediction", "label")
	##evaluator_rf= MulticlassClassificationEvaluator(metricName="accuracy")
	##print("Word2vec + RF Accuracy: " + str(evaluator_rf.evaluate(predictionAndLabels_rf)))
	
	#merge test_data and training
	test_data=sqlContext.read.load("C:/Users/yuy/Desktop/kaggle/movie/test.tsv", format='com.databricks.spark.csv', header="true", delimiter="\t", inferSchema="true",mode="DROPMALFORMED")
	tokenizer=Tokenizer(inputCol="Phrase", outputCol="words")
	testDF=tokenizer.transform(test_data)
	cv=CountVectorizer(inputCol="words", outputCol="features")
	test_cv=cv.fit(testDF).transform(testDF)
	idf = IDF(inputCol="features", outputCol="IDF_features")
	test_idf = idf.fit(test_cv).transform(test_cv).select("IDF_features")
	test_idf.show(3)
	pred=model_ovr.transform(test_idf)
	
	#prediction_test=pred.select("prediction").rdd.map(lambda r:r[0]).collect()
	#pred_csv=pd.DataFrame()
	#pred_csv['PhraseId']=test_data.select('PhraseId').rdd.map(lambda r: r[0]).collect()
	#pred_csv['Sentiment']=prediction_test
	#pred_csv.to_csv("submission(1).csv", index=False)
	spark.stop()
