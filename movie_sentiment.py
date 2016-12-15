from pyspark import SparkConf 
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, udf, lit
from pyspark.sql.types import *
import atexit
import pandas as pd
from pyspark.ml.classification import LogisticRegression, NaiveBayes, OneVsRest, DecisionTreeClassifier
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
	test_data=sqlContext.read.load("C:/Users/yuy/Desktop/kaggle/movie/test.tsv", format='com.databricks.spark.csv', header="true", delimiter="\t", inferSchema="true",mode="DROPMALFORMED")
	
	train_raw=raw_data.select("PhraseId", "SentenceId", "Phrase")
	train_size=train_raw.count()
	test_size=test_data.count()
	#concate the two data set for tokenization and countvector
	total_data=train_raw.union(test_data)
	
	#regex tokenization
	tokenizer=RegexTokenizer(inputCol="Phrase", outputCol="words",pattern="\\w+", gaps=False)
	wordsDF=tokenizer.transform(total_data)
	#wordsDF.select("words").show(3, False)
	
	#dont remove stop words because in test file there's stop words to be predicted
	
	#count vectorized the word tokens
	cv=CountVectorizer(inputCol="words", outputCol="features")
	word_vec=cv.fit(wordsDF).transform(wordsDF).select("features", "PhraseId")
	#word_vec.select("features").show(3)
	
	#IDF
	idf = IDF(inputCol="features", outputCol="IDF_features")
	idfModel = idf.fit(word_vec).transform(word_vec)
	idf_data=idfModel.select("PhraseId","features", "IDF_features")
	#Note that this method should only be used if the resulting Pandas’s DataFrame is expected to be small, 
	#as all the data is loaded into the driver’s memory
	x=idf_data.toPandas()
	
	train=x.iloc[:train_size, :]
	test=x.iloc[train_size:, :]
	
	
	train_data_idf=spark.createDataFrame(train)
	test_data_idf=spark.createDataFrame(test)
	#join training data with target values
	target=raw_data.select("PhraseId", "Sentiment")
	train_data=train_data_idf.join(target, ["PhraseId"])
	
	
	(training_set, val_set) = train_data.randomSplit([0.8, 0.2], 24)
	
	evaluator = MulticlassClassificationEvaluator(metricName="accuracy", labelCol="Sentiment")
	
	#try one-vs-rest using LR as base classifier
	lr = LogisticRegression(maxIter=100, tol=1E-1, fitIntercept=True)
	ovr=OneVsRest(classifier=lr, featuresCol="IDF_features", labelCol="Sentiment")
	model_ovr= ovr.fit(training_set)
	result_ovr = model_ovr.transform(val_set)
	predictionAndLabels_ovr = result_ovr.select("prediction", "Sentiment")
	print("Logistic Regression Accuracy: " + str(evaluator.evaluate(predictionAndLabels_ovr)))
	
	
	
	pred=model_ovr.transform(test_data_idf)
	prediction_test=pred.select("prediction").rdd.map(lambda r:r[0]).collect()
	
	pred_csv=pd.DataFrame()
	pred_csv['PhraseId']=test_data.select('PhraseId').rdd.map(lambda r: int(r[0])).collect()
	pred_csv['Sentiment']=prediction_test
	pred_csv.to_csv("submission(3).csv", index=False)
	spark.stop()
