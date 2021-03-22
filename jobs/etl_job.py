# pylint: disable: import-error
import numpy as np
import pandas as pd

from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.linalg import Vector, SparseVector, DenseVector
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.window import Window

def start():
    '''
    Main ETL script
    '''
    spark = SparkSession.builder.getOrCreate()

    # Execute ETL pipeline
    data = extract_data(spark)
    data_cleaned = clean_data(data)
    data_transformed = transform_data(data_cleaned)
    load_data(data_transformed)

    # Return pandas DataFrame for analysis and training
    return data_transformed

def extract_data(spark):
    '''
    Load data from csv
    Input:
        (spark) Spark session object
    Output:
        Spark DataFrame
    '''
    return spark.read.csv('../data/healthcare-dataset-stroke-data.csv', header=True)

def clean_data(data):
    '''
    Filter rows with null values and reformat data
    Input:
        (data) Spark DataFrame
    Output:
        Cleaned DataFrame
    '''
    # Convert all string values to lowercase
    # Ensure all numeric columns are numeric values
    numeric_columns = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level']
    string_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    cleaned = data.select(*(col(c).cast('double').alias(c) for c in numeric_columns),
                          *(lower(col(c)).alias(c) for c in string_columns))
    cleaned.show()

    return cleaned

def transform_data(data):
    '''
    Transform dataset
    Input:
        (data) Spark DataFrame
    Output:
        Transformed DataFrame
    '''
    # Save features that are already numerical
    data_num = data.select('age', 'hypertension', 'heart_disease', 'avg_glucose_level')

    # Use StringIndexer to transform categories to numerical values
    inputs = ['gender', 'ever_married', 'work_type',
                'Residence_type', 'smoking_status']
    outputs = ['gender_i', 'ever_married_i', 'work_type_i',
                'Residence_type_i', 'smoking_status_i']
    indexer = StringIndexer(inputCols=inputs, outputCols=outputs)
    indexed = indexer.fit(data).transform(data)
    indexed = indexed.select(*outputs)

    # Use OneHotEncoder to map the numerical values to vectors
    encoder = OneHotEncoder(inputCols=indexed.columns, outputCols=inputs)
    encoded = encoder.fit(indexed).transform(indexed)
    encoded = encoded.select(*inputs)

    # Combine numerical features into a single DataFrame
    w = Window.orderBy(lit(1))
    data_num = data_num.withColumn('rn', row_number().over(w)-1)
    encoded = encoded.withColumn('rn', row_number().over(w)-1)
    combined_data = data_num.join(encoded, ['rn']).drop('rn')

    # Combine features into a single feature column using VectorAssembler
    assembler = VectorAssembler(inputCols=combined_data.columns, outputCol='features')
    assembled = assembler.transform(combined_data)

    # Convert sparse vectors to NumPy arrays
    assembled = assembled.toPandas()
    assembled['features'] = assembled['features'].apply(np.asarray)
    
    # Transform feature arrays to columns
    new_columns = range(len(df['features'][0]))
    new_data = assembled.features.to_list()
    assembled = assembled.DataFrame(new_data, columns=new_columns)

    return assembled

def load_data(data):
    '''
    Collect data locally and write to parquet
    Input:
        (data) Spark DataFrame
    '''
    data.write.mode('overwrite').parquet('../data/transformed_healthcare_data')

# Entry point for PySpark ETL application
if __name__ == '__start__':
    start()