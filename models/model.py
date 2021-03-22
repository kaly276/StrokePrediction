import etl_job as etl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def main():
    # Execute ETL pipeline
    df = etl.start()

    # Train the logistic regression model with selected features
    model = LogisticRegression(max_iter=5000)

    df['stroke'] = data['stroke']

    train, test = train_test_split(df, test_size=0.10, random_state=42)

    X_train, y_train = train.iloc[:,:len(df.columns)-1], train['stroke']
    X_test, y_test = test.iloc[:,:len(df.columns)-1], test['stroke']
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)

    print('Training Accuracy: ', model.score(X_train, y_train))
    print('Test Accuracy: ', model.score(X_test, y_test))