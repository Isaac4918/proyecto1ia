## Imports
import pandas
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

## Data loading
dataframe = pandas.read_csv("data/diabetes.csv")
## Data analysis
### Outliers and invalid values
def interquartile_range(variable, dataframe):
    Q1 = dataframe[variable].quantile(0.25)
    Q3 = dataframe[variable].quantile(0.75)
    return (Q1, Q3, Q3 - Q1)

def remove_outliers (dataframe):
    threshold = 1.5
    for i in range(len(dataframe.columns)-1):
        Q1 = interquartile_range(dataframe.columns[i], dataframe)[0]
        Q3 = interquartile_range(dataframe.columns[i], dataframe)[1]
        IQR = interquartile_range(dataframe.columns[i], dataframe)[2]
        outliers = dataframe[(dataframe[dataframe.columns[i]] < Q1 - threshold * IQR) | (dataframe[dataframe.columns[i]] > Q3 + threshold * IQR)]
        dataframe = dataframe.drop(outliers.index)
    return dataframe

dataframe = dataframe.dropna() # drop empty values
dataframe = remove_outliers(dataframe) # drop outliers

