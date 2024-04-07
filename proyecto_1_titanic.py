## Imports
import pandas
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

## Data loading
dataframe = pandas.read_csv("data/titanic.csv")
## Data analysis
### One-hot encoding 
# One-hot encoding categorical data for 'Pclass' and 'Sex' columns
dataframe = pandas.get_dummies(dataframe, columns=['Pclass', 'Sex','Embarked'], dtype=int)
### Dropping columns
# Drop unnecessary columns
dataframe.drop(['Cabin','PassengerId','Name','Ticket'], axis='columns', inplace=True)
### Outliers and invalid values
def interquartile_range(variable, dataframe):
    Q1 = dataframe[variable].quantile(0.25)
    Q3 = dataframe[variable].quantile(0.75)
    return (Q1, Q3, Q3 - Q1)


def remove_outliers(dataframe: pandas.DataFrame, dataframe_columns):
    threshold = 1.5

    for i in range(len(dataframe_columns)):
        (Q1, Q3, IQR) = interquartile_range(dataframe_columns[i], dataframe)
        low_outliers = dataframe[dataframe[dataframe_columns[i]] < Q1 - threshold * IQR]
        dataframe = dataframe.drop(low_outliers.index)
        high_outliers = dataframe[dataframe[dataframe_columns[i]] > Q3 + threshold * IQR]
        dataframe = dataframe.drop(high_outliers.index)
    return dataframe

def remove_nan_from_column(dataframe, column_name):
    mean = dataframe[column_name].mean()
    dataframe[column_name].replace(np.NaN, mean, inplace=True) 

remove_nan_from_column(dataframe, 'Age')
dataframe = remove_outliers(dataframe, ['Age', 'Fare']) # drop outliers
### Balance evaluation
negative = dataframe.Survived.value_counts()[0]
positive = dataframe.Survived.value_counts()[1]

print(negative)
print(positive)

x = np.array(["Did not survive", "Survived"])
y = np.array([negative,positive])
plt.bar(x,y)
plt.show()
### Pairwise correlation
#### Kendall
dataframe.corr(method='kendall', numeric_only=False)
#### Pearson
dataframe.corr(method='pearson', min_periods=1, numeric_only=False)
#### Spearman
dataframe.corr(method='spearman', min_periods=1, numeric_only=False)