## Imports
import pandas
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, classification_report

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


### Balance evaluation
negative = dataframe.Outcome.value_counts()[0]
positive = dataframe.Outcome.value_counts()[1]

x = np.array(["Negative", "Positive"])
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
# Data Visualization
blood_pressure = dataframe.BloodPressure
age = dataframe.Age
glucose = dataframe.Glucose
outcome = dataframe.Outcome
bmi = dataframe.BMI
insulin = dataframe.Insulin

scatter_plot = plt.scatter(glucose, age, c=outcome)
plt.colorbar(scatter_plot)
plt.xlabel("glucose")
plt.ylabel("age")
plt.show()

scatter_plot2 = plt.scatter(blood_pressure, glucose, c=outcome)
plt.colorbar(scatter_plot2)
plt.xlabel("blood pressure")
plt.ylabel("glucose")
plt.show()

scatter_plot3 = plt.scatter(insulin, glucose, c=outcome)
plt.colorbar(scatter_plot3)
plt.xlabel("insulin")
plt.ylabel("glucose")
plt.show()

scatter_plot4 = plt.scatter(bmi, glucose, c=outcome)
plt.colorbar(scatter_plot4)
plt.xlabel("bmi")
plt.ylabel("glucose")
plt.show()
## Dataset splitting
X, y = dataframe[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]], dataframe["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=outcome)
print("training set data proportion: ",y_train.value_counts()[0]/y_train.value_counts()[1])
print("testing set data proportion: ",y_test.value_counts()[0]/y_test.value_counts()[1])

## Logistic Regression
# Initialize the logistic regression model
log_reg_model = LogisticRegression(max_iter=1000) #Check!!
# Fit the model to the training set
log_reg_model.fit(X_train, y_train)

# Predictions
y_pred = log_reg_model.predict(X_test)

## Metrics Evaluations

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Precision
precision = precision_score(y_test, y_pred)
print("Precision:", precision)

# Recall
recall = recall_score(y_test, y_pred)
print("Recall:", recall)

# Classification Report

print(classification_report(y_test, y_pred))
