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
# Data Visualization
outcome = dataframe.Survived
fare = dataframe.Fare
age = dataframe.Age
parch = dataframe.Parch

female_survivors = len(dataframe.query("Sex_female==1 and Survived==1"))
female_deaths = len(dataframe.query("Sex_female==1 and Survived==0"))
all_females = dataframe.Sex_female.value_counts()[1]

male_survivors = len(dataframe.query("Sex_female==0 and Survived==1"))
male_deaths = len(dataframe.query("Sex_female==0 and Survived==0"))
all_males = dataframe.Sex_female.value_counts()[0]

# Class 1 Tickets
c1_survivors = len(dataframe.query("Pclass_1==1 and Survived==1"))
c1_deaths = len(dataframe.query("Pclass_1==1 and Survived==0"))

# Class 2 Tickets
c2_survivors = len(dataframe.query("Pclass_2==1 and Survived==1"))
c2_deaths = len(dataframe.query("Pclass_2==1 and Survived==0"))

# Class 3 Tickets
c3_survivors = len(dataframe.query("Pclass_3==1 and Survived==1"))
c3_deaths = len(dataframe.query("Pclass_3==1 and Survived==0"))

# Bar Plots
x = np.array(["Did not survive", "Survived"])
y_female = np.array([female_deaths,female_survivors])
y_male = np.array([male_deaths,male_survivors])
p1 = plt.bar(x, y_female, color='hotpink')
p2 = plt.bar(x, y_male, bottom=y_female, color='lightseagreen')
plt.title('Survivor amount by gender')
plt.legend((p1[0], p2[0]), ('Female', 'Male'))
plt.show()

y_class_1 = np.array([c1_deaths,c1_survivors])
y_class_2 = np.array([c2_deaths,c2_survivors])
y_class_3 = np.array([c3_deaths,c3_survivors])
p3 = plt.bar(x, y_class_1, color='darkorange')
p4 = plt.bar(x, y_class_2, bottom=y_class_1, color ='gold')
p5 = plt.bar(x, y_class_3, bottom=y_class_2,color='cornflowerblue')
plt.title('Survivor amount by ticket class')
plt.legend((p3[0], p4[0], p5[0]), ('First Class', 'Second Class', 'Third Class'))
plt.show()

# Scatter plot
scatter_plot2 = plt.scatter(age, fare, c=outcome)
plt.colorbar(scatter_plot2)
plt.xlabel("age")
plt.ylabel("fare")
plt.show()

## Dataset splitting
X, y = dataframe[["Age", "SibSp", "Parch", "Fare", "Pclass_1", "Pclass_2", "Pclass_3", "Sex_female", "Sex_male", "Embarked_C", "Embarked_Q", "Embarked_S"]], dataframe["Survived"]
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