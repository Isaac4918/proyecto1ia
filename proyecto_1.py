## Imports
import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, classification_report, accuracy_score,  roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

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

def remove_zeros_from_column(dataframe, column_name):
    dataframe[column_name].replace(0, np.NaN, inplace=True)
    mean = dataframe[column_name].mean()
    print(mean)
    dataframe[column_name].replace(np.NaN, mean, inplace=True) 

remove_zeros_from_column(dataframe, 'Glucose')
remove_zeros_from_column(dataframe, 'BloodPressure')
remove_zeros_from_column(dataframe, 'SkinThickness')
remove_zeros_from_column(dataframe, 'Insulin')
remove_zeros_from_column(dataframe, 'BMI')
remove_zeros_from_column(dataframe, 'Age')

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
# Standardize features
scaler = StandardScaler()

# Scale the training and testing features usign the same scaler
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs', 'sag', 'newton-cg']
}

# Initialize GridSearchCV with LogisticRegression estimator
grid_search = GridSearchCV(estimator=LogisticRegression(max_iter=1000), param_grid=param_grid, cv=5, scoring='accuracy')

# Print results of all parameter combinations
results = grid_search.fit(X_train_scaled, y_train).cv_results_
for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print("Parameters:", params)
    print("Mean Accuracy:", mean_score)
    # Train model with current parameters
    current_model = LogisticRegression(**params, max_iter=1000)
    current_model.fit(X_train_scaled, y_train)
    # Predictions
    y_pred = current_model.predict(X_test_scaled)
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, current_model.predict_proba(X_test_scaled)[:,1])
    roc_auc = roc_auc_score(y_test, y_pred)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Print metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("=" * 50)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    # Plot heatmap for confusion matrix
    plt.figure()
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

# Get the best estimator
best_model = grid_search.best_estimator_

# Print best parameters and metrics
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)
print("Best Model Accuracy:", accuracy_score(y_test, best_model.predict(X_test_scaled)))
print("Best Model Precision:", precision_score(y_test, best_model.predict(X_test_scaled)))
print("Best Model Recall:", recall_score(y_test, best_model.predict(X_test_scaled)))
print("Best Model Classification Report:")
print(classification_report(y_test, best_model.predict(X_test_scaled)))

# ROC curve for the best model
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test_scaled)[:,1])
roc_auc = roc_auc_score(y_test, best_model.predict(X_test_scaled))

# Plot ROC curve for the best model
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Best Model)')
plt.legend(loc="lower right")
plt.show()

# Confusion matrix for the best model
best_model_conf_matrix = confusion_matrix(y_test, best_model.predict(X_test_scaled))

# Plot heatmap for the best model confusion matrix
plt.figure()
sns.heatmap(best_model_conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
plt.title("Confusion Matrix (Best Model)")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

## K Nearest Neighbors
print("=" * 50)
print("K Nearest Neighbors")

k_values = [3, 5, 7, 9, 11, 13, 15]

print("Unscaled Features")

for k in k_values:
    print(f"K={k}")
    knn = KNeighborsClassifier(k)

    # Fit the model to the training set
    knn.fit(X_train, y_train)

    # Predictions
    y_pred = knn.predict(X_test)

    # Calculate metrics
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for k={k}: {accuracy}")

    # Precision
    precision = precision_score(y_test, y_pred)
    print(f"Precision for k={k}: {precision}")

    # Recall 
    recall = recall_score(y_test, y_pred)
    print(f"Recall for k={k}: {recall}")

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"F1 Score for k={k}: {f1}")

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for k={k}:")
    print(conf_matrix)
    print("\n")

    # Plot heatmap for confusion matrix
    plt.figure()
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
    plt.title(f"Confusion Matrix for k={k}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

print("Scaled Features")

for k in k_values:
    print(f"K={k}")
    knn = KNeighborsClassifier(k)

    # Fit the model to the training set
    knn.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = knn.predict(X_test_scaled)

    # Calculate metrics
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy for k={k}: {accuracy}")

    # Precision
    precision = precision_score(y_test, y_pred)
    print(f"Precision for k={k}: {precision}")

    # Recall 
    recall = recall_score(y_test, y_pred)
    print(f"Recall for k={k}: {recall}")

    # F1 Score
    f1 = 2 * (precision * recall) / (precision + recall)
    print(f"F1 Score for k={k}: {f1}")

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix for k={k}:")
    print(conf_matrix)
    print("\n")

    # Plot heatmap for confusion matrix
    plt.figure()
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
    plt.title(f"Confusion Matrix for k={k}")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

## Neural Network
print("=/" * 50)
print("Neural Network")

# Define the parameter grid
param_grid = {
    'hidden_layer_sizes': [(10,), (20,), (30,), (40,), (50,)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.1],
    'learning_rate': ['constant', 'invscaling', 'adaptive']
}

# Initialize GridSearchCV with MLPClassifier estimator
grid_search = GridSearchCV(estimator=MLPClassifier(max_iter=1000), param_grid=param_grid, cv=5, scoring='accuracy')

# Print results of all parameter combinations
results = grid_search.fit(X_train_scaled, y_train).cv_results_

for mean_score, params in zip(results["mean_test_score"], results["params"]):
    print("Parameters:", params)
    print("Mean Accuracy:", mean_score)
    # Train model with current parameters
    current_model = MLPClassifier(**params, max_iter=1000)
    current_model.fit(X_train_scaled, y_train)
    # Predictions
    y_pred = current_model.predict(X_test_scaled)
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, current_model.predict_proba(X_test_scaled)[:,1])
    roc_auc = roc_auc_score(y_test, y_pred)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    # Print metrics
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc)
    print("Confusion Matrix:")
    print(conf_matrix)
    print("=" * 50)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    # Plot heatmap for confusion matrix
    plt.figure()
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

# Get the best estimator
best_model = grid_search.best_estimator_

# Print best parameters and metrics
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)
print("Best Model Accuracy:", accuracy_score(y_test, best_model.predict(X_test_scaled)))
print("Best Model Precision:", precision_score(y_test, best_model.predict(X_test_scaled)))
print("Best Model Recall:", recall_score(y_test, best_model.predict(X_test_scaled)))
print("Best Model Classification Report:")
print(classification_report(y_test, best_model.predict(X_test_scaled)))

# ROC curve for the best model
fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test_scaled)[:,1])