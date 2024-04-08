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
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

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

# neural network with torch running on the gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).to(device)

# Create a Dataset from the tensors
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create a DataLoader from the dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network model
class Net(nn.Module):
    def __init__(self, hidden_layer_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(X_train.shape[1], hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
    
# Define a loss function
criterion = nn.BCELoss()

# Define the hyperparameters based on param_grid

layers = [5, 10, 15, 20]
alphas = [0.01, 0.1, 1, 10]

for layer in layers:
    for alpha in alphas:
        print("=" * 50)
        print(f"Hidden Layer Size: {layer}")
        print(f"Learning Rate: {alpha}")
        print("\n")
        model = Net(layer).to(device)
        optimizer = optim.Adam(model.parameters(), lr=alpha)

        # Train the model
        epochs = 100

        for epoch in range(epochs):
            model.train()
            for i, data in enumerate(train_loader):
                inputs, labels = data
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.view(-1, 1))
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch} Loss: {loss.item()}")

        # Evaluate the model
        model.eval()
        y_pred = model(X_test_tensor).cpu().detach().numpy()
        y_pred = np.where(y_pred > 0.5, 1, 0)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")

        precision = precision_score(y_test, y_pred)
        print(f"Precision: {precision}")

        recall = recall_score(y_test, y_pred)
        print(f"Recall: {recall}")

        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall)
        print(f"F1 Score: {f1}")

        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(conf_matrix)

        # Plot heatmap for confusion matrix
        plt.figure()
        sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.show()