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