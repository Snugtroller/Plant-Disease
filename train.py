import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import numpy as np

data_dict=pickle.load(open("leaf_data.pickle","rb"))
print(data_dict)