import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

DF = pd.read_csv("/Users/parmisazizi/Downloads/Coding Documents/ML and AI/cars.csv")
print(DF.head())