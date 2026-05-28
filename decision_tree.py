import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing

DF1 = pd.read_csv("/Users/parmisazizi/Downloads/Coding Documents/ML and AI/cars.csv")
print(DF1.head())

#Label encoding

label_encoder = preprocessing.LabelEncoder()

DF1["buying"] = label_encoder.fit_transform(DF1["buying"])
DF1["maint"] = label_encoder.fit_transform(DF1["maint"])
DF1["doors"] = label_encoder.fit_transform(DF1["doors"])
DF1["persons"] = label_encoder.fit_transform(DF1["persons"])
DF1["lug_boot"] = label_encoder.fit_transform(DF1["lug_boot"])
DF1["safety"] = label_encoder.fit_transform(DF1["safety"])
DF1["class"] = label_encoder.fit_transform(DF1["class"])

print(DF1.head())

x = DF1[["buying","maint","doors","persons","lug_boot","safety"]]
y = DF1[["class"]]

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size = 0.2, random_state = 13)
''
decision_model = DecisionTreeClassifier(criterion = "entropy", random_state = 5)
decision_model.fit(xtrain,ytrain)


ypred = decision_model.predict(xtest)
matrix = confusion_matrix(ytest,ypred)
sns.heatmap(matrix,annot = True)

plt.title("confusion matrix")
plt.xlabel("predicted")
plt.ylabel("actual")
plt.show()

print(classification_report(ytest,ypred))