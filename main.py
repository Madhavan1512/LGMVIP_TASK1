import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('Iris.csv')
data_points = data.iloc[:, 1:5]
labels = data.iloc[:, 5]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_points  , labels , test_size=0.2)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
Standard_obj = StandardScaler()
Standard_obj.fit(x_train)
x_train_std = Standard_obj.transform(x_train)
x_test_std = Standard_obj.transform(x_test)
from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
svm.fit(x_train_std, y_train)
print('Training data accuracy {:.2f}'.format(svm.score(x_train_std, y_train)*100))
print('Testing data accuracy {:.2f}'.format(svm.score(x_test_std, y_test)*100))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 7, p = 2, metric='minkowski')
knn.fit(x_train_std,y_train)
print('Training data accuracy {:.2f}'.format(knn.score(x_train_std, y_train)*100))
print('Testing data accuracy {:.2f}'.format(knn.score(x_test_std, y_test)*100))
from sklearn import tree
decision_tree = tree.DecisionTreeClassifier(criterion='gini')
decision_tree.fit(x_train_std, y_train)
print('Training data accuracy {:.2f}'.format(decision_tree.score(x_train_std, y_train)*100))
print('Testing data accuracy {:.2f}'.format(decision_tree.score(x_test_std, y_test)*100))
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier()
random_forest.fit(x_train_std, y_train)
print('Training data accuracy {:.2f}'.format(random_forest.score(x_train_std, y_train)*100))
print('Testing data accuracy {:.2f}'.format(random_forest.score(x_test_std, y_test)*100))

