import pandas as pd
from sklearn.datasets import load_iris
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -===-=-=-=-=-= A Basic Example =-=-==-=-=-=-=-
# Step 1: Prepare and import the data
iris = load_iris()
# Step 2: Split the data into training and test sets. 
X, y = iris.data[:, :2], iris.target #X is the data, y is the target
# In this case, 80% of the data is used for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=33, test_size=.2)
#Step 3: Preprocess the data
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# Step 4: Create the model
knn = neighbors.KNeighborsClassifier(n_neighbors=5)
# Step 5: Train the model
knn.fit(X_train, y_train)
# Step 6: Evaluate the model, get the accuracy
y_pred = knn.predict(X_test)
# Step 7: Print the accuracy
acc = accuracy_score(y_test, y_pred)
print(acc)