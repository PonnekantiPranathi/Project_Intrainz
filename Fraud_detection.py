#Importing the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Reading the dataset from a '.csv' file . This file should be placed under the same location where the current file is present i.e. the same directory
data = pd.read_csv("payment_dataset.csv")
# Printing the top elememts from the dataset
print(data.head())
# getting the size of the datset
print(data.shape)
# Checking the null-values in the data set
print(data.isnull().sum())
# Getting the details of the count of transcation types
print(data.type.value_counts())

data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})

# Convert the target variable to numeric format
data["isFraud"] = pd.factorize(data["isFraud"])[0]

print(data.head())

# Splitting data to work on our model
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data["isFraud"])

# TRaining the ML model
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# Testing the model
features = np.array([[1, 8900.2, 8990.2, 0.0]])
print(model.predict(features))


# This model detects fraud in the online payments using the given datasets