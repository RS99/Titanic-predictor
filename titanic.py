import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
data = pd.read_csv("test.csv")
print(data, "\n")
data.drop(["PassengerId", "Name", "Ticket", "Fare"], axis = 1, inplace = True)
print(data, "\n")
X = data.drop("Survived", axis = 1)
y = data["Survived"]
print(X, "\n")
print(y, "\n")
print(X.isna().sum(),"\n")
print(X.dtypes)
categorical_features = ["Cabin", "Embarked"]
si_categorical = SimpleImputer(strategy = "constant", fill_value = "Missing")
numerical_features = ["Age"]
si_numerical = SimpleImputer(strategy="mean")
transformer = ColumnTransformer(transformers =[
    ("categorical_transformation", si_categorical, categorical_features),
    ("numerical_transformation", si_numerical, numerical_features)],
    remainder="passthrough"
)
X_transformed = transformer.fit_transform(X)
X_transformed = pd.DataFrame(X_transformed, columns= ["Cabin", "Embarked", "Age", "Pclasss", "Sex", "SibSp", "Parch"])
print(X_transformed.isna().sum(), "\n")
print(X_transformed.head(), "\n")
print(X_transformed["Cabin"].nunique(), "\n")
for i in range(len(X_transformed["Cabin"])):
    X_transformed["Cabin"][i] = X_transformed["Cabin"][i][: 1]

print(X_transformed, "\n")
print(X_transformed["Cabin"].nunique(), "\n")
print(X_transformed["Embarked"].nunique(), "\n")
print(X_transformed["Sex"].nunique(), "\n")
print(X_transformed.dtypes, "\n")
features = ["Cabin", "Embarked", "Sex"]
encoder = OneHotEncoder(handle_unknown="ignore")
transformer2 = ColumnTransformer(transformers=[
    ("onehotencoding", encoder, features)],
    remainder="passthrough"
)
X_final = transformer2.fit_transform(X_transformed)
X_final = pd.DataFrame(X_final)
x_train, x_test, y_train, y_test = train_test_split(X_final, y, test_size=0.4)
print("shape, \n", x_train.shape, x_test.shape, y_train.shape, y_test.shape, "\n")

clf = RandomForestClassifier()
print(clf.fit(x_train, y_train), "\n")

y_pred = clf.predict(x_test)
print("y_pred \n", y_pred,"\n")

print(round(accuracy_score(y_test, y_pred) * 100), 3, "\n")

#print(round(clf.score(x_test, y_test)*100), 3, "\n")
y_pred = pd.DataFrame(y_pred)
print(y_pred, "\n")
y_pred.to_csv("Predicted_Titanic.csv", index=False)
#*********************************************************************************************
x_test = pd.DataFrame(x_test)
x_test.to_csv("x_test.csv", index=False)

x_train = pd.DataFrame(x_train)
x_train.to_csv("x_train.csv", index=False)

y_test = pd.DataFrame(y_test)
y_test.to_csv("y_test.csv", index=False)

y_train = pd.DataFrame(y_pred)
y_train.to_csv("y_train.csv", index=False)