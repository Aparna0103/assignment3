import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer

def handle_data(Test_set):
    
    for column in Test_set.columns:
        if Test_set[column].dtype == 'object':
            Test_set[column].fillna(Test_set[column].mode()[0], inplace=True)
        else:
            Test_set[column].fillna(Test_set[column].mean(), inplace=True)
            
    if 'goal' in Test_set.columns:
        X = Test_set.drop('goal', axis=1)
        Y = Test_set['goal']
        randomforest = RandomForestRegressor()
        randomforest.fit(X, Y)
        return randomforest
    else:
        return "Error: 'goal' column is not in the data set"

Test_set = pd.read_csv(r"C:\Users\Ankita Sriwastawa\Desktop\Java\Python\Test_set.csv")
model = handle_data(Test_set)
if isinstance(model, str):
    print(model)
else:
    accuracy_of_data = model.score(X, Y)
    print("The Accuracy of the data is:", accuracy_of_data)