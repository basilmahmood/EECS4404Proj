import pandas as pd
import numpy as np
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.svm import SVC

# References:
# https://www.kaggle.com/overload10/income-prediction-on-uci-adult-dataset

data = pd.read_csv("adult.data", names=[
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "over-50K"
], skipinitialspace=True)

data.replace(to_replace={
    "over-50K": {'>50K': 0, '<=50K': 1},
}, inplace=True)

data = pd.get_dummies(data)

corr_df = data.corr()
relevant_features = corr_df.loc[abs(corr_df["over-50K"]) > 0.25]["over-50K"].index.to_numpy()
relevant_features = relevant_features[relevant_features != "over-50K"]

# Split into training and test data (80 and 20% each)
training_data_unshuffled, test_data = np.split(data, [int(0.8*len(data))])

# Shuffle training data by randomly sampling entire set
training_data = training_data_unshuffled.sample(frac=1)

X = training_data[relevant_features]
X_test = test_data[relevant_features]
Y = training_data["over-50K"]
Y_test = test_data["over-50K"]

classifier = SVC(kernel='poly', cache_size=2000)

classifier.fit(X, Y)

Y_predict = classifier.predict(X_test)
print(f1_score(Y_test, Y_predict))


