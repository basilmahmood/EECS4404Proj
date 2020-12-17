import pandas as pd
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
    "over-50K": {'>50K': 1, '<=50K': 0},
}, inplace=True)

data = pd.get_dummies(data)
corr_df = data.corr()

corr_df["over-50K"].to_csv('./corr.csv')
corr_df.loc[abs(corr_df["over-50K"]) > 0.25]["over-50K"].to_csv('./corr_relevant.csv')

