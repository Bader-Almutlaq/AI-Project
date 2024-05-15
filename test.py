import pandas as pd
from models.classification import run_classification
from models.regression import run_regression
from models.clustering import run_clustering

data = pd.read_csv("data/breast_cancer.csv")

results = run_clustering(data)

print(results)