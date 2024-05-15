from flask import Flask, request, render_template
import pandas as pd
from models.classification import run_classification
from models.regression import run_regression
from models.clustering import run_clustering

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/results", methods=["POST"])
def results():
    task = request.form["task"]
    dataset_name = request.form["dataset"]

    if dataset_name == "heart":
        data = pd.read_csv("data/heart.csv")
    elif dataset_name == "diabetes":
        data = pd.read_csv("data/diabetes.csv")
    elif dataset_name == "breast_cancer":
        data = pd.read_csv("data/breast_cancer.csv")

    if task == "classification":
        results = run_classification(data)
    elif task == "regression":
        results = run_regression(data)
    elif task == "clustering":
        results = run_clustering(data)

    return render_template("results.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)
