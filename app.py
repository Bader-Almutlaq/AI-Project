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

    if task == "classification":
        data = pd.read_csv("data/heart.csv")
        results, metrics = run_classification(data)
    elif task == "regression":
        data = pd.read_csv("data/diabetes.csv")
        results, metrics = run_regression(data)
    elif task == "clustering":
        data = pd.read_csv("data/breast_cancer.csv")
        results, metrics = run_clustering(data)

    #print(results)
    #print(metrics)
    return render_template("results.html", results=results, metrics=metrics)


if __name__ == "__main__":
    app.run(debug=True)
