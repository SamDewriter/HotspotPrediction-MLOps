import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow

import logging

tracking_uri = f"http://localhost:4000"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("MLOPs Demo")

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


# Get URL from DVC
import dvc.api

path = "data/hotspot_demo.csv"
repo = "/home/mubarak/mlops-demo"
version = "v3"


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings
    np.random.seed(40)

    # Load the csv file
    data_url = dvc.api.get_url(path=path, repo=repo,rev=version)
    try:
        data = pd.read_csv(data_url, sep=",")
    except Exception as e:
        logger.exception(
            "Unable to load the training data, check if it's the url is correct"
        )

        # Split the data into training and test sets

    train, test = train_test_split(data)

    # The predicted column is burn_area
    x_train = train.drop(["burn_area"], axis=1)
    x_test = test.drop(["burn_area"], axis=1)
    y_train = train[["burn_area"]]
    y_test = test[["burn_area"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        lr.fit(x_train, y_train)

        predicted_qualities = lr.predict(x_test)

        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        
        # Log artifacts: columns used for modelling
        cols_x = pd.DataFrame(list(x_train.columns))
        cols_x.to_csv('features.csv', header=False, index=False)
        mlflow.log_artifact("features.csv")

        cols_y = pd.DataFrame(list(y_train.columns))
        cols_y.to_csv('targets.csv', header=False, index=False)
        mlflow.log_artifact("targets.csv")

        # Log data params
        mlflow.log_param('data url', data_url)
        mlflow.log_param('data_version', version)
        mlflow.log_param('input_rows', data.shape[0])
        mlflow.log_param('input_cols', data.shape[1])
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Log the sklearn model and register as version 1
        mlflow.sklearn.log_model(
            sk_model= lr,
            artifact_path="sklearn-model",
            registered_model_name="sk-learn-elasticnet-hotspot/MLOPs")