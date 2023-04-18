import argparse
import logging
import warnings

import dvc.api
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

tracking_uri = f"http://localhost:4000"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Hotspot Prediction")

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
path = "data/hotspot_demo.csv"


# repo = "/home/mubarak/mlops-demo"
# version = "nv1"


def eval_metrics(actual, pred):
    """Evaluate model performance.


    Args:
        actual (_int_): The ground truth values
        pred (_int_): The predicted values

    Returns:
        _type_: The RMSE, MAE and R2 scores
    """
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    
    
    return rmse, mae, r2

def train_model(params):
    """Train the model using the training data and save the model to the output directory.

    Args:
        params (_type_): _description_
    """
    warnings.filterwarnings
    np.random.seed(40)


    # Load the csv file from dvc repo
    remote = str(params.remote)
    version = str(params.dversion)
    run_name = str(params.run_name)

    # Get URL from DVC
    data_url = dvc.api.get_url(path=path, remote=remote, rev=version)
    try:
        data = pd.read_csv(data_url, sep=",")
    except Exception:
        logger.exception("Unable to load the training data, check if the url is correct")
    
    # Split the data into training and test sets

    train, test = train_test_split(data, random_state=40)

    # The predicted column is burn_area
    x_train = train.drop(["burn_area", "Unnamed: 0"], axis=1)
    x_test = test.drop(["burn_area", "Unnamed: 0"], axis=1)
    y_train = train[["burn_area"]]
    y_test = test[["burn_area"]]

    # # Scale the data
    # scaler = StandardScaler()
    # x_scaled = scaler.fit_transform(x_train)
    # y_scaled = scaler.fit_transform(y_train)

    alpha = float(params.alpha) if float(params.alpha) > 1 else 0.5
    l1_ratio = float(params.l1_ratio) if float(params.l1_ratio) > 2 else 0.5

    with mlflow.start_run(run_name=run_name):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        # lr.fit(x_train, y_train)

        pipe = make_pipeline(StandardScaler(), lr)
        pipe.fit(x_train, y_train)
        

        predicted_qualities = pipe.predict(x_test)

        y_test = StandardScaler().fit_transform(y_test)
        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        
        # Log artifacts: columns used for modelling
        cols_x = pd.DataFrame(list(x_train.columns))
        cols_x.to_csv('utils/features.csv', header=False, index=False)
        mlflow.log_artifact("features.csv")

        cols_y = pd.DataFrame(list(y_train.columns))
        cols_y.to_csv('utils/targets.csv', header=False, index=False)
        mlflow.log_artifact("targets.csv")

        # Log data params
        #mlflow.log_param('data url', data_url)
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
            registered_model_name="sk-learn-elasticnet-hotspot/v1")



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Enter the necessary arguments')
    parser.add_argument('--alpha', help='Alpha value of the model')
    parser.add_argument('--remote', help='The name of the dvc remote storage')
    parser.add_argument('--l1_ratio', help='l1 ratio value of the model')
    parser.add_argument('--dversion', help='The version of the data to be used')
    parser.add_argument('--run_name', help='The run name of the experiment')

    args = parser.parse_args()
    
    train_model(args)

# The above code is the train.py file. It is a simple script that trains a model using the data from the DVC repository. The script is run using the following command:
   # python src/train.py --alpha 0.5 --l1_ratio 0.5 --remote storage --dversion nv1 --run_name "First Run"
   # First Run is the name of the run in the MLflow UI
   # The alpha and l1_ratio are the hyperparameters of the model
   # The remote is the name of the DVC remote storage
    