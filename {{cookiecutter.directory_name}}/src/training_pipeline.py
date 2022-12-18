import itertools
import logging
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import sklearn
from sklearn import metrics
from dotenv import load_dotenv
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %I:%M:%S%p",
)

config = {
    # Paths, hyperparameters, environments,
    "experiment_name": "Default",
    "tags": {
        "calibration_date": "2021-12-31",
    },
    "model_path": "sklearn_classifier",
    "test_size": 0.25,
    "random_state": 0,
    "bacc_threshold": 0.5,
    "model_registry_name": "iris_classifer",
}

# Improvements: Functions should be moved to within a class within modules


def create_experiment(
    mlflow_client: mlflow.MlflowClient, experiment_name: str
) -> str:
    """
    Creates experiment in MLFlow if it doesn't exist

    Args:
        mlflow_client mlflow.MlflowClient(): mlflow client
        experiment_name (str): Experiennt Name

    Returns:
        str: Experiment ID

    """
    try:
        experiment_id = mlflow_client.create_experiment(experiment_name)
        logging.info(
            f"Experiment {experiment_name} created with id {experiment_id}"
        )
    except MlflowException as error:
        experiment = mlflow_client.get_experiment_by_name(experiment_name)
        if experiment.lifecycle_stage == LifecycleStage.DELETED:
            logging.error(f"Experiment {experiment_name} already DELETED")
            raise
        experiment_id = experiment.experiment_id
        logging.info(
            f"Experiment {experiment_name} exists with id {experiment_id}"
        )
    return experiment_id


def start_mlflow_run(
    experiment_name: str, run_name: str = None
) -> mlflow.ActiveRun:
    """
    Start an mlflow run under specified experiment.
    Will generate experiment if it doesn't exist.

    Args:
        experiment_name (str): Experiment name in MLFlow
        run_name (str=None): Optional run name

    Returns:
        mlflow.ActiveRun

    """
    mlflow_client = MlflowClient()
    experiment_id = create_experiment(mlflow_client, experiment_name)
    run = mlflow.start_run(
        experiment_id=experiment_id, run_name=run_name, tags=config["tags"]
    )
    return run


def load_data() -> sklearn.utils._bunch.Bunch:
    """
    Load iris dataset

    Returns:
        sklearn.utils._bunch.Bunch

    """
    iris_dataset = load_iris()
    return iris_dataset


def process_data(
    iris_dataset: sklearn.utils._bunch.Bunch,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, ...]:
    """
    Generate test train split

    Args:
        iris_dataset (sklearn.utils._bunch.Bunch): iris dataset

    Returns:
        Tuple[np.ndarray, ...]: (X_train, X_test, y_train, y_test)

    """
    X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset["data"],
        iris_dataset["target"],
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """
    Train sklearn decision tree classifier

    Args:
        X_train (np.ndarray): X_train
        y_train (np.ndarray): y_train

    Returns:
        Any: Fitted model

    """
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model


def log_model(
    model: Any,
    model_path: str,
) -> mlflow.models.model.ModelInfo:
    """
    Log model params, model and model signature (inputs / outputs) to mlflow

    Args:
        model (Any): Sklearn Decision tree model
        X_train (np.ndarray): X_train
        y_train (np.ndarray): y_train
        model_path (str): Artifact path in mlflow

    Returns:
        None

    """
    input_schema = Schema(
        [
            ColSpec("double", "sepal length (cm)"),
            ColSpec("double", "sepal width (cm)"),
            ColSpec("double", "petal length (cm)"),
            ColSpec("double", "petal width (cm)"),
        ]
    )
    output_schema = Schema([ColSpec("long")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    mlflow.log_params(model.get_params())
    result = mlflow.sklearn.log_model(model, model_path, signature=signature)
    return result.model_uri, result.run_id


def get_preds(model: Any, X_test: np.ndarray) -> np.ndarray:
    """
    Get predictions from sklearn model

    Args:
        model (Any): sklearn model
        X_test (np.ndarray): data

    Returns:
        np.ndarray: Probability predictions

    """
    probs = model.predict_proba(X_test)
    y_pred = probs[:, 1]
    return y_pred


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List,
    normalize: bool = False,
    title: str = "Confusion matrix",
) -> plt.figure:
    """
    Generate multiclass confusion matrix plot

    Args:
        cm (np.ndarray): Confusion matrix results
        classes (List): List of all classes
        normalize (bool=False): Show normalised results
        title (str="Confusionmatrix"): Plot title

    Returns:
        plt.figure: confusion matrix plot

    """
    cmap = plt.cm.Blues
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return fig


def log_model_validation(
    model: Any, X_test: np.ndarray, y_test: np.ndarray
) -> None:
    """
    Log model validation metrics to mlflow
    - Training curves
    - AUC
    - Prediction distributions
    - Confusion matrices

    Args:
        model (Any): model
        X_test (np.ndarray): X_test
        y_test (np.ndarray): y_test

    Returns:
        None

    """
    y_pred = get_preds(model, X_test)
    bacc_score = metrics.balanced_accuracy_score(y_test, y_pred)
    mlflow.log_metric("balanced_accuracy_score", bacc_score)
    cm = metrics.confusion_matrix(y_test, y_pred)
    cm_png = plot_confusion_matrix(cm, classes=list(np.unique(y_test)))
    mlflow.log_figure(cm_png, "confusionmatrix.png")
    return bacc_score


def register_model_name(
    mlflow_client: mlflow.MlflowClient,
    model_name: str,
    description: str = None,
) -> None:
    """
    Create new registered model if it doesn't exist

    Args:
        mlflow_client (mlflow.MlflowClient): mlflow client
        model_name (str): Name of registered model
        description (str=None): Model description

    Returns:
        None

    """
    try:
        mlflow_client.create_registered_model(model_name, description)
        logging.info(f"Registered model {model_name} created")
    except MlflowException as create_error:
        logging.info(f"Registered model {model_name} already exists")

    return None


def register_model(ba_score: float, model_uri: str, run_id: str) -> Any:
    """
    Register model from experiment run as new version

    Args:
        ba_score (float): balanced accuracy score
        model_uri (str): model uri of experiment run
        run_id (str): experiment run id that generated model

    Returns:
        Any: registered model version as str or nothing

    """
    if ba_score > config["bacc_threshold"]:
        logging.info(
            f"Model being registered. Balanced accuracy score: {ba_score}"
        )
        mlflow_client = MlflowClient()
        register_model_name(mlflow_client, config["model_registry_name"])
        result = mlflow_client.create_model_version(
            config["model_registry_name"],
            model_uri,
            run_id=run_id,
            tags=config["tags"],
        )
        return result.version
    else:
        logging.info(
            f"Model was not registered. Balanced accuracy score: {ba_score}"
        )
        return None


def promote_model(model_version: str) -> None:
    """
    Promote model to "production" stage on MLFlow

    Args:
        model_version (str): model version to promote

    Returns:
        None

    """
    mlflow_client = MlflowClient()
    mlflow_client.transition_model_version_stage(
        name=config["model_registry_name"],
        version=model_version,
        stage="Production",
        archive_existing_versions=True,
    )
    return None


def main(config) -> None:
    load_dotenv()
    start_mlflow_run(config["experiment_name"])
    iris_dataset = load_data()
    X_train, X_test, y_train, y_test = process_data(
        iris_dataset, config["test_size"], config["random_state"]
    )
    model = train_model(X_train, y_train)
    model_uri, run_id = log_model(model, config["model_path"])
    bacc_score = log_model_validation(model, X_test, y_test)
    mlflow.end_run()
    model_version = register_model(bacc_score, model_uri, run_id)
    if model_version is not None:
        promote_model(model_version)


if __name__ == "__main__":
    main(config)
