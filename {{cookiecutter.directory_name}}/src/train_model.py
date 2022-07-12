import hydra
from omegaconf import DictConfig
import logging
import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


@hydra.main(config_path="../config", config_name="main")
def train_model(config: DictConfig):
    """Function to train the model"""
    
    logging.info("Loading Data")

    iris_dataset = load_iris()

    mlflow.set_registry_uri(config.mlflow.registry_uri)
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.start_run(run_name="iris_data", experiment_id=config.mlflow.experiment_id)

    X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], test_size=0.25, random_state=0)

    logging.info("Training Model")

    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    valid_accuracy = model.score(X_test, y_test)

    logging.info("Saving Model")

    mlflow.log_metric("validation_accuracy", valid_accuracy)
    mlflow.sklearn.log_model(model, "model")

    mlflow.end_run()

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    train_model()
