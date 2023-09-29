import logging

import mlflow


class ExperimentTracking:
    def __init__(self) -> None:
        pass

    def create_experiment(self, experiment_name):
        try:
            mlflow.create_experiment(name=experiment_name)
        except Exception:
            logging.info("Experiment already exists")
