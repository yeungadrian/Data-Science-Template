import mlflow
import pandas as pd
from experiments import ExperimentTracking
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

target = "MedHouseVal"
cont_names = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "HouseAge2",
]


def main() -> None:
    ExperimentTracking().create_experiment("California-Housing")
    mlflow.start_run(experiment_id="516215408145937271")
    df = pd.read_parquet("data/processed/housing.pq")
    train_x, test_x, train_y, test_y = train_test_split(
        df[cont_names].values, df[target].values
    )
    model = RandomForestRegressor()
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    mse = mean_squared_error(test_y, pred_y)
    mlflow.log_metric("mse", mse)
    mlflow.end_run()


if __name__ == "__main__":
    main()
