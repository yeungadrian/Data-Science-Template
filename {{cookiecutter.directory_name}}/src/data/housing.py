import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing


def type_cols(df: pd.DataFrame) -> pd.DataFrame:
    int_cols = ["HouseAge", "Population"]
    float_cols = [
        "MedInc",
        "AveRooms",
        "AveBedrms",
        "AveOccup",
        "Latitude",
        "Longitude",
        "MedHouseVal",
    ]
    df[int_cols] = df[int_cols].astype(np.int64)
    df[float_cols] = df[float_cols].astype(np.float64)
    return df


def main() -> None:
    ca_housing = fetch_california_housing(as_frame=True)
    df = pd.concat([ca_housing["data"], ca_housing["target"]], axis=1)
    df = type_cols(df)
    df.to_parquet("data/interim/housing.pq")


if __name__ == "__main__":
    main()
