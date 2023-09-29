import numpy as np
import pandas as pd


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    df["HouseAge2"] = np.square(df["HouseAge"].values)
    return df


def main() -> None:
    df = pd.read_parquet("data/interim/housing.pq")
    df = add_age_features(df)
    df.to_parquet("data/processed/housing.pq")


if __name__ == "__main__":
    main()
