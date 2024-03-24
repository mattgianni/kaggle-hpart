import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sklearn.preprocessing import OneHotEncoder, StandardScaler


class FeatureSet:
    def __init__(self, path: os.path, y_column: str = None) -> None:
        logger.debug(f"DataLoader initialized")
        if path is not None:
            self.load_data(path, y_column)

    def load_data(self, path: os.path, y_column: str = None) -> pd.DataFrame:
        logger.debug(f"DataLoader loading {path}")

        self.path = path
        self.y_column = y_column

        rawdf = pd.read_csv(self.path)
        if y_column is not None:
            self.y = rawdf[y_column]
            rawdf.drop(y_column, axis=1, inplace=True)
        else:
            self.y = None

        self.rawdf = rawdf
        self.preprocess()

    def preprocess(self):
        tempVector = self.rawdf.dtypes == "object"
        cat_cols = list(tempVector[tempVector].index)

        tempVector = self.rawdf.dtypes != "object"
        ncat_cols = list(tempVector[tempVector].index)

        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        sse = StandardScaler()

        cat_cols_df = pd.DataFrame(ohe.fit_transform(self.rawdf[cat_cols]))
        cat_cols_df.columns = [
            f"cf-{index}" for index in range(len(cat_cols_df.columns))
        ]

        ncat_cols_df = pd.DataFrame(sse.fit_transform(self.rawdf[ncat_cols]))
        ncat_cols_df.columns = [
            f"ncf-{index}" for index in range(len(ncat_cols_df.columns))
        ]

        full_df = pd.concat([cat_cols_df, ncat_cols_df], axis=1)

        # Handle missing values by adding indicator columns for nulls and filling with 0
        null_df = full_df.isna()
        l = null_df.any()
        null_df = null_df[l[l == True].index]

        # label the null columns
        null_df.columns = [f"nicf-{index}" for index in range(len(null_df.columns))]

        # fill the nulls with 0
        full_df.fillna(0, inplace=True)

        # add the null columns to the full dataframe
        full_df = pd.concat([full_df, null_df], axis=1).astype(float)

        self.normalized_df = full_df


if __name__ == "__main__":
    fs = FeatureSet(os.path.join("..", "data", "train.csv"))
