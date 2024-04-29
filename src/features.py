import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors
import numpy as np
import math

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from sklearn.preprocessing import OneHotEncoder, StandardScaler


def plot_feature_vs_target(
    plt: plt,
    X: pd.DataFrame,
    y: pd.Series,
    color_feature: pd.DataFrame | pd.Series | str = "blue",
    cmap: mcolors.Colormap | str = "viridis",
    title: str = "Feature vs SalePrice",
    alpha: float = 0.5,
    size: pd.DataFrame | int = 30,
) -> tuple[Figure, any]:
    """
    Plot the feature columns against the target variable
    """
    # Get the list of feature column names
    feature_cols = X.columns
    n = len(feature_cols)

    # Set the number of rows and columns for the subplots
    num_cols = round(math.sqrt(n))
    num_rows = math.ceil(n / num_cols)

    # Create the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 18))

    # Flatten the axes array
    axes = axes.flatten()

    # Iterate over the feature columns and plot each feature against the target variable
    for i, feature in enumerate(feature_cols):
        # logger.debug(f"Plotting {feature} vs {y.name}")
        ax = axes[i]
        if isinstance(color_feature, str):
            ax.scatter(X[feature], y, alpha=alpha, c=color_feature, s=size)
        else:
            ax.scatter(X[feature], y, alpha=alpha, c=color_feature, cmap=cmap, s=size)
        ax.set_xlabel(feature)

        # Remove the y-axis labels to prevent clutter
        ax.set_yticklabels([])
        # ax.set_ylabel('SalePrice')

    # Remove any extra subplots
    if len(feature_cols) < num_rows * num_cols:
        for j in range(len(feature_cols), num_rows * num_cols):
            fig.delaxes(axes[j])

    fig.suptitle(title, fontsize=20, y=0.99)

    # Adjust the spacing between subplots
    fig.tight_layout()

    return (fig, axes)


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
