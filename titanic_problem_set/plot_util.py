import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List

def plot_count(df: pd.DataFrame, 
               features: Union[str, List[str]], 
               hue: str, 
               title: str, 
               colour_list=["#A5D7E8", "#576CBC"]) -> None:
    if type(features) is str:
        col = features
        sns.countplot(
            data = df.dropna(subset=[col]),  #handle missing values
            x=col,  
            hue=hue,
            palette=colour_list)
        plt.title(f"{title} / {col}")
    else:
        # set number of columns
        ncols = 2
        # calculate number of rows
        nrows = len(features) // ncols + (len(features) % ncols > 0)  
        # set up subplot
        fig, axes = plt.subplots(nrows, ncols, figsize = (10,10))
        # loop through each target column to plot
        for i, ax in enumerate(axes.flatten()):
            if i >= len(features):
                continue
            else:
                col = features[i]
                sns.countplot(
                    data = df.dropna(subset=[col]),  #handle missing values
                    x=col,  
                    hue=hue,
                    palette=colour_list, 
                    ax=ax)
                # Set title and show plot
                ax.set_title(f"{title} / {col}")

        fig.tight_layout()
    plt.show()


def plot_distribution(df: pd.DataFrame, 
                      features: Union[str, List[str]], 
                      multiple: str,
                      hue: str,
                      title: str,
                      colour_list: List) -> None:
    if type(features) is str:
        col = features
        sns.histplot(
            data = df.dropna(subset=[col]),  #handle missing values
            x=col,
            multiple=multiple,
            hue=hue,
            palette=colour_list)
        plt.title(f"{title} / {col}")
    else:
        # set number of columns (use 3 to demonstrate the change)
        ncols = 2
        # calculate number of rows
        nrows = len(features) // ncols + (len(features) % ncols > 0)  
        # set up subplot
        fig, axes = plt.subplots(nrows, ncols, figsize = (10,5))
        # loop through each target column to plot
        for i, ax in enumerate(axes.flatten()):
            if i >= len(features):
                continue
            else:
                col = features[i]
                sns.histplot(
                    data = df.dropna(subset=[col]),  #handle missing values
                    x=col,  
                    multiple=multiple,
                    hue=hue,
                    palette=colour_list,
                    ax=ax)
            # Set title and show plot
            ax.set_title(f"{title} / {col}")
        fig.tight_layout()
    plt.show()