# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:37:51 2023

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany

plot function using python for 
"""
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

def cross_corr_heatmap(data_dict, cmap='coolwarm', savefig=False,
                       figname='heatmap.svg', **pltfigkwargs):
    
    """
    Generate a heatmap to visualize the cross-correlation matrix of a given dataset.

    Parameters:
    - data_dict (dict): A dictionary containing data for the heatmap. Keys are 
      column names, and values are lists or arrays.
    - cmap (str, optional): The colormap for the heatmap. Default is 'coolwarm'.
    - savefig (bool, optional): If True, save the generated heatmap as a file.
      Default is False.
    - figname (str, optional): The filename for the saved figure if savefig is True. 
      Default is 'heatmap.svg'.
    - **pltfigkwargs: Additional keyword arguments to be passed to the matplotlib 
      figure creation.

    Returns:
    None

    Example:
    ```
    data = {
        'Column1': [1, 2, 3, 4],
        'Column2': [5, 6, 7, 8],
        'Column3': [9, 10, 11, 12]
    }

    cross_corr_heatmap(data, cmap='viridis', savefig=True, figname='correlation_heatmap.png', figsize=(8, 6))
    ```
    This example generates and displays a heatmap of the cross-correlation matrix for the provided data and saves it as 'correlation_heatmap.png' if savefig is True.

    Note:
    - Requires pandas and seaborn libraries to be installed.

    """
    
    # create a pandas dataframe from the dictionary
    df = pd.DataFrame(data_dict)

    # calculate the correlation matrix
    corr_matrix = df.corr()

    fig = plt.figure(**pltfigkwargs)
    # plot the heatmap
    sns.heatmap(corr_matrix, annot=True, cmap=cmap)
    plt.tight_layout()
    
    if savefig:
        plt.savefig(figname, bbox_inches="tight")
    
    plt.show()
    


