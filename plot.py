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
    

from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np

def box_plot(data, groups=None, showfliers=False, showmeans=False, pval=False):
    fig, ax = plt.subplots(figsize=(10,10))
    box_plot = ax.boxplot(data, showmeans=showmeans, showfliers=showfliers, patch_artist=True)
    
    # Add color to the boxes
    colors = ['lightblue', 'lightgreen', 'wheat', 'plum']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    # Extract components from the boxplot
    caps = [cap.get_ydata() for cap in box_plot['caps']]
    # boxes = [box.get_ydata() for box in box_plot['boxes']]
    # medians = [median.get_ydata() for median in box_plot['medians']]
    
    cap_max = np.max(caps)
    
    
    if pval and all(isinstance(sublist, list) for sublist in data):
        
        all_data = []
        all_labels = []
        for i, d in enumerate(data):
            all_data += d
            if groups is not None:
                all_labels += [groups[i]] * len(d)
            else:
                all_labels += [f'grp{i}']* len(d)
        
        # Perform Tukey-Kramer post hoc test for pairwise comparisons
        tukey_results = pairwise_tukeyhsd(all_data, all_labels, alpha=0.05)
        tukey_data = tukey_results.summary().data
        p_font = 20
        y = cap_max
        count = 1
        for a in range(len(data)):
            for b in range(a + 1, len(data)):
                
                x1 = a+1
                x2 = b+1
                y, h, col = (y + cap_max*0.15), 7e3, 'k'
                p = tukey_data[count][3]
                count = count + 1
                plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, c=col)
                plt.text((x1+x2)*.5, y+h, f'p={p}', ha='center', va='bottom', color=col, fontsize=p_font)
                
                
    ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    plt.rc('font', size=20)
    plt.show()
    
    
    
    
    