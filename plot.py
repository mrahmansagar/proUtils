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

def box_plot(data, groups=None, showfliers=False, showmeans=False, pval=False, 
             figsize=None, p_font=None, rc_font=12, figName=None):
    """
    Create a box plot for multiple datasets with optional group labels and pairwise comparison p-values.

    Parameters:
    - data (list of arrays): List of datasets to be plotted.
    - groups (list or None): List of group labels corresponding to each dataset. If None, default labels are used.
    - showfliers (bool): Whether to display outliers in the box plot. Default is False.
    - showmeans (bool): Whether to display mean values in the box plot. Default is False.
    - pval (bool): Whether to perform Tukey-Kramer post hoc test for pairwise comparisons and display p-values.
                   Default is False.
    - figsize (tuple or None): Size of the figure (width, height). If None, default size is used.
    - p_font (int or None): Font size for p-values. If None, default size is used.
    - rc_font (int): Font size for the entire plot. Default is 12.
    - figName (str or None): Filepath to save the figure. If None, the plot is displayed but not saved.

    Returns:
    - None: Displays the box plot with optional p-values and saves the figure if figName is provided.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    plt.rc('font', size=rc_font)

    box_plot = ax.boxplot(data, showmeans=showmeans, showfliers=showfliers, patch_artist=True)
    
    # adding color to each plot
    colormaps = plt.cm.Set3(np.linspace(0, 1, len(data)))
    
    for patch, color in zip(box_plot['boxes'], colormaps):
        patch.set_facecolor(color)
    
    # Extract components from the boxplot
    caps = [cap.get_ydata() for cap in box_plot['caps']]
    # boxes = [box.get_ydata() for box in box_plot['boxes']]
    # medians = [median.get_ydata() for median in box_plot['medians']]
    
    cap_max = np.max(caps)
    
    all_data = []
    for i, d in enumerate(data):
        all_data.extend(d)
    
    if pval and len(data)>1:
        all_labels = []
        for i, d in enumerate(data):
            if groups is not None:
                all_labels += [groups[i]] * len(d)
            else:
                all_labels += [f'grp{i}'] * len(d)
        
        # Perform Tukey-Kramer post hoc test for pairwise comparisons
        tukey_results = pairwise_tukeyhsd(all_data, all_labels, alpha=0.05)
        tukey_data = tukey_results.summary().data
        
        y = cap_max
        count = 1
        for a in range(len(data)):
            for b in range(a + 1, len(data)):
                
                x1 = a+1
                x2 = b+1
                y, h, col = (y + cap_max*0.10), cap_max*0.02, 'k'
                p = tukey_data[count][3]
                count = count + 1
                plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.0, c=col)
                plt.text((x1+x2)*.5, y+h, f'p={p}', ha='center', va='bottom', color=col, fontsize=p_font)
                
                
    if np.max(all_data) >= 1000:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(4,4))
    
    
    # Add labels and title
    if groups is not None:
        ax.set_xticklabels(groups)
    else:
        ax.set_xticklabels([f'data {i}' for i in range(1, len(data) + 1)])
    
    if figName is not None:
        plt.savefig(figName)
        
    plt.show()
    
    
    
    
    