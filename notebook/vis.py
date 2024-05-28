# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 16:40:19 2023

@author: Mmr Sagr
PhD Researcher | MPI-NAT Goettingen, Germany

visualization functionallities with jupyter notebooks
"""

import ipywidgets as widgets
import numpy as np
import matplotlib.pyplot as plt 

def interactive_visualize(arr, all_side=False, cmap='gray',figsize=None, colorbar=False, **kwargs):
    """
    This function creates an interactive visualization of the input array in 2D or 3D.

    Parameters:
    arr (ndarray): The input array to be visualized.
    cmap (str): The colormap to use for the plot. Default is 'viridis'
    **kwargs: Additional keyword arguments to be passed to plt.imshow.

    Returns:
    None
    """
    
    if arr.ndim == 2:
        plt.imshow(arr, cmap=cmap, **kwargs)
        plt.show()
    elif arr.ndim == 3:
    
        slider = widgets.IntSlider(
            min=0, max=arr.shape[0]-1, step=1, value=np.rint(arr.shape[0] / 2))

        def visualize_3d(slice_number):
            """
            This function plots the 3D visualization of the input array.

            Parameters:
            slice_number (int): The slice number of the input array to be visualized.

            Returns:
            None
            """
            if all_side:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
                ax1.imshow(arr[slice_number, :, :], cmap=cmap, **kwargs)
                ax2.imshow(arr[:, slice_number, :], cmap=cmap, **kwargs)
                ax2.set_xticks([])
                ax2.set_yticks([])
                ax3.imshow(arr[:, :, slice_number], cmap=cmap, **kwargs)
                ax3.set_xticks([])
                ax3.set_yticks([])
                plt.show()
                
            else:
                fig, ax = plt.subplots(1, 1,  figsize=figsize)
                plt.subplots_adjust(hspace=0.1, wspace=0.1)
                pos = ax.imshow(arr[slice_number, :, :], cmap=cmap, **kwargs)
                if colorbar:
                    fig.colorbar(pos, ax=ax)
                plt.show()


        widgets.interact(visualize_3d, slice_number=slider)
    
    else:
        raise ValueError("Input array must be either 2D or 3D.")


