# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage import filters, img_as_ubyte
from skimage.transform import hough_line, hough_line_peaks

def combine_ndarrays(*arrays, space_btwn=10):
    """
    Combine multiple NumPy ndarrays with the same dimensions along the last dimension.

    Args:
        *arrays: Variable number of input ndarrays to be combined.
        space_btwn (int, optional): The amount of space to leave between input arrays
            along the last dimension. Default is 10.

    Returns:
        numpy.ndarray: A new ndarray containing the combined data from input arrays.

    Raises:
        ValueError: If less than two input arrays are provided or if input arrays have
            different dimensions.

    Example:
        data1 = np.ones((2, 3, 4), dtype=int)
        data2 = np.zeros((2, 3, 4), dtype=int)
        data3 = np.full((2, 3, 4), 2, dtype=int)

        combined_result = combine_ndarrays(data1, data2, data3)

        print(combined_result.shape)  # Output: (2, 3, 38)
    """
    if len(arrays) < 2:
        raise ValueError("At least two input arrays are required.")
    
    # Check if all input arrays have the same shape
    expected_shape = arrays[0].shape
    for array in arrays:
        if array.shape != expected_shape:
            raise ValueError("Input arrays must have the same dimensions.")

    # Calculate the shape of the combined array
    combined_shape = list(expected_shape)
    combined_shape[-1] = sum(array.shape[-1] for array in arrays) + (len(arrays) - 1) * space_btwn

    # Initialize the combined array with zeros
    combined_array = np.zeros(combined_shape, dtype=arrays[0].dtype)

    # Copy data from input arrays into the combined array
    offset = 0
    for array in arrays:
        combined_array[..., offset:offset + array.shape[-1]] = array
        offset += array.shape[-1] + space_btwn

    return combined_array


def tilt_correction(imarray, edge_th=1, edge_filter=filters.prewitt, ang_vari=2, plot=False):
    """
   Corrects the tilt in an input image by detecting and removing tilt angle.

   Parameters:
   - imarray (numpy.ndarray): The input image as a NumPy array.
   - edge_th (int, optional): Threshold for edge detection. Default is 1.
   - edge_filter (function, optional): Edge detection filter function. Default is filters.prewitt.
   - ang_vari (int/float, optional): Allowed variation of angles. Default is 2 degree.
   - plot (bool, optional): Whether to plot intermediate results. Default is False.

   Returns:
   - rotated_imarray (numpy.ndarray or None): Corrected image if successful, None otherwise.
   - rotation (float): Detected rotation angle in degrees.
   - d (float): Distance parameter from Hough transform.
   """
    # Threshold the input image to create a binary image
    th_im = imarray > edge_th

    # Fill small holes in the binary image
    th_im = nd.binary_fill_holes(th_im, np.ones((20, 20)))

    # Detect edges in the binary image
    edges = img_as_ubyte(edge_filter(th_im))

    # Define a range of angles to test for Hough transform
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)

    # Perform Hough transform to detect lines
    hspace, theta, distance = hough_line(edges, tested_angles)

    # Find the peaks in the Hough transform space
    h, q, d = hough_line_peaks(hspace, theta, distance)
    
    angle_list=[]
    for _, angle, dist in zip(*hough_line_peaks(hspace, theta, distance)):
        angle_list.append(angle*180/np.pi)

    
    angle_variation = np.zeros(shape=(len(angle_list),len(angle_list)))
    for i in range(len(angle_list)):
        for j in range(len(angle_list)):
            angle_variation[i, j] = abs(angle_list[j] - angle_list[i])
    
    if angle_variation.max() > ang_vari:
        print("Could not find exact straight line for tilt correction")
        fig, axes = plt.subplots(1, 3, figsize=(15, 6))
        ax = axes.ravel()

        ax[0].imshow(edges, cmap='gray')
        ax[0].set_title('Input image')
        ax[0].set_axis_off()

        ax[1].imshow(np.log(1 + hspace),
                     extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), distance[-1], distance[0]],
                     cmap='gray', aspect=1/1.5)
        ax[1].set_title('Hough transform')
        ax[1].set_xlabel('Angles (degrees)')
        ax[1].set_ylabel('Distance (pixels)')
        ax[1].axis('image')

        ax[2].imshow(edges, cmap='gray')

        origin = np.array((0, edges.shape[1]))

        for _, angle, dist in zip(*hough_line_peaks(hspace, theta, distance)):
            y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
            ax[2].plot(origin, (y0, y1), '-r')
        ax[2].set_xlim(origin)
        ax[2].set_ylim((edges.shape[0], 0))
        ax[2].set_axis_off()
        ax[2].set_title('Detected lines')
        
        plt.tight_layout()
        plt.show()
        
        return None, angle_variation
    
    else:
        rotation = np.array([a for a in angle_list]).mean()
        if rotation < 0 :
            rotated_imarray = nd.rotate(imarray, angle=90+rotation, reshape=False)
        else:
            rotated_imarray = nd.rotate(imarray, angle=180+90+rotation, reshape=False)
        
        if plot:
            plt.figure(figsize=(12,6))
            plt.subplot(1,2,1)
            plt.imshow(imarray, cmap='gray')
            plt.subplot(1,2,2)
            plt.imshow(rotated_imarray, cmap='gray')
            plt.show()
        
        return rotated_imarray, rotation, d