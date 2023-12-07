# -*- coding: utf-8 -*-
"""
@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany
"""
import os
from tkinter import Tcl
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as nd
from skimage import filters, img_as_ubyte
from skimage.transform import hough_line, hough_line_peaks

from PIL import Image

import tifffile


# scaling the data/global range adjustment to match with old data range 
def map_values_to_range(input_array, from_low, from_high, to_low, to_high):
    """
    Map values from one range to another using linear scaling.

    Parameters:
    - input_array (numpy.ndarray): The array containing values to be mapped.
    - from_low (float): The lower bound of the original range.
    - from_high (float): The upper bound of the original range.
    - to_low (float): The lower bound of the target range.
    - to_high (float): The upper bound of the target range.

    Returns:
    numpy.ndarray: An array with values mapped from the original range to the target range.
    """
    from_diff = from_high - from_low
    to_diff = to_high - to_low
    mapped_array = (input_array - from_low) / from_diff * to_diff + to_low
    
    return mapped_array


def load_roi(roi_path, check_blank=False):
    """
    Load a stack of 2D image slices from the specified directory and create a 3D volume.

    Args:
        roi_path (str): The path to the directory containing the image slices.
        check_for_blank (bool, optional): Whether to check for and handle blank slices (default is False).

    Returns:
        numpy.ndarray: A 3D NumPy array representing the volume formed by stacking the image slices.
                      The shape of the array is (num_slices, height, width), and the data type
                      matches the data type of the image slices.

    This function reads image slices from the specified directory, sorts them in a natural
    order, and assembles them into a 3D volume. If check_blank is set to true then 
    Blank slices (those containing all zeros) are identified and added at the 
    end of the volume. The resulting volume can be used for various medical imaging 
    and scientific applications.

    Note:
    - Image file formats supported for reading must be compatible with the 'PIL' library.
    - The slices are assumed to be 2D images with uniform dimensions (height and width).
    - The function assumes that the image slices have consistent data types.

    Example usage:
    ```python
    roi_volume = load_roi('/path/to/slice_directory')
    ```

    In the example above, `roi_volume` will contain a 3D NumPy array representing the volume
    formed by stacking the image slices from the specified directory.
    """
    
    files = os.listdir(roi_path)
    slices = Tcl().call('lsort', '-dict', files)
    sample = Image.open(os.path.join(roi_path, slices[0]))
    sample = np.array(sample)
    vol = np.empty(shape=(len(slices), *sample.shape), dtype=sample.dtype)
    
    #temporary list to hold blank slices 
    blank_slices = []
    
    for i, fname in enumerate(slices):
        im = Image.open(os.path.join(roi_path, fname))
        imarray = np.array(im)
        
        if check_blank and np.all(imarray == 0):
            blank_slices.append(imarray)
        else:
            vol[i - len(blank_slices), :, :] = imarray
    
    # Append blank slices at the end
    if len(blank_slices) > 0:
        vol[-len(blank_slices):] = blank_slices
    
    return vol
                        


def norm8bit(v, minVal=None, maxVal=None):
    """
    NORM8BIT function takes an array and normalized it before converting it into 
    a 8 bit unsigned integer and returns it.

    Parameters
    ----------
    v : numpy.ndarray
        Array of N dimension.
    minVal : number 
        Any value that needs to be used as min value for normalization. If no
        value is provided then it uses min value of the given array. The default is None.
    maxVal : number 
        Any value that needs to be used as max value for normalization. If no
        value is provided then it uses max value of the given array. The default is None.

    Returns
    -------
    numpy.ndarray (uint8)
        Numpy Array of same dimension as input with data type as unsigned integer 8 bit

    """
    if minVal == None:
        minVal = v.min()
    
    if maxVal == None:
        maxVal = v.max()
      
    maxVal -= minVal
      
    v = ((v - minVal)/maxVal) * 255
    
    return v.astype(np.uint8)

def norm16bit(v, min_val=None, max_val=None):
    """
    Normalize the input array to 16-bit range [0, 65535].

    Args:
        v (numpy.ndarray): Input array to be normalized.
        min_val (float or None): Minimum value of the range. If None, the minimum value of the array is used.
        max_val (float or None): Maximum value of the range. If None, the maximum value of the array is used.

    Returns:
        numpy.ndarray: Normalized array in the range [0, 65535] with data type uint16.

    """
    if min_val is None:
        min_val = v.min()

    if max_val is None:
        max_val = v.max()

    max_val -= min_val

    v = ((v - min_val) / max_val) * 65535

    return v.astype(np.uint16)


def ratio_above_threshold(arr, threshold, start_range=None, end_range=None):
    """
    Calculates the ratio of numbers in `my_list` that are greater than or equal to `threshold`
    and fall within the range `[start_range, end_range]`.

    Args:
        arr (list or numpy array): A list or numpy array of numbers.
        threshold (int or float): The minimum value (inclusive) for a number to be counted in the ratio.
        start_range (int or float, optional): The minimum value for a number to be considered in the ratio.
            If not provided, the minimum value in `my_list` is used.
        end_range (int or float, optional): The maximum value for a number to be considered in the ratio.
            If not provided, the maximum value in `my_list` is used.

    Returns:
        float: The ratio of numbers in `my_list` that are greater than or equal to `threshold`
        and fall within the range `[start_range, end_range]`.

    Raises:
        ValueError: If `threshold` is not provided.

    """
    if threshold is None:
        raise ValueError("threshold must be provided.")

    arr = np.ravel(arr)  # Flatten the input array if it is multidimensional.

    if start_range is None:
        start_range = np.min(arr)

    if end_range is None:
        end_range = np.max(arr)

    sub_arr = np.array([x for x in arr if start_range <= x <= end_range])

    count = np.count_nonzero(sub_arr >= threshold)

    ratio = count / len(sub_arr)

    return ratio

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


def save_vol_as_slices(volume, folderName):
    """
    Save a 3D volume as individual 2D slices in a specified folder.

    Args:
        volume (ndarray): A 3D array representing the volume to be saved as 2D slices.
        folderName (str): The name of the folder where the slices will be saved.

    This function takes a 3D volume represented as a NumPy ndarray and saves 
    each individual 2D slice as a separate image file in the specified folder. 
    It creates the folder if it does not exist. Each slice is saved as a TIFF 
    image with a filename indicating the slice number.

    Example usage:
    volume_data = load_3d_volume("volume_data.npy")
    save_vol_as_slices(volume_data, "output_slices_folder")
    """
    depth = volume.shape[0]
    for aSlice in range(depth):
        img = volume[aSlice, :, :]
        if not os.path.exists(folderName):
            os.makedirs(folderName)
        fName = os.path.join(folderName, f'slice_{aSlice:04d}.tif') 
        tifffile.imwrite(fName, img)




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


def apply_morphology(arr, filter_type, filter_size=None, iterations=1):
    """
    Perform morphological operations on binary input arrays.

    Parameters
    ----------
    arr : numpy.ndarray
        Binary input array to apply morphological operations on.
    filter_type : str
        Type of morphological operation to apply. Available options are:
        - 'erosion'
        - 'dilation'
        - 'opening'
        - 'closing'
        - 'fill_holes'
    filter_size : list of ints, optional
        Size of the morphological filter. Defaults to None.
    iterations : int, optional
        Number of iterations to perform the morphological operation. Defaults to 1.

    Returns
    -------
    filtered_arr : numpy.ndarray
        Binary array after applying the morphological operation.

    Raises
    ------
    ValueError
        If the input array is not of binary data type.
        If the filter size is not None and does not have the same number of dimensions as the input array.
        If the filter size is not None and is greater than or equal to the size of the input array in each dimension.
        If the filter type is not a valid option.
    """
    
    if not np.issubdtype(arr.dtype, np.bool_):
        raise ValueError("Input array is not of binary data type.")
    
    if filter_size is not None and len(filter_size) != arr.ndim:
        raise ValueError("Filter size must have the same number of dimensions as the input array.")
    
    if filter_size is not None:
        for i in range(arr.ndim):
            if filter_size[i] >= arr.shape[i]:
                raise ValueError("Filter size must be smaller than the size of the input array in each dimension.")        
        size = np.ones(shape=filter_size)
    
    if filter_type == "erosion":
        filtered_arr = nd.binary_erosion(arr, structure=size, iterations=iterations)
    elif filter_type == "dilation":
        filtered_arr = nd.binary_dilation(arr, structure=size, iterations=iterations)
    elif filter_type == "opening":
        filtered_arr = nd.binary_opening(arr, structure=size, iterations=iterations)
    elif filter_type == "closing":
        filtered_arr = nd.binary_closing(arr, structure=size, iterations=iterations)
    elif filter_type == "fill_holes":
        filtered_arr = nd.binary_fill_holes(arr, structure=size)
    else:
        raise ValueError("Invalid filter type. Available options are 'erosion', 'dilation', 'opening', 'closing', and 'fill_holes'.")
    
    return filtered_arr


def apply_morphology_inorder(arr, morphology_operations=None):
    """
    Applies morphology operations to the input array in a specified order.

    Args:
    - arr (numpy.ndarray): Input array to which morphology operations will be applied.
    - morphology_operations (list): List of tuples, where each tuple represents a morphology operation
    to be applied to the input array. Each tuple should have 2 or 3 elements:
      - First element is a string representing the name of the morphology operation.
      - Second element is an integer representing the size of the structuring element or filter.
      - Third element (optional) is an integer representing the number of iterations for the morphology
      operation. This element is only required for erosion, dilation, opening, and closing operations.

    Returns:
    - morph_arr (numpy.ndarray): Output array after applying all specified morphology operations
    in the specified order.

    If morphology_operations is None, the function prints a message and returns None.

    """
    if morphology_operations:
        morph_arr = arr
        for operation in morphology_operations:
            if operation[0] == 'fill_holes':
                morph_arr = apply_morphology(morph_arr, filter_type=operation[0], filter_size=operation[1])
            else:
                # Apply other morphological operations
                morph_arr = apply_morphology(morph_arr, filter_type=operation[0], filter_size=operation[1], iterations=operation[2])

        return morph_arr
    else: 
        print('No morphology operations are specified.')
    