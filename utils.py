# -*- coding: utf-8 -*-
"""
Spyder Editor

@author: Mmr Sagar
PhD Researcher | MPI-NAT Goettingen, Germany
"""

import numpy as np

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
