"""
Minimal stub for MinkowskiEngine.utils.sparse_quantize
This provides the sparse_quantize function without requiring full MinkowskiEngine installation.
Based on the original MinkowskiEngine implementation.
"""

import numpy as np
from typing import Tuple, Union, Optional


def sparse_quantize(
    coordinates: np.ndarray,
    features: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    ignore_label: int = -100,
    return_index: bool = False,
    return_inverse: bool = False,
    quantization_size: Optional[Union[float, np.ndarray]] = None,
) -> Union[np.ndarray, Tuple]:
    """
    Given coordinates, and features (optionally labels), the function
    generates quantized (voxelized) coordinates and respective features and labels.
    
    Args:
        coordinates: a numpy array of size N x D where N is the number of points
                     and D is the dimension of the space.
        features: a numpy array of size N x F where F is the number of features per point.
        labels: a numpy array of size N. Points with label == ignore_label are filtered out.
        ignore_label: label value to ignore/filter out.
        return_index: if True, return the index of the unique coordinates.
        return_inverse: if True, return the inverse mapping.
        quantization_size: voxel size for quantization. If None, no quantization is applied.
    
    Returns:
        quantized_coordinates: voxelized coordinates (N' x D).
        unique_features: (if features provided) features corresponding to unique coordinates.
        unique_labels: (if labels provided) labels corresponding to unique coordinates.
        unique_indices: (if return_index=True) indices into original array.
        inverse_indices: (if return_inverse=True) inverse mapping for reconstruction.
    """
    coordinates = np.array(coordinates)
    features = np.array(features) if features is not None else None
    labels = np.array(labels) if labels is not None else None
    
    # Filter out ignored labels if labels are provided
    if labels is not None:
        valid_mask = labels != ignore_label
        coordinates = coordinates[valid_mask]
        if features is not None:
            features = features[valid_mask]
        labels = labels[valid_mask]
    
    # Apply quantization if specified
    if quantization_size is not None:
        if isinstance(quantization_size, (list, tuple, np.ndarray)):
            quantization_size = np.array(quantization_size)
        quantized_coords = np.floor(coordinates / quantization_size).astype(np.int32)
    else:
        quantized_coords = coordinates.astype(np.int32)
    
    # Find unique coordinates
    _, unique_indices, inverse_indices = np.unique(
        quantized_coords, axis=0, return_index=True, return_inverse=True
    )
    
    # Get unique features and labels
    unique_coords = quantized_coords[unique_indices]
    
    results = [unique_coords]
    
    if features is not None:
        unique_features = features[unique_indices]
        results.append(unique_features)
    
    if labels is not None:
        unique_labels = labels[unique_indices]
        results.append(unique_labels)
    
    if return_index:
        results.append(unique_indices)
    
    if return_inverse:
        results.append(inverse_indices)
    
    if len(results) == 1:
        return results[0]
    return tuple(results)
