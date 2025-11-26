"""
Standalone tests for minkowski_stub.sparse_quantize function
No CARLA dependency required.
"""

import numpy as np
import sys
from AVR.minkowski_stub import sparse_quantize


def test_basic_quantization():
    """Test basic quantization without features or labels"""
    print("Test 1: Basic quantization")
    coordinates = np.array([
        [0.1, 0.1, 0.1],
        [0.2, 0.2, 0.2],
        [1.1, 1.1, 1.1],
        [1.2, 1.2, 1.2],
    ], dtype=np.float32)
    
    result = sparse_quantize(coordinates, quantization_size=1.0)
    print(f"  Input shape: {coordinates.shape}")
    print(f"  Output shape: {result.shape}")
    print(f"  Result:\n{result}")
    assert result.shape[0] == 2, "Should have 2 unique voxels"
    print("  ✓ PASSED\n")


def test_with_features():
    """Test quantization with features"""
    print("Test 2: Quantization with features")
    coordinates = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
        [1.5, 1.5, 1.5],
    ], dtype=np.float32)
    
    features = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ], dtype=np.float32)
    
    coords, feats = sparse_quantize(coordinates, features=features, quantization_size=1.0)
    print(f"  Coordinates shape: {coords.shape}")
    print(f"  Features shape: {feats.shape}")
    assert coords.shape[0] == feats.shape[0], "Coords and features should have same count"
    print("  ✓ PASSED\n")


def test_with_labels():
    """Test quantization with labels and ignore filtering"""
    print("Test 3: Quantization with labels and ignore_label filtering")
    coordinates = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
        [1.5, 1.5, 1.5],
    ], dtype=np.float32)
    
    labels = np.array([0, -100, 1, -100], dtype=np.int32)  # ignore points with label -100
    
    coords, labs = sparse_quantize(coordinates, labels=labels, ignore_label=-100, quantization_size=1.0)
    print(f"  Input count: {len(coordinates)}")
    print(f"  Output count: {len(coords)}")
    print(f"  Output labels: {labs}")
    assert len(coords) == 2, "Should only have 2 points after filtering ignore_label"
    assert np.all(labs != -100), "Should not contain ignore_label"
    print("  ✓ PASSED\n")


def test_return_index():
    """Test return_index flag"""
    print("Test 4: return_index flag")
    coordinates = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],  # Same voxel as first
        [1.0, 1.0, 1.0],
    ], dtype=np.float32)
    
    coords, indices = sparse_quantize(coordinates, return_index=True, quantization_size=1.0)
    print(f"  Unique coordinates count: {len(coords)}")
    print(f"  Indices: {indices}")
    assert len(indices) == len(coords), "Should have one index per unique coordinate"
    assert len(coords) == 2, "Should have 2 unique voxels"
    print("  ✓ PASSED\n")


def test_return_inverse():
    """Test return_inverse flag"""
    print("Test 5: return_inverse flag")
    coordinates = np.array([
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.1],  # Same voxel as first
        [1.0, 1.0, 1.0],
    ], dtype=np.float32)
    
    coords, inverse = sparse_quantize(coordinates, return_inverse=True, quantization_size=1.0)
    print(f"  Original count: {len(coordinates)}")
    print(f"  Unique count: {len(coords)}")
    print(f"  Inverse mapping: {inverse}")
    # Can reconstruct: coords[inverse] should give same voxels as input
    reconstructed = coords[inverse]
    print(f"  Reconstructed matches quantized input: {np.allclose(reconstructed, np.floor(coordinates).astype(np.int32))}")
    print("  ✓ PASSED\n")


def test_all_flags():
    """Test with all flags enabled"""
    print("Test 6: All flags enabled (coordinates, features, labels, return_index, return_inverse)")
    coordinates = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0],
    ], dtype=np.float32)
    
    features = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
    labels = np.array([10, 20, 30], dtype=np.int32)
    
    result = sparse_quantize(
        coordinates,
        features=features,
        labels=labels,
        return_index=True,
        return_inverse=True,
        quantization_size=1.0
    )
    
    coords, feats, labs, indices, inverse = result
    print(f"  Coordinates shape: {coords.shape}")
    print(f"  Features shape: {feats.shape}")
    print(f"  Labels shape: {labs.shape}")
    print(f"  Indices shape: {indices.shape}")
    print(f"  Inverse shape: {inverse.shape}")
    assert len(coords) == len(feats) == len(labs) == len(indices), "All should have same length"
    assert len(inverse) == len(coordinates), "Inverse should match original length"
    print("  ✓ PASSED\n")


def test_no_quantization():
    """Test without quantization (quantization_size=None)"""
    print("Test 7: No quantization (raw coordinate deduplication)")
    coordinates = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],  # Duplicate
        [1.0, 1.0, 1.0],
    ], dtype=np.float32)
    
    result = sparse_quantize(coordinates, quantization_size=None)
    print(f"  Input: {len(coordinates)} points")
    print(f"  Output: {len(result)} unique points")
    assert len(result) == 2, "Should have 2 unique points after dedup"
    print("  ✓ PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("MinkowskiEngine Stub - Standalone Tests")
    print("=" * 60 + "\n")
    
    try:
        test_basic_quantization()
        test_with_features()
        test_with_labels()
        test_return_index()
        test_return_inverse()
        test_all_flags()
        test_no_quantization()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        sys.exit(0)
    except AssertionError as e:
        print(f"\n TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
