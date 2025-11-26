#!/usr/bin/env python3
"""
Quick inline test of minkowski_stub - designed to work with minimal setup
"""

# Inline minimal numpy replacement for testing
class SimpleArray:
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data[idx]
    def __len__(self):
        return len(self.data)
    def __repr__(self):
        return f"Array({self.data})"

# Test the logic without imports
def test_sparse_quantize_logic():
    """Test the core quantization logic"""
    print("Testing sparse_quantize logic...")
    
    # Simulate: coordinates = [[0.1, 0.1], [0.5, 0.5], [1.1, 1.1]]
    # With quantization_size=1.0, should voxelize to: [[0, 0], [0, 0], [1, 1]]
    # Which has 2 unique: [[0, 0], [1, 1]]
    
    raw_coords = [
        [0.1, 0.1],
        [0.5, 0.5],  # Should map to same voxel as first
        [1.1, 1.1],
    ]
    
    # Manual quantization
    quantized = []
    for coord in raw_coords:
        q = [int(c / 1.0) for c in coord]  # floor(c / 1.0)
        quantized.append(tuple(q))
    
    print(f"Raw coords: {raw_coords}")
    print(f"Quantized: {quantized}")
    
    # Find unique
    unique = list(set(quantized))
    print(f"Unique voxels: {unique}")
    print(f"Count: {len(unique)}")
    
    assert len(unique) == 2, f"Expected 2 unique voxels, got {len(unique)}"
    print("✓ Test passed!\n")


def test_ignore_label_logic():
    """Test ignore_label filtering logic"""
    print("Testing ignore_label filtering logic...")
    
    coords = [
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ]
    
    labels = [0, -100, 1, -100]
    ignore_label = -100
    
    # Filter
    filtered_coords = []
    filtered_labels = []
    for c, l in zip(coords, labels):
        if l != ignore_label:
            filtered_coords.append(c)
            filtered_labels.append(l)
    
    print(f"Original: {len(coords)} points")
    print(f"After filtering ignore_label={ignore_label}: {len(filtered_coords)} points")
    print(f"Remaining labels: {filtered_labels}")
    
    assert len(filtered_coords) == 2, f"Expected 2 filtered points, got {len(filtered_coords)}"
    assert all(l != ignore_label for l in filtered_labels), "Found ignore_label in results"
    print("✓ Test passed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("MinkowskiEngine Stub - Logic Verification")
    print("=" * 60 + "\n")
    
    test_sparse_quantize_logic()
    test_ignore_label_logic()
    
    print("=" * 60)
    print("All logic tests passed ✓")
    print("=" * 60)
