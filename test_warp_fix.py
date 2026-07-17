#!/usr/bin/env python3
"""
Test script to check if the WarpPy type annotation fix resolves the parsing error.
"""

import sys
sys.path.append('/home/bjgilhul/workspace/labwork/phd/occlusions/src/thirdParty/polycheck')

from polycheck.poly_warp import faux_scan
import numpy as np

def test_faux_scan():
    # Create a simple test polygon
    polygons = [np.array([[0, 0], [2, 0], [2, 2], [0, 2]])]

    # Test parameters
    origin = [1, 1]
    angle_start = 0
    angle_inc = 0.1
    num_rays = 10
    max_range = 5.0
    resolution = 0.05

    print("Testing faux_scan with WarpPy kernel...")

    try:
        scan_data, indices = faux_scan(
            polygons,
            origin=origin,
            angle_start=angle_start,
            angle_inc=angle_inc,
            num_rays=num_rays,
            max_range=max_range,
            resolution=resolution,
        )
        print("SUCCESS: faux_scan completed without errors")
        print(f"Scan data shape: {scan_data.shape}")
        print(f"Indices shape: {indices.shape}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    test_faux_scan()
