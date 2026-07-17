#!/usr/bin/env python3
"""Test multiple MPPI calls to verify different trajectories are generated."""

import sys

sys.path.append("/home/bjgilhul/workspace/labwork/phd/model_predictive/warp_mppi")

import numpy as np
from mppi_warp import WarpMPPI


def test_multiple_calls():
    """Test that multiple MPPI calls generate different trajectories."""
    print("Testing multiple MPPI calls for trajectory diversity...")

    # Create controller
    controller = WarpMPPI(
        samples=100,
        vehicle_length=2.5,
        u_limits=[3.0, 0.5],
        u_dist_limits=[0.5, 0.1],
        Q=[1.0, 1.0, 0.1, 0.1],
        Qf=[10.0, 10.0, 1.0, 1.0],
        R=[0.1, 0.1],
        debug=True,
    )

    # Common test data
    costmap = np.random.rand(50, 50).astype(np.float32)
    x_init = [0.0, 0.0, 1.0, 0.0]
    x_goal = [10.0, 0.0, 1.0, 0.0]
    x_nom = np.array([[i, 0.0, 1.0, 0.0] for i in range(11)])
    u_nom = np.array([[0.0, 0.0] for _ in range(10)])
    actors = [[5.0, 2.0, 1.0]]

    # Store results from multiple calls
    results = []

    for i in range(3):
        print(f"\n=== Call {i+1} ===")

        u_opt, u_samples, weights = controller.find_control(
            costmap=costmap,
            origin=(0.0, 0.0),
            resolution=0.1,
            x_init=x_init,
            x_goal=x_goal,
            x_nom=x_nom,
            u_nom=u_nom,
            actors=actors,
            dt=0.1,
        )

        results.append((u_opt.copy(), u_samples.copy(), weights.copy()))

        # Show first few control values for comparison
        print(f"First 3 optimal controls: {u_opt[:3].tolist()}")
        print(f"First sample's first 3 controls: {u_samples[0,:3].tolist()}")

    # Compare results
    print(f"\n=== Comparison ===")
    for i in range(1, len(results)):
        u_opt_diff = np.allclose(results[0][0], results[i][0], atol=1e-6)
        u_samples_diff = np.allclose(results[0][1], results[i][1], atol=1e-6)
        weights_diff = np.allclose(results[0][2], results[i][2], atol=1e-6)

        print(f"Call 1 vs Call {i+1}:")
        print(f"  Optimal controls identical: {u_opt_diff}")
        print(f"  Sample controls identical: {u_samples_diff}")
        print(f"  Weights identical: {weights_diff}")

    # Check if any results are identical (should be very unlikely)
    all_identical = all(
        [
            np.allclose(results[0][0], results[i][0], atol=1e-6)
            and np.allclose(results[0][1], results[i][1], atol=1e-6)
            for i in range(1, len(results))
        ]
    )

    if all_identical:
        print("\n❌ ERROR: All calls produced identical results!")
        return False
    else:
        print("\n✅ SUCCESS: Calls produced different results!")
        return True


if __name__ == "__main__":
    success = test_multiple_calls()
    if not success:
        exit(1)
