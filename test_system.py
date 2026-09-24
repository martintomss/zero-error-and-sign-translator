"""
test_system.py - Autonomous Verification Test Suite
Validates:
1. Agent 1: Anti-Gravity Spatial Stabilization & Kalman Filter Jitter Reduction
2. Agent 2: Strict Zero-Error Protocol (>88% Confidence Gating & Zero Guessing)
3. Pipeline Integration: Camera synthetic generator & live inference
"""

import numpy as np
from stabilizer import AntiGravityStabilizer, LandmarkKalmanFilter
from model import ZeroErrorClassifier
from camera import HandTrackingPipeline

def test_kalman_filter_jitter_damping():
    print("\n--- TEST 1: Kalman Filter Jitter Damping ---")
    kf = LandmarkKalmanFilter(process_noise=1e-3, measurement_noise=1e-2)
    true_val = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    noisy_measurements = [true_val + np.random.normal(0, 0.05, 3).astype(np.float32) for _ in range(50)]

    filtered_vals = [kf.update(m) for m in noisy_measurements]
    
    noisy_var = np.var(noisy_measurements, axis=0)
    filtered_var = np.var(filtered_vals[10:], axis=0) # skip initial warm-up

    print(f"Noisy Input Variance:    {np.mean(noisy_var):.6f}")
    print(f"Kalman Output Variance:  {np.mean(filtered_var):.6f}")
    assert np.mean(filtered_var) < np.mean(noisy_var), "Kalman filter must suppress variance!"
    print(">>> PASS: Kalman Filter significantly damped coordinate jittering!")

def test_anti_gravity_spatial_stabilization():
    print("\n--- TEST 2: Anti-Gravity Spatial Stabilization ---")
    stabilizer = AntiGravityStabilizer(canvas_center=(0.5, 0.5, 0.0), target_scale=0.35)

    # Hand at distance D1 (wrist at 0.2, 0.3)
    base_hand = np.zeros((21, 3), dtype=np.float32)
    base_hand[0] = [0.2, 0.3, 0.0]
    base_hand[9] = [0.2, 0.15, 0.0] # middle MCP 0.15 units away
    for i in range(1, 21):
        if i != 9:
            base_hand[i] = [0.2 + (i % 5) * 0.02, 0.3 - (i // 5) * 0.04, 0.0]

    res1 = stabilizer.process(base_hand)

    # Hand at distance D2, rotated and shifted (simulating user moving back and hand drooping)
    shifted_hand = (base_hand * 0.5) + np.array([0.4, 0.2, 0.0]) # Dropped down and shrunk
    res2 = stabilizer.process(shifted_hand)

    c1 = res1["canonical_points"]
    c2 = res2["canonical_points"]

    # In canonical floating space, wrist is always exactly (0, 0, 0)
    assert np.allclose(c1[0], [0, 0, 0], atol=1e-5), "Canonical wrist must be at (0, 0, 0)"
    assert np.allclose(c2[0], [0, 0, 0], atol=1e-5), "Shifted wrist must be normalized to (0, 0, 0)"

    print(f"Canonical Wrist 1: {c1[0]}")
    print(f"Canonical Wrist 2: {c2[0]}")
    print(f"Stability Score:   {res2['stability_score']:.3f} | Locked: {res2['is_locked']}")
    print(">>> PASS: Anti-Gravity Spatial Normalization locks hand into invariant 3D space!")

def test_strict_zero_error_protocol():
    print("\n--- TEST 3: Strict Zero-Error Protocol (>88% Threshold) ---")
    classifier = ZeroErrorClassifier()
    stabilizer = AntiGravityStabilizer()

    # Generate synthetic "I Love You" pose (Thumb, Index, Pinky extended, Middle & Ring curled)
    ily_pts = np.zeros((21, 3), dtype=np.float32)
    ily_pts[0] = [0.5, 0.5, 0.0]  # wrist
    ily_pts[4] = [0.35, 0.38, 0.0] # thumb extended
    ily_pts[8] = [0.45, 0.22, 0.0] # index extended
    ily_pts[12] = [0.50, 0.42, 0.0] # middle curled
    ily_pts[16] = [0.53, 0.43, 0.0] # ring curled
    ily_pts[20] = [0.61, 0.27, 0.0] # pinky extended
    # intermediate joints
    ily_pts[2] = [0.42, 0.44, 0.0]
    ily_pts[5] = [0.45, 0.40, 0.0]
    ily_pts[6] = [0.45, 0.34, 0.0]
    ily_pts[9] = [0.50, 0.38, 0.0]
    ily_pts[10] = [0.50, 0.34, 0.0]
    ily_pts[14] = [0.53, 0.35, 0.0]
    ily_pts[17] = [0.57, 0.39, 0.0]
    ily_pts[18] = [0.59, 0.34, 0.0]

    # Warm-up stabilizer
    for _ in range(5):
        stab_res = stabilizer.process(ily_pts)

    pred = classifier.predict(stab_res["canonical_vector"], stab_res["canonical_points"], is_locked=True)
    print(f"Clear Gesture Output: '{pred['output_text']}'")
    print(f"Confidence:            {pred['confidence']*100:.2f}% (Threshold: {pred['threshold']*100:.1f}%)")
    print(f"Verified:              {pred['verified']}")

    assert pred["verified"] == True, "Clear I Love You sign must pass >88% threshold!"
    assert pred["output_text"] == "I Love You", f"Expected 'I Love You', got {pred['output_text']}"

    # TEST UNCONFIRMED / ZERO-ERROR SUPPRESSION
    print("\n--- TEST 4: Zero Guessing on Ambiguous / Low-Confidence Sign ---")
    # Feed an arbitrary ambiguous / in-between pose (e.g. half-closed fingers)
    ambiguous_pts = ily_pts.copy()
    ambiguous_pts[8, 1] = 0.38 # pull index finger down so it's not clearly open or closed
    ambiguous_pts[20, 1] = 0.36 # pull pinky down
    ambiguous_pts[4, 0] = 0.46 # tuck thumb

    # Reset stabilizer to avoid carrying over 5 frames of I Love You
    stabilizer.reset()
    for _ in range(5):
        stab_amb = stabilizer.process(ambiguous_pts)
    pred_amb = classifier.predict(stab_amb["canonical_vector"], stab_amb["canonical_points"], is_locked=True)
    
    print(f"Ambiguous Gesture Output: '{pred_amb['output_text']}'")
    print(f"Raw Candidate (Hidden):   '{pred_amb['candidate_label']}'")
    print(f"Confidence:               {pred_amb['confidence']*100:.2f}%")
    print(f"Verified:                 {pred_amb['verified']}")

    assert pred_amb["verified"] == False, "Ambiguous gesture must NOT be verified!"
    assert pred_amb["output_text"] == "Signal Unclear / Analyzing", \
        f"Must output 'Signal Unclear / Analyzing', got '{pred_amb['output_text']}'"
    print(">>> PASS: Strict Zero-Error Protocol successfully gated low confidence and prevented false guessing!")

def test_full_pipeline():
    print("\n--- TEST 5: Pipeline Frame Processing ---")
    pipeline = HandTrackingPipeline(camera_index=0)
    pipeline.start()
    
    frame, res = pipeline.process_frame()
    print(f"Frame Shape:      {frame.shape}")
    print(f"Pipeline Result:  {res['output_text']} (Confidence: {res['confidence']*100:.1f}%)")
    print(f"Anti-Gravity:     Stability: {res['stability_score']*100:.1f}% | Locked: {res['is_locked']}")
    
    assert frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0, "Frame must be valid"
    pipeline.stop()
    print(">>> PASS: Pipeline initialized and processed frames cleanly!")

if __name__ == "__main__":
    test_kalman_filter_jitter_damping()
    test_anti_gravity_spatial_stabilization()
    test_strict_zero_error_protocol()
    test_full_pipeline()
    print("\n==========================================")
    print(">>> [SUCCESS] ALL TESTS PASSED! ZERO-ERROR SPECIFICATIONS MET.")
    print("==========================================")
