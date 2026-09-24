"""
stabilizer.py - Agent 1: Computer Vision & Pre-processing Specialist
Anti-Gravity Spatial Stabilization and Kalman Filter Jitter Reduction
"""

import numpy as np
import time

class LandmarkKalmanFilter:
    """
    Kalman Filter for single 3D point (x, y, z) with velocity state tracking.
    State vector: [x, y, z, vx, vy, vz]^T
    Measurement vector: [x, y, z]^T
    """
    def __init__(self, process_noise=1e-3, measurement_noise=1e-2):
        self.state = np.zeros(6, dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 1.0
        self.Q = np.eye(6, dtype=np.float32) * process_noise
        self.R = np.eye(3, dtype=np.float32) * measurement_noise
        self.H = np.zeros((3, 6), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.initialized = False
        self.last_time = time.time()

    def update(self, measurement):
        now = time.time()
        dt = max(0.001, min(0.1, now - self.last_time))
        self.last_time = now

        # Transition matrix F
        F = np.eye(6, dtype=np.float32)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt

        if not self.initialized:
            self.state[:3] = measurement
            self.state[3:] = 0.0
            self.initialized = True
            return self.state[:3].copy()

        # Predict
        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q

        # Update
        z = np.array(measurement, dtype=np.float32)
        y = z - (self.H @ self.state)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ y
        self.P = (np.eye(6, dtype=np.float32) - K @ self.H) @ self.P

        return self.state[:3].copy()

    def reset(self):
        self.initialized = False
        self.state = np.zeros(6, dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 1.0


class AntiGravityStabilizer:
    """
    Anti-Gravity Spatial Stabilization Engine:
    1. Kalman filtering on all 21 hand landmarks to suppress coordinate noise/jitter.
    2. Origin Translation: Anchors the hand relative to the wrist coordinate L0 -> (0, 0, 0).
    3. Scale Normalization: Normalizes scale using the palm dimension (wrist to middle MCP).
    4. Rotational Invariance: Aligns palm axis vertically to eliminate camera tilt and hand droop.
    5. Anti-Gravity Virtual Floating Canvas: Projects the stabilized canonical hand into a 
       virtual hovering 3D canvas centered in the viewport.
    """
    def __init__(self, canvas_center=(0.5, 0.5, 0.0), target_scale=0.32):
        self.filters = [LandmarkKalmanFilter() for _ in range(21)]
        self.canvas_center = np.array(canvas_center, dtype=np.float32)
        self.target_scale = float(target_scale)
        self.last_canonical = None
        self.stability_history = []

    def reset(self):
        for f in self.filters:
            f.reset()
        self.last_canonical = None
        self.stability_history = []

    def process(self, raw_landmarks):
        """
        Args:
            raw_landmarks: np.ndarray of shape (21, 3) or list of 21 points with [x, y, z]
        Returns:
            dict containing:
                - filtered_raw: (21, 3) smoothed landmarks in camera space
                - canonical_vector: (63,) 1D normalized feature vector for neural network
                - canonical_points: (21, 3) canonical hand points centered at (0, 0, 0)
                - floating_canvas_points: (21, 3) points positioned in the virtual anti-gravity viewport
                - stability_score: float [0.0, 1.0] indicating lock quality
                - is_locked: bool indicating hand is solidly anchored in anti-gravity zone
        """
        pts = np.array(raw_landmarks, dtype=np.float32)
        if pts.shape != (21, 3):
            raise ValueError(f"Expected landmarks of shape (21, 3), got {pts.shape}")

        # Step 1: Kalman Filter Smoothing on raw input
        filtered = np.zeros((21, 3), dtype=np.float32)
        for i in range(21):
            filtered[i] = self.filters[i].update(pts[i])

        # Step 2: Wrist-Relative Origin Anchoring
        wrist = filtered[0].copy()
        relative_pts = filtered - wrist

        # Step 3: Palm Scale Invariance (Wrist L0 to Middle MCP L9 distance)
        middle_mcp = relative_pts[9]
        palm_length = np.linalg.norm(middle_mcp[:2])
        if palm_length < 1e-4:
            palm_length = 0.1

        scaled_pts = relative_pts / palm_length

        # Step 4: Rotational Alignment (Locks hand vector to upright Y-axis)
        dx = scaled_pts[9, 0]
        dy = scaled_pts[9, 1]
        current_angle = np.arctan2(dy, dx)
        # We want middle MCP (L9) to point upward (-Y in screen space or +Y canonically)
        target_angle = -np.pi / 2.0
        rotation_angle = target_angle - current_angle

        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)
        rot_matrix_2d = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ], dtype=np.float32)

        canonical_pts = scaled_pts.copy()
        canonical_pts[:, :2] = (rot_matrix_2d @ scaled_pts[:, :2].T).T

        # Flip horizontally if palm is facing reversed (handedness normalization)
        if canonical_pts[5, 0] > canonical_pts[17, 0]:
            canonical_pts[:, 0] = -canonical_pts[:, 0]

        # Step 5: Anti-Gravity Floating 3D Projection
        # Project canonical points into the virtual hovering zone at canvas_center
        floating_pts = (canonical_pts * self.target_scale) + self.canvas_center

        # Step 6: Stability & Anti-Gravity Zone Lock Calculation
        if self.last_canonical is not None:
            movement_delta = np.mean(np.linalg.norm(canonical_pts - self.last_canonical, axis=1))
            # Smooth stability metric with resilient damping
            stability = float(np.exp(-movement_delta * 6.0))
        else:
            stability = 0.88
        self.last_canonical = canonical_pts.copy()

        self.stability_history.append(stability)
        if len(self.stability_history) > 8:
            self.stability_history.pop(0)

        smooth_stability = float(np.mean(self.stability_history))
        is_locked = smooth_stability >= 0.50


        return {
            "filtered_raw": filtered,
            "canonical_vector": canonical_pts.flatten(), # shape (63,)
            "canonical_points": canonical_pts,
            "floating_canvas_points": floating_pts,
            "palm_scale": float(palm_length),
            "stability_score": smooth_stability,
            "is_locked": is_locked
        }
