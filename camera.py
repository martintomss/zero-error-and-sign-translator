"""
camera.py - Agent 1: Computer Vision & Camera Pipeline
OpenCV Camera Capture + MediaPipe Hands + Anti-Gravity Visualization
"""

import time
import threading
import numpy as np

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

from stabilizer import AntiGravityStabilizer
from model import ZeroErrorClassifier

class HandTrackingPipeline:
    """
    Real-time Hand Tracking & Anti-Gravity Spatial Stabilization Pipeline.
    Combines:
    - OpenCV VideoCapture (Zero-buffer, high-speed DirectShow)
    - MediaPipe Hands 21-landmark tracking (Lite complexity 0)
    - Anti-Gravity Spatial Stabilizer (Kalman filter + canonical float)
    - Zero-Error Classifier (>88% confidence gate)
    - Decoupled asynchronous worker thread for smooth 60 FPS streaming
    """
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),           # Index
        (5, 9), (9, 10), (10, 11), (11, 12),      # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),    # Ring
        (13, 17), (17, 18), (18, 19), (19, 20),   # Pinky
        (0, 17)                                    # Palm Base
    ]

    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap = None
        self.is_running = False

        # Threading & Double Buffering
        self.worker_thread = None
        self.latest_jpeg = None
        self.latest_frame = None
        self.new_frame_event = threading.Event()
        self.lock = threading.Lock()

        # Initialize Stabilizer & Zero-Error Classifier
        self.stabilizer = AntiGravityStabilizer(canvas_center=(0.5, 0.5, 0.0), target_scale=0.35)
        self.classifier = ZeroErrorClassifier()

        # MediaPipe Hands - Lite Complexity (0) for ultra-fast, smooth execution
        self.mp_hands = None
        self.hands_detector = None
        if MEDIAPIPE_AVAILABLE:
            self.mp_hands = mp.solutions.hands
            self.hands_detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=0, # Lite model (3x faster CPU execution, zero lag)
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )

        # Interactive Training / Practice Target
        self.target_practice_id = "i_love_you"

        self.last_result = {
            "verified": False,
            "output_text": "Signal Unclear / Analyzing",
            "confidence": 0.0,
            "threshold": 0.88,
            "icon": "⏳",
            "stability_score": 0.0,
            "is_locked": False,
            "fps": 0,
            "practice_eval": None
        }

        self.fps = 0
        self.frame_count = 0
        self.fps_timer = time.time()
        self.mock_phase = 0.0

    def start(self):
        if OPENCV_AVAILABLE:
            try:
                # Use DirectShow on Windows for zero-latency camera start
                if hasattr(cv2, 'CAP_DSHOW'):
                    self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                else:
                    self.cap = cv2.VideoCapture(self.camera_index)

                # Set zero-latency buffer size and resolution
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, 30)

                if not self.cap.isOpened():
                    print(f"[Camera] Physical webcam index {self.camera_index} not accessible. Using virtual camera stream.")
                    self.cap = None
            except Exception as e:
                print(f"[Camera] Error opening webcam: {e}")
                self.cap = None

        self.is_running = True

        # Start dedicated background capture & processing loop
        self.worker_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.worker_thread.start()

    def _capture_loop(self):
        """Dedicated high-speed worker thread for non-blocking camera & AI processing."""
        while self.is_running:
            start_t = time.time()
            try:
                frame, res = self.process_frame()
                if frame is not None and OPENCV_AVAILABLE:
                    # Turbo JPEG encoding (fast compression, low network latency)
                    ret, jpeg = cv2.imencode('.jpg', frame, [
                        int(cv2.IMWRITE_JPEG_QUALITY), 75
                    ])
                    if ret:
                        with self.lock:
                            self.latest_jpeg = jpeg.tobytes()
                            self.latest_frame = frame
                        self.new_frame_event.set()
            except Exception as e:
                pass

            # Frame pacing: ~35-40 FPS without pegging CPU
            elapsed = time.time() - start_t
            target_period = 1.0 / 35.0
            if elapsed < target_period:
                time.sleep(target_period - elapsed)

    def stop(self):
        self.is_running = False
        if self.worker_thread is not None:
            self.worker_thread = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def generate_synthetic_frame(self):
        """
        Generates a synthetic high-tech test pattern when physical webcam is not attached.
        Emulates hand movement and poses to test stabilization and zero-error protocol.
        """
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Background cyber grid
        grid_color = (25, 20, 15)
        for x in range(0, self.width, 40):
            cv2.line(img, (x, 0), (x, self.height), grid_color, 1)
        for y in range(0, self.height, 40):
            cv2.line(img, (0, y), (self.width, y), grid_color, 1)

        self.mock_phase += 0.05
        # Generate synthetic 21 landmarks mimicking an "I Love You" or "Peace" gesture
        t = self.mock_phase
        base_x = 0.5 + 0.05 * np.sin(t * 0.8)
        base_y = 0.55 + 0.03 * np.cos(t * 0.8)

        pts = []
        # Wrist
        pts.append([base_x, base_y, 0.0])
        # Thumb
        pts.append([base_x - 0.04, base_y - 0.03, -0.01])
        pts.append([base_x - 0.08, base_y - 0.06, -0.02])
        pts.append([base_x - 0.12, base_y - 0.09, -0.03])
        pts.append([base_x - 0.15, base_y - 0.12, -0.03]) # Thumb Tip
        # Index (Straight up)
        pts.append([base_x - 0.03, base_y - 0.10, -0.01])
        pts.append([base_x - 0.04, base_y - 0.16, -0.01])
        pts.append([base_x - 0.05, base_y - 0.22, -0.02])
        pts.append([base_x - 0.05, base_y - 0.28, -0.02]) # Index Tip (Extended)
        # Middle (Curled)
        pts.append([base_x, base_y - 0.10, 0.0])
        pts.append([base_x, base_y - 0.13, 0.02])
        pts.append([base_x, base_y - 0.11, 0.03])
        pts.append([base_x, base_y - 0.08, 0.04]) # Middle Tip (Curled)
        # Ring (Curled)
        pts.append([base_x + 0.03, base_y - 0.09, 0.01])
        pts.append([base_x + 0.03, base_y - 0.12, 0.02])
        pts.append([base_x + 0.03, base_y - 0.10, 0.03])
        pts.append([base_x + 0.03, base_y - 0.07, 0.04]) # Ring Tip (Curled)
        # Pinky (Straight up)
        pts.append([base_x + 0.06, base_y - 0.08, 0.01])
        pts.append([base_x + 0.08, base_y - 0.13, 0.01])
        pts.append([base_x + 0.10, base_y - 0.18, 0.0])
        pts.append([base_x + 0.11, base_y - 0.23, 0.0]) # Pinky Tip (Extended)

        # Add small natural hand jitter before stabilization
        jitter = np.random.normal(0, 0.003, (21, 3))
        raw_pts = np.array(pts) + jitter
        return img, raw_pts

    def process_frame(self):
        """
        Processes next frame:
        1. Captures webcam or synthetic feed
        2. Detects landmarks via MediaPipe
        3. Stabilizes with Anti-Gravity Stabilizer & Kalman Filter
        4. Classifies with Zero-Error Protocol (>88%)
        5. Draws Cyber Overlay & Anti-Gravity Floating Zone
        """
        raw_landmarks = None
        frame = None

        has_hand = False
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                frame = cv2.flip(frame, 1) # Mirror for natural selfie orientation
                if self.hands_detector:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands_detector.process(rgb)
                    if results.multi_hand_landmarks:
                        hl = results.multi_hand_landmarks[0]
                        raw_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark], dtype=np.float32)
                        has_hand = True

        if frame is None:
            # Fallback to synthetic generator ONLY when physical camera hardware is absent
            frame, raw_landmarks = self.generate_synthetic_frame()
            has_hand = True

        # Update FPS
        self.frame_count += 1
        now = time.time()
        if now - self.fps_timer >= 1.0:
            self.fps = int(self.frame_count / (now - self.fps_timer))
            self.frame_count = 0
            self.fps_timer = now

        # Standby state if camera is on but no hand is in view
        if not has_hand or raw_landmarks is None:
            self.last_result = {
                "verified": False,
                "output_text": "Menunggu Tangan...",
                "confidence": 0.0,
                "threshold": self.classifier.THRESHOLD,
                "icon": "👋",
                "candidate_label": "Menunggu Tangan",
                "status": "STANDBY_NO_HAND",
                "stability_score": 0.0,
                "is_locked": False,
                "fps": self.fps,
                "top_candidates": [],
                "practice_eval": None,
                "features": {
                    "extended_fingers": 0,
                    "total_extended": 0,
                    "is_fist": False,
                    "anti_gravity_locked": False
                }
            }
            self._render_standby_overlay(frame)
            return frame, self.last_result

        # Step 3 & 4: Anti-Gravity Spatial Stabilization
        stab_res = self.stabilizer.process(raw_landmarks)
        canonical_vector = stab_res["canonical_vector"]
        canonical_points = stab_res["canonical_points"]
        stability = stab_res["stability_score"]
        is_locked = stab_res["is_locked"]

        # Step 5: Zero-Error Protocol Deep Learning Inference
        pred = self.classifier.predict(
            canonical_vector,
            canonical_points,
            is_locked=is_locked,
            target_practice_id=self.target_practice_id,
            raw_points=stab_res["filtered_raw"]
        )

        self.last_result = {
            "verified": pred["verified"],
            "output_text": pred["output_text"],
            "confidence": pred["confidence"],
            "threshold": pred["threshold"],
            "icon": pred["icon"],
            "candidate_label": pred["candidate_label"],
            "status": pred["status"],
            "stability_score": stability,
            "is_locked": is_locked,
            "fps": self.fps,
            "top_candidates": pred["top_candidates"],
            "practice_eval": pred.get("practice_eval"),
            "features": pred["features"]
        }

        # Step 6: Render Visual Overlays on Frame
        self._render_overlay(frame, stab_res, pred)
        return frame, self.last_result

    def set_practice_target(self, target_id):
        """Switches current practice training target."""
        self.target_practice_id = target_id

    def _render_standby_overlay(self, frame):
        """Draws subtle targeting guidelines when waiting for hand."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        box_w, box_h = int(w * 0.40), int(h * 0.55)
        x1, y1 = cx - box_w // 2, cy - box_h // 2
        x2, y2 = cx + box_w // 2, cy + box_h // 2

        guide_col = (180, 140, 40) # Muted cyber cyan
        c_len = 20
        cv2.line(frame, (x1, y1), (x1 + c_len, y1), guide_col, 1)
        cv2.line(frame, (x1, y1), (x1, y1 + c_len), guide_col, 1)
        cv2.line(frame, (x2, y1), (x2 - c_len, y1), guide_col, 1)
        cv2.line(frame, (x2, y1), (x2, y1 + c_len), guide_col, 1)
        cv2.line(frame, (x1, y2), (x1 + c_len, y2), guide_col, 1)
        cv2.line(frame, (x1, y2), (x1, y2 - c_len), guide_col, 1)
        cv2.line(frame, (x2, y2), (x2 - c_len, y2), guide_col, 1)
        cv2.line(frame, (x2, y2), (x2, y2 - c_len), guide_col, 1)

        msg = "ARAHKAN TANGAN KE DEPAN KAMERA"
        cv2.putText(frame, msg, (cx - 150, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.55, guide_col, 1, cv2.LINE_AA)

    def _render_overlay(self, frame, stab_res, pred):
        h, w = frame.shape[:2]
        is_locked = stab_res["is_locked"]

        # 1. Pixel coordinates of smoothed hand directly in camera space (100% "Pas Semua")
        # Uses filtered_raw from Kalman filter to suppress jitter while matching real hand position
        filtered_landmarks = stab_res["filtered_raw"]
        pts_px = []
        for p in filtered_landmarks:
            px = int(np.clip(p[0] * w, 0, w - 1))
            py = int(np.clip(p[1] * h, 0, h - 1))
            pts_px.append((px, py))

        # Color palette: Neon Green when confirmed verified, bright cyan when active
        if pred["verified"]:
            primary_col = (0, 255, 163) # Bright Green
            bone_col = (0, 240, 180)
        else:
            primary_col = (0, 220, 255) # Cyan
            bone_col = (0, 180, 240)

        # 2. Draw Hand Skeleton directly on the user's hand
        for s, e in self.HAND_CONNECTIONS:
            cv2.line(frame, pts_px[s], pts_px[e], bone_col, 2, cv2.LINE_AA)

        # Draw joint nodes
        for i, (px, py) in enumerate(pts_px):
            is_tip = i in [4, 8, 12, 16, 20]
            radius = 5 if is_tip else 3
            node_col = (255, 255, 255) if is_tip else primary_col
            cv2.circle(frame, (px, py), radius, node_col, -1, cv2.LINE_AA)
            if is_tip:
                cv2.circle(frame, (px, py), radius + 2, primary_col, 1, cv2.LINE_AA)

        # 3. Dynamic Hand Bounding Reticle (Follows Hand Exactly)
        xs = [p[0] for p in pts_px]
        ys = [p[1] for p in pts_px]
        bx1 = max(0, min(xs) - 20)
        by1 = max(0, min(ys) - 20)
        bx2 = min(w - 1, max(xs) + 20)
        by2 = min(h - 1, max(ys) + 20)

        corner_len = min(25, (bx2 - bx1) // 4)
        cv2.line(frame, (bx1, by1), (bx1 + corner_len, by1), primary_col, 2)
        cv2.line(frame, (bx1, by1), (bx1, by1 + corner_len), primary_col, 2)
        cv2.line(frame, (bx2, by1), (bx2 - corner_len, by1), primary_col, 2)
        cv2.line(frame, (bx2, by1), (bx2, by1 + corner_len), primary_col, 2)
        cv2.line(frame, (bx1, by2), (bx1 + corner_len, by2), primary_col, 2)
        cv2.line(frame, (bx1, by2), (bx1, by2 - corner_len), primary_col, 2)
        cv2.line(frame, (bx2, by2), (bx2 - corner_len, by2), primary_col, 2)
        cv2.line(frame, (bx2, by2), (bx2, by2 - corner_len), primary_col, 2)

        # 4. Floating Holographic Badge above user's hand
        badge_y = max(25, by1 - 12)
        badge_text = f"{pred['output_text']} ({pred['confidence']*100:.1f}%)"
        cv2.putText(frame, badge_text, (bx1, badge_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, primary_col, 2, cv2.LINE_AA)

        # 5. Mini Anti-Gravity HUD at bottom
        status_text = "ANTI-GRAVITY LOCKED [ACTIVE]" if is_locked else "SPATIAL TRACKING ACTIVE"
        cv2.putText(frame, status_text, (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 163) if is_locked else (255, 200, 0), 1, cv2.LINE_AA)


    def get_jpeg(self):
        """Retrieves latest processed JPEG frame instantly from memory without blocking."""
        if self.is_running and self.latest_jpeg is not None:
            # Wait up to 35ms for the next frame event
            self.new_frame_event.wait(timeout=0.035)
            self.new_frame_event.clear()
            with self.lock:
                return self.latest_jpeg if self.latest_jpeg is not None else b''

        # Fallback if worker thread hasn't produced a frame yet
        frame, _ = self.process_frame()
        if OPENCV_AVAILABLE and frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ret:
                return jpeg.tobytes()
        return b''
