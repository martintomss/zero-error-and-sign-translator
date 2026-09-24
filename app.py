"""
app.py - Agent 3: UI/UX & Web Integration Specialist
Flask Web Server, MJPEG Camera Stream, Real-Time Telemetry API
"""

import time
import json
import threading
from flask import Flask, render_template, Response, jsonify, request

from camera import HandTrackingPipeline

app = Flask(__name__)

# Global camera tracking pipeline
pipeline = HandTrackingPipeline(camera_index=0)
pipeline.start()

def generate_mjpeg_stream():
    """Generator yielding MJPEG frame stream to web clients with graceful disconnect handling."""
    try:
        while pipeline.is_running:
            frame_bytes = pipeline.get_jpeg()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                time.sleep(0.03)
    except (GeneratorExit, ConnectionResetError, BrokenPipeError):
        pass
    except Exception:
        pass

@app.route('/')
def index():
    """Renders the main Zero-Error Real-Time Hand Sign Translator dashboard."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """MJPEG streaming route for the video viewport."""
    return Response(generate_mjpeg_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/telemetry')
def api_telemetry():
    """Returns current real-time AI and stabilization telemetry with training metrics."""
    return jsonify(pipeline.last_result)

@app.route('/api/practice/catalog')
def api_practice_catalog():
    """Returns training catalog of all 18 hand signs with step-by-step instructions."""
    catalog = []
    for c in pipeline.classifier.CLASSES:
        catalog.append({
            "id": c["id"],
            "label": c["label"],
            "icon": c["icon"],
            "type": c.get("type", "general"),
            "instruction": c.get("instruction", ""),
            "finger_hints": c.get("finger_hints", {}),
            "practice_tip": c.get("practice_tip", "")
        })
    return jsonify({
        "current_target": pipeline.target_practice_id,
        "catalog": catalog
    })

@app.route('/api/practice/select', methods=['POST'])
def api_practice_select():
    """Sets the active training practice target sign."""
    try:
        data = request.get_json(force=True)
        target_id = data.get("target_id")
        if target_id:
            pipeline.set_practice_target(target_id)
            return jsonify({"status": "ok", "target_id": target_id})
        return jsonify({"error": "Missing target_id"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict_landmarks', methods=['POST'])
def api_predict_landmarks():
    """
    Direct inference endpoint for raw landmarks input (21 points [x, y, z]).
    Useful for browser client-side MediaPipe landmark submission.
    """
    try:
        data = request.get_json(force=True)
        raw_pts = data.get('landmarks', [])
        if len(raw_pts) != 21:
            return jsonify({"error": "Expected 21 landmarks"}), 400

        stab_res = pipeline.stabilizer.process(raw_pts)
        pred = pipeline.classifier.predict(
            stab_res["canonical_vector"],
            stab_res["canonical_points"],
            is_locked=stab_res["is_locked"]
        )

        return jsonify({
            "stabilization": {
                "stability_score": stab_res["stability_score"],
                "is_locked": stab_res["is_locked"],
                "palm_scale": stab_res["palm_scale"],
                "canonical_points": stab_res["canonical_points"].tolist(),
                "floating_points": stab_res["floating_canvas_points"].tolist()
            },
            "prediction": pred
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/status')
def api_status():
    """Healthcheck endpoint."""
    return jsonify({
        "status": "online",
        "engine": "Zero-Error Real-Time Hand Sign Translator",
        "zero_error_threshold": 0.88,
        "anti_gravity_stabilization": "active",
        "kalman_filter": "active",
        "fps": pipeline.fps
    })

if __name__ == '__main__':
    print("[Server] Launching Zero-Error Hand Sign Translator on http://localhost:5000...")
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
