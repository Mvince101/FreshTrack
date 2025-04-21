from flask import Flask, Response, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load the YOLO model
model = YOLO(r"FreshnessDetection\YOLOv8_fruits_veg\weights\best.pt")

# Live video capture endpoint
@app.route('/live', methods=['GET'])
def live():
    def generate_frames():
        cap = cv2.VideoCapture(0)  # Open webcam (use 0 for default camera)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run the model on the frame
            results = model(frame)
            annotated_frame = results[0].plot()  # Annotate the frame with detections

            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()

            # Yield the frame as part of an HTTP response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Read the image from the request
    file = request.files['image']
    np_img = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Run the model
    results = model(img)
    detections = results[0].boxes.xyxy.tolist()  # Bounding boxes
    classes = results[0].boxes.cls.tolist()     # Class IDs
    scores = results[0].boxes.conf.tolist()     # Confidence scores

    return jsonify({
        "detections": detections,
        "classes": classes,
        "scores": scores
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)