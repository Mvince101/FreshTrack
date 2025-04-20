from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Load the YOLO model
model = YOLO(r"FreshnessDetection\YOLOv8_fruits_veg\weights\best.pt")

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