from roboflow import Roboflow
from ultralytics import YOLO
import cv2

rf = Roboflow(api_key="aMqBtXMVGpr6YULgE5No")
project = rf.workspace("college-74jj5").project("freshness-fruits-and-vegetables")
version = project.version(7)
dataset = version.download("yolov8") 

# model = YOLO("yolov8n.pt")


# model.train(
#     data=f"{dataset.location}/data.yaml",
#     epochs=5,       
#     imgsz=640,     
#     batch=16,       
#     project="FreshnessDetection", 
#     name="YOLOv8_fruits_veg",    
#     exist_ok=True   
# )



model = YOLO(r"FreshnessDetection\YOLOv8_fruits_veg\weights\best.pt")



cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

   
    results = model(frame)

   
    annotated_frame = results[0].plot()

   
    cv2.imshow("Freshness Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()