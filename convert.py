from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO('ultralytics/best.pt')

# Export the model to TFLite format
model.export(format='tflite')