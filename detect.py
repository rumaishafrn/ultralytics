from ultralytics import YOLO
import cv2
import os

# Load the YOLOv8 model
model = YOLO("best.pt")  # Ensure the model file is in the same folder or provide the full path

# Create an output folder if it doesn't exist
output_folder = "output_folder"
os.makedirs(output_folder, exist_ok=True)

# Initialize the webcam
cap = cv2.VideoCapture(0)  # 0 is the default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get the video frame width, height, and frames per second (fps)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
output_video_path = os.path.join(output_folder, "detected_output3.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi files
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Perform detection
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Display the frame with detections
    cv2.imshow("YOLOv8 Money Detection", annotated_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()