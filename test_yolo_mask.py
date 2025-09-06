import cv2
import numpy as np
from ultralytics import YOLO

# Path to your YOLO model
MODEL_PATH = "C:\\Users\\Aksha\\OneDrive\\Documents\\Projects\\Solar\\yolov8n_translucent_best.pt"  # Update if needed
IMAGE_PATH = "C:\\Users\\Aksha\\Downloads\\panel10_solar_Fri_Jun_16_11__1__20_2017_L_0.885687170475_I_0.557843137255_8.jpg"  # Update to your test image

# Function to get merged mask from YOLO segmentation
def yolo_segmentation_predict(img, model):
    """Run YOLO segmentation and return a single merged binary mask."""
    results = model.predict(img, verbose=False,conf=0.25)
    total_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for r in results:
        if r.masks is not None:
            for m in r.masks.data:
                m_resized = cv2.resize(
                    m.cpu().numpy().astype(np.uint8),
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                total_mask = np.maximum(total_mask, m_resized)  # union of all masks
    return total_mask

# Load model
model = YOLO(MODEL_PATH)

# Read image
image = cv2.imread(IMAGE_PATH)
if image is None:
    print(f"Failed to load image: {IMAGE_PATH}")
    exit(1)

# Get merged mask
merged_mask = yolo_segmentation_predict(image, model)
print(f"Merged mask shape: {merged_mask.shape}, dtype: {merged_mask.dtype}")
cv2.imwrite("merged_mask.png", merged_mask)
