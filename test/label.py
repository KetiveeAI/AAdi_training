import cv2
import os
from label_studio_sdk import Client
from label_studio_converter import LabelStudioConverter
from label_studio_converter.imports import brush

# Load your model (example with YOLO)
model = load_yolo_model()

def predict_and_convert(image_path):
    img = cv2.imread(image_path)
    results = model(img)  # Get predictions
    
    # Convert to Label Studio format
    ls_annotations = []
    for pred in results:
        ls_annotations.append({
            "type": "rectanglelabels",
            "value": {
                "x": pred.x, "y": pred.y, 
                "width": pred.width, "height": pred.height,
                "rectanglelabels": [pred.label]
            }
        })
    return ls_annotations