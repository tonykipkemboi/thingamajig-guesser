from PIL import Image
import numpy as np
import cv2

def run_object_detection(model: object, image_bytes: bytes) -> Image:
    """
    Run object detection on the provided image using the specified YOLOv5 model.
    Args:
        model (object): The YOLOv5 model to use for object detection.
        image_bytes (bytes): The image bytes to run object detection on.
    Returns:
        Image: Image with detected objects and bounding boxes rendered.
    """
    # Convert image bytes to numpy array in BGR format
    image_array = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Run object detection
    results = model(image_bgr)
    
    # Render detected objects and bounding boxes on the original image
    rendered_image = results.render()
    rendered_image_rgb = cv2.cvtColor(rendered_image[0], cv2.COLOR_BGR2RGB)
    return Image.fromarray(rendered_image_rgb)
