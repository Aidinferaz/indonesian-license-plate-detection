import gradio as gr
from ultralytics import YOLO
import torch

# Load your custom-trained YOLOv8 model (ONNX format)
# Make sure you have exported your model to 'best.onnx'
model = YOLO('best.onnx')

def license_plate_detector(image):
    """
    Takes an image, runs it through the YOLOv8 model,
    and returns the image with bounding boxes drawn on it.
    """
    # Run inference
    results = model(image)

    # Plot the results on the image
    # The plot() method returns a NumPy array (image) with boxes, labels, and scores
    annotated_image = results[0].plot()

    return annotated_image

# Create the Gradio interface
iface = gr.Interface(
    fn=license_plate_detector,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Image(type="pil", label="Detection Result"),
    title="🇮🇩 Indonesian License Plate Detector",
    description="Upload an image of a vehicle to detect its license plate. This model is based on YOLOv8n.",
    examples=[['example_image.jpg']] # Optional: add an example image file
)

# Launch the app
iface.launch()