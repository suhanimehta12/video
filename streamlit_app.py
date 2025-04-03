import streamlit as st
import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import tempfile
import os
from io import BytesIO

# Set up page configurations
st.set_page_config(page_title="Video Captioning App", layout="wide")

# Create a Blip model and processor for captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    caption_ids = model.generate(**inputs)
    caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
    return caption

def process_video_with_caption(video_path, processor, model):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return None

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Temporary output video file
    output_video_path = os.path.join(tempfile.mkdtemp(), "output_video_with_caption.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Process the first frame to generate a caption
    ret, frame = cap.read()
    if not ret:
        st.error("

