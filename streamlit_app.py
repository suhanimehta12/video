import streamlit as st
import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import tempfile
import os
import logging

# Set logging for debugging (optional)
logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(page_title="Video Captioning App", layout="wide")

# Cache model loading to optimize performance
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_blip_model()

def generate_caption(image):
    text = "a picture of"
    inputs = processor(image, text, return_tensors="pt")
    caption_ids = model.generate(**inputs)
    caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
    return caption

def process_video_with_caption(video_path, processor, model):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video. Please upload a valid MP4 file.")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 24  # fallback

    output_video_path = os.path.join(tempfile.mkdtemp(), "output_video_with_caption.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Get the first frame to generate a caption
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Could not read a frame from the video.")
        return None

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    caption = generate_caption(pil_image)

    # Rewind video to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.putText(frame, caption, (50, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        out.write(frame)

    cap.release()
    out.release()

    return output_video_path

# ---- UI Styling ----
st.markdown("""
    <style>
        .stApp {
            background-color: #F4F8FC;
            padding: 20px;
        }
        .header {
            font-size: 3em;
            color: #333;
            font-weight: bold;
            text-align: center;
        }
        .description {
            font-size: 1.2em;
            color: #555;
            text-align: center;
            margin-top: 20px;
        }
        .button {
            background-color: #FF9800;
            color: white;
            border-radius: 10px;
            padding: 12px 25px;
            font-size: 1.2em;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            transition: 0.3s ease;
            margin-top: 20px;
        }
        .button:hover {
            background-color: #FF5722;
            box-shadow: 0px 6px 8px rgba(0, 0, 0, 0.15);
        }
    </style>
""", unsafe_allow_html=True)

# ---- App Layout ----
st.markdown("<div class='header'>Video Captioning App</div>", unsafe_allow_html=True)
st.markdown("<div class='description'>Upload your video, and we will generate a caption and overlay it onto the video.</div>", unsafe_allow_html=True)

# ---- File Upload ----
video_file = st.file_uploader("Choose a video file", type=["mp4"])

if video_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(video_file.read())
        video_path = tmp_file.name

    st.video(video_file)

    st.markdown("<div class='description'>Click on 'Process Video' to generate and embed captions.</div>", unsafe_allow_html=True)

    if st.button("Process Video", key="process_video"):
        with st.spinner("Processing video... please wait"):
            output_video_path = process_video_with_caption(video_path, processor, model)
            if output_video_path:
                st.session_state.processed_video_path = output_video_path
                st.success("Video processed successfully!")

    if 'processed_video_path' in st.session_state:
        output_video_path = st.session_state.processed_video_path
        with open(output_video_path, "rb") as f:
            st.download_button(
                label="Download Captioned Video",
                data=f,
                file_name="output_video_with_caption.mp4",
                mime="video/mp4"
            )
else:
    st.warning("Please upload an MP4 video to get started.")

