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
        # Fix: ensure the string is properly closed
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
        # Fix: ensure the string is properly closed
        st.error("Error: Could not read frame.")
        return None
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    caption = generate_caption(pil_image)

    # Go back to the start and overlay the caption
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

# Custom CSS for styling
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

# Streamlit app layout with radio buttons for navigation
page = st.radio("Choose a page", ["Upload Video", "Download Processed Video"])

if page == "Upload Video":
    st.markdown("<div class='header'>Upload a Video</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="description">
        Upload your video, and we will generate captions for it.
    </div>
    """, unsafe_allow_html=True)

    video_file = st.file_uploader("Choose a video", type=["mp4"])
    if video_file is not None:
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())
            video_path = tmp_file.name

        st.video(video_file)

        st.markdown("""
        <div class="description">
            Click on "Process Video" to add captions to your video.
        </div>
        """, unsafe_allow_html=True)

        if st.button("Process Video", key="process_video"):
            with st.spinner("Processing video..."):
                output_video_path = process_video_with_caption(video_path, processor, model)
                st.session_state.processed_video_path = output_video_path
                st.success("Video processed successfully!")

elif page == "Download Processed Video":
    st.markdown("<div class='header'>Download Processed Video</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="description">
        After processing, click the button below to download your video with


