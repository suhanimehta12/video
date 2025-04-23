import streamlit as st
import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import tempfile
import os
import logging

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Video Captioning App", layout="wide")

# Load model and processor with caching
@st.cache_resource
def load_blip_model():
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="./blip_model")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir="./blip_model")
        return processor, model
    except Exception as e:
        st.error("Error loading model. Please ensure you're connected to the internet during the first run.")
        st.stop()

processor, model = load_blip_model()

def generate_caption(image):
    try:
        text = "a picture of"
        inputs = processor(image, text, return_tensors="pt")
        caption_ids = model.generate(**inputs)
        caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
        return caption
    except Exception as e:
        st.error(f"Caption generation failed: {e}")
        return "No caption"

def process_video_with_caption(video_path, processor, model):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Failed to open the video. Ensure it's a valid .mp4 file.")
            return None

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or fps is None:
            fps = 24  # Fallback default

        temp_dir = tempfile.gettempdir()
        output_video_path = os.path.join(temp_dir, "output_video_with_caption.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        # Read first frame for caption
        ret, frame = cap.read()
        if not ret:
            st.error("Could not read frame from video.")
            return None

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        caption = generate_caption(pil_image)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            try:
                cv2.putText(frame, caption, (50, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, lineType=cv2.LINE_AA)
            except Exception as e:
                st.warning(f"Could not render text on frame: {e}")
            out.write(frame)

        cap.release()
        out.release()
        return output_video_path

    except Exception as e:
        st.error(f"Unexpected error during video processing: {e}")
        return None

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
    </style>
""", unsafe_allow_html=True)

# ---- App Layout ----
st.markdown("<div class='header'>Video Captioning App</div>", unsafe_allow_html=True)
st.markdown("<div class='description'>Upload a short video file (.mp4). We will generate a caption from the first frame and overlay it on the entire video.</div>", unsafe_allow_html=True)

# ---- Upload Video ----
video_file = st.file_uploader("Upload your video file (max 100MB)", type=["mp4"])

if video_file is not None:
    # Check file size
    if video_file.size > 100_000_000:
        st.error("Video is too large. Please upload a file under 100MB.")
        st.stop()

    # Save uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    st.video(video_file)

    st.markdown("<div class='description'>Click below to generate captions and process your video.</div>", unsafe_allow_html=True)

    if st.button("Process Video"):
        with st.spinner("Processing video... this might take a moment."):
            output_video_path = process_video_with_caption(video_path, processor, model)
            if output_video_path:
                st.success("Video processed successfully!")
                with open(output_video_path, "rb") as f:
                    st.download_button(
                        label="Download Captioned Video",
                        data=f,
                        file_name="output_video_with_caption.mp4",
                        mime="video/mp4"
                    )
else:
    st.warning("Please upload a video to get started.")

