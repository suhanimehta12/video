import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import streamlit as st
import tempfile
import os
from io import BytesIO

def main():
    st.title("Video Caption Generator")

    # Navigation for the two pages
    page = st.sidebar.radio("Choose a page", ["Upload the Video", "Output of the video"])

    if page == "Upload the Video":
        upload_the_video()
    elif page == "Output of the Video":
        Output_of_the_Video()

def upload_and_preview_video():
    st.subheader("Upload Your Video")
    

    st.write("""
  This application allows you to upload your video and based on that video it generates a caption for the video
    """)


    video_file = st.file_uploader("Upload a Video", type=["mp4"])

    if video_file is not None:
    
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())
            video_path = tmp_file.name

        st.video(video_file)


        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

  
        output_video_path = process_video_with_caption(video_path, processor, model)


        st.session_state.output_video_path = output_video_path


def download_processed_video():
    st.subheader("Download the Output Video")

    st.write("""
    After your video is processed, you can download the video with the generated captions by clciking the button below.
    """)

    # Check if the processed video is available
    if "output_video_path" in st.session_state:
        output_video_path = st.session_state.output_video_path
        

        with open(output_video_path, "rb") as f:
            st.download_button(
                label="Download the output Video",
                data=f,
                file_name="output_video_with_caption.mp4",
                mime="video/mp4"
            )
    else:
        st.warning("Please upload and process a video first to download it.")

def process_video_with_caption(video_path, processor, model):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return None


    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)


    output_video_path = os.path.join(tempfile.mkdtemp(), "output_video_with_caption.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    ret, frame = cap.read()
    if not ret:
        st.error("Error: Could not read frame.")
        return None


    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    inputs = processor(pil_image, return_tensors="pt")
    out_text = model.generate(**inputs)
    common_caption = processor.decode(out_text[0], skip_special_tokens=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        cv2.putText(frame, common_caption, (50, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    st.success("Video Processing Completed!")

    return output_video_path

if __name__ == "__main__":
    main()
()
