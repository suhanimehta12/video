import cv2
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import streamlit as st
import tempfile
import os
from io import BytesIO

# Streamlit App
def main():
    st.title("Video Captioning with BLIP")

    # Upload video file through Streamlit
    video_file = st.file_uploader("Upload a Video", type=["mp4"])

    if video_file is not None:
        # Write the uploaded video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())
            video_path = tmp_file.name

        # Display video
        st.video(video_file)

        # Load the BLIP model and processor for image captioning
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # Process the video and overlay caption
        output_video_path = process_video_with_caption(video_path, processor, model)

        # Display processed video
        st.video(output_video_path)

        # Provide download button for the processed video
        with open(output_video_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="output_video_with_common_caption.mp4",
                mime="video/mp4"
            )

def process_video_with_caption(video_path, processor, model):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return None

    # Get the video frame width, height, and FPS (frames per second)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a temporary output video file
    output_video_path = os.path.join(tempfile.mkdtemp(), "output_video_with_common_caption.mp4")

    # Initialize the video writer to write the new video with captions
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Process the first frame (or any frame you want) to generate one caption
    ret, frame = cap.read()
    if not ret:
        st.error("Error: Could not read frame.")
        return None

    # Convert the frame to a PIL image for captioning
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Generate the caption for this frame (the same caption will be used for all frames)
    inputs = processor(pil_image, return_tensors="pt")
    out_text = model.generate(**inputs)
    common_caption = processor.decode(out_text[0], skip_special_tokens=True)

    # Go back to the start of the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Read frames and overlay the common caption
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Overlay the common caption on the frame
        cv2.putText(frame, common_caption, (50, frame_height - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Write the frame with the common caption to the output video
        out.write(frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

    st.success("Video Processing Completed!")

    return output_video_path

if __name__ == "__main__":
    main()
