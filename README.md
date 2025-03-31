# BLIP-Based Video Captioning App

This Streamlit application generates captions for uploaded videos using the **BLIP model** from **Salesforce**.  
It processes a video, extracts frames, generates a caption for the first frame, and overlays it on all frames.  
The final processed video is available for preview and download.

## Features:
- Upload a video (MP4 format)
- Generate captions using the BLIP model
- Overlay captions on video frames
- Preview and download the processed video

## Installation:
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/video-captioning-blip.git
cd video-captioning-blip
pip install -r requirements.txt
