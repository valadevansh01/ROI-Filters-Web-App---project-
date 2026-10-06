# ROI Filters Studio — Web App

A Streamlit web application for selecting rectangular regions of interest (ROIs) and applying a different image filter to each region.

## Features
- Upload PNG, JPG, JPEG, BMP, or WEBP images
- Draw multiple rectangular ROIs in the browser
- Assign Gaussian Blur, Median Filter, Sharpening, or Edge Detection per ROI
- Preview original and processed images
- Download the final image as PNG
- Handles images with transparency by converting them to RGB for processing

## Run locally
Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL (usually http://localhost:8501) to open in your browser.

## Deploy online
Recommended: Streamlit Community Cloud.
1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, and this README.
3. Open https://share.streamlit.io/ and connect your GitHub account.
4. Create an app, select the repository and `app.py`, then deploy.

This is a web app, not a Tkinter desktop program. A generic online Python compiler may not support Streamlit servers or browser canvas components; use Streamlit Community Cloud or another host that supports Streamlit.

## Notes
- Rectangles are mapped from the displayed canvas back to original-image pixel coordinates.
- ROIs are processed in drawing order. If they overlap, later ROIs overwrite earlier results in the overlap.
- Large images may take longer to process.
