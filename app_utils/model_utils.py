import torch
import streamlit as st

@st.cache_resource
def load_yolov5_model() -> object:
    """
    Load the pre-trained YOLOv5 model using the torch.hub utility.
    This function is cached to avoid re-loading the model each time.
    Returns:
        object: The loaded YOLOv5 model.
    """
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return model
