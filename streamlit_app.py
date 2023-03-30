import streamlit as st
from PIL import Image
import io
from app_utils.model_utils import load_yolov5_model
from app_utils.object_detection import run_object_detection

def app():
    # Set page title and display app title and description
    st.set_page_config(page_title='Guesser: The Thingamajig Revealer!')
    st.title("Guesser 🕵️‍♀️🪄: The Thingamajig Revealer! 🧐")
    st.write("A cure for 'What the heck is that thingamabob?!'")
    
    # Create a dropdown menu to select image input method
    image_input_method = st.selectbox(
        "Choose how to provide the image:",
        ["Select an option", "Take a picture", "Upload an image"]
    )

    # Process selected image input method
    image_to_process = None
    if image_input_method == "Take a picture":
        image_to_process = st.camera_input("Take a picture")
    elif image_input_method == "Upload an image":
        image_to_process = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Process the selected image
    if image_to_process is not None:
        # Display the selected image
#         image = Image.open(image_to_process)
        if st.button("Detect Objects"):
            # Load YOLOv5 model and run object detection
            model = load_yolov5_model()
            result_image = run_object_detection(model=model, image_bytes=image_to_process.getvalue())
            st.image(result_image, caption='Detected Objects', use_column_width=True)
            
            # Allow users to download the final image with labels
            img_buffer = io.BytesIO()
            result_image.save(img_buffer, format='PNG')
            st.download_button(label='Download image with labels', data=img_buffer, file_name='detected_objects.png', mime='image/png')
            

# Run the app when the script is executed directly
if __name__ == "__main__":
    app()
