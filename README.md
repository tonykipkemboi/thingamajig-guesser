# Guesser: The Thingamajig Revealer! 🕵️‍♀️🪄

## Overview

Welcome to Guesser: The Thingamajig Revealer! 🧐
Are you tired of not knowing what that thingamabob is? Say no more!
Guesser is here to solve all your object identification woes.

This app uses the power of YOLOv5 object detection to identify various objects in images. Take a picture or upload an image, and let Guesser do the rest. The app will identify objects in the image and display them with bounding boxes and labels.

## How to Use

1. Launch the app by running `streamlit run streamlit_app.py` in the terminal.
2. Choose how to provide the image by selecting an option from the dropdown menu:
   - Take a picture using your camera
   - Upload an image from your device
3. After selecting the image, click the "Detect Objects" button.
4. View the detected objects and bounding boxes on the image.
5. Optionally, download the image with the labels.

## Installation

To install the required dependencies, run the following command in your terminal:

```bash
pip install -r requirements
```


## Technologies Used

- Streamlit
- PyTorch (YOLOv5)
- Pillow
- NumPy
- OpenCV

## Contributing

Contributions are welcome! If you have any ideas, bug reports, or enhancements, feel free to open an issue or submit a pull request on the GitHub repository.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

