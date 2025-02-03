# Gesture-Based Spotify Controller

## Introduction

This project is my final year capstone for the Machine Learning specialization at Holberton School. The goal is to develop a gesture-based system that allows users to control Spotify using hand movements. By leveraging computer vision and deep learning, the model recognizes gestures and maps them to Spotify functions without requiring an API, relying instead on GUI automation.

## Architecture

The project consists of the following key components:

- **Gesture Recognition Model**: Utilizes OpenCV and a deep learning model to classify hand gestures.
- **Data Collection & Preprocessing**: Captures and processes images to train the model.
- **Machine Learning Model**: A CNN-based model trained to recognize different hand gestures.
- **Gesture Mapping System**: Maps recognized gestures to corresponding Spotify commands.
- **GUI Automation**: Employs PyAutoGUI to simulate keypresses and control Spotify.

## Model Patterns

The model is built using a Convolutional Neural Network (CNN):

- **Input**: Real-time hand images processed with OpenCV.
- **Classification**: The model categorizes gestures, each corresponding to a Spotify function.
- **Real-Time Predictions**: Utilizes a webcam feed for live gesture recognition.

## Data Collection & Preprocessing

- **Dataset Creation**: Collected images of hand gestures using OpenCV.
- **Gesture Representation**: Each gesture corresponds to a specific action (e.g., play, pause, next, previous, volume up/down).
- **Image Processing**: Resized and converted images to grayscale for consistency.
- **Data Augmentation**: Applied techniques to enhance model robustness.
- **Dataset Split**: Divided into training and validation sets.

## Training Process

- **Framework**: Trained the CNN model using TensorFlow/Keras.
- **Optimization**: Employed categorical cross-entropy loss function and Adam optimizer.
- **Training Duration**: Continued for multiple epochs until achieving high accuracy.

## Custom Data Collection

Users can collect their own hand gesture samples and assign them to specific Spotify functions:

1. **Run the Data Collection Script**: Execute `data_collection.py` to capture images of custom gestures.
2. **Labeling**: Assign labels to each gesture corresponding to the desired action.
3. **Retraining**: Use the collected dataset to retrain the CNN model.
4. **Update Mappings**: Modify the gesture-to-action mapping in `config.py` to reflect the new gestures.

## Gesture Mapping

Each recognized gesture is mapped to a corresponding Spotify function using PyAutoGUI to send keyboard shortcuts:

- ✋ **Open palm** → Play/Pause
- 👆 **One finger** → Next track
- ✌️ **Two fingers** → Previous track
- 🤘 **Rock sign** → Increase volume
- 👊 **Fist** → Decrease volume

## Dependencies

- Python 3.x
- OpenCV
- TensorFlow/Keras
- PyAutoGUI
- NumPy

## How to Run

1. **Install Dependencies**:

   ```bash
   pip install opencv-python tensorflow keras numpy pyautogui




To enhance the visual hierarchy of your README.md file and make titles more prominent, you can utilize Markdown's heading syntax. Here's how you can structure your README with appropriately sized headings:

markdown
Copy
Edit
# Gesture-Based Spotify Controller

## Introduction

This project is my final year capstone for the Machine Learning specialization at Holberton School. The goal is to develop a gesture-based system that allows users to control Spotify using hand movements. By leveraging computer vision and deep learning, the model recognizes gestures and maps them to Spotify functions without requiring an API, relying instead on GUI automation.

## Architecture

The project consists of the following key components:

- **Gesture Recognition Model**: Utilizes OpenCV and a deep learning model to classify hand gestures.
- **Data Collection & Preprocessing**: Captures and processes images to train the model.
- **Machine Learning Model**: A CNN-based model trained to recognize different hand gestures.
- **Gesture Mapping System**: Maps recognized gestures to corresponding Spotify commands.
- **GUI Automation**: Employs PyAutoGUI to simulate keypresses and control Spotify.

## Model Patterns

The model is built using a Convolutional Neural Network (CNN):

- **Input**: Real-time hand images processed with OpenCV.
- **Classification**: The model categorizes gestures, each corresponding to a Spotify function.
- **Real-Time Predictions**: Utilizes a webcam feed for live gesture recognition.

## Data Collection & Preprocessing

- **Dataset Creation**: Collected images of hand gestures using OpenCV.
- **Gesture Representation**: Each gesture corresponds to a specific action (e.g., play, pause, next, previous, volume up/down).
- **Image Processing**: Resized and converted images to grayscale for consistency.
- **Data Augmentation**: Applied techniques to enhance model robustness.
- **Dataset Split**: Divided into training and validation sets.

## Training Process

- **Framework**: Trained the CNN model using TensorFlow/Keras.
- **Optimization**: Employed categorical cross-entropy loss function and Adam optimizer.
- **Training Duration**: Continued for multiple epochs until achieving high accuracy.

## Custom Data Collection

Users can collect their own hand gesture samples and assign them to specific Spotify functions:

1. **Run the Data Collection Script**: Execute `data_collection.py` to capture images of custom gestures.
2. **Labeling**: Assign labels to each gesture corresponding to the desired action.
3. **Retraining**: Use the collected dataset to retrain the CNN model.
4. **Update Mappings**: Modify the gesture-to-action mapping in `config.py` to reflect the new gestures.

## Gesture Mapping

Each recognized gesture is mapped to a corresponding Spotify function using PyAutoGUI to send keyboard shortcuts:


✋ Open palm → Play

👆 One finger → Next track

👎 Thumb down sign → Decrease volume

👍 Thumb up sign → Increase volume

👊 Fist → Stop

## Dependencies

- Python 3.x
- OpenCV
- TensorFlow/Keras
- PyAutoGUI
- NumPy

## How to Run

1. **Install Dependencies**:

   ```bash
   pip install opencv-python tensorflow keras numpy pyautogui
Execute the Gesture Recognition Script:

bash
Copy
Edit
python main.py
Control Spotify: Use predefined hand gestures to interact with Spotify.

Future Improvements
Enhance model accuracy with a larger dataset.
Support for custom gesture mappings.
Integration with additional media applications beyond Spotify.
Conclusion
This project demonstrates how computer vision and machine learning can be combined with GUI automation to create an intuitive, touchless way to control media playback. It serves as an excellent example of how AI can enhance user experience in everyday applications.
