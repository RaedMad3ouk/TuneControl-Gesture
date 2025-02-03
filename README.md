Introduction

This project is my final year capstone for my Machine Learning specialization at Holberton School. The goal is to develop a gesture-based system that allows users to control Spotify using hand movements. By leveraging computer vision and deep learning, the model recognizes gestures and maps them to Spotify functions without requiring an API, relying instead on GUI automation.

Architecture

The project consists of the following key components:

Gesture Recognition Model – Uses OpenCV and a deep learning model to classify hand gestures.

Data Collection & Preprocessing – Captures and processes images to train the model.

Machine Learning Model – A CNN-based model trained to recognize different hand gestures.

Gesture Mapping System – Maps recognized gestures to corresponding Spotify commands.

GUI Automation – Uses PyAutoGUI to simulate keypresses and control Spotify.

Model Patterns

The model is built using a Convolutional Neural Network (CNN).

The input consists of real-time hand images processed with OpenCV.

The model classifies gestures into different categories, each corresponding to a Spotify function.

Predictions are made in real-time using a webcam feed.

Data Collection & Preprocessing

A dataset of hand gestures was collected using OpenCV.

Each gesture represents a different action (play, pause, next, previous, volume up/down, etc.).

Images were resized and converted to grayscale for consistency.

Data augmentation techniques were applied to improve model robustness.

The dataset was split into training and validation sets.

Training Process

The CNN model was trained using TensorFlow/Keras.

Categorical cross-entropy loss function and Adam optimizer were used.

The model was trained for multiple epochs until achieving high accuracy.

Custom Data Collection

Users can collect their own hand gesture samples and assign them to specific Spotify functions:

Run the data_collection.py script to capture images of custom gestures.

Label each gesture according to the action it should trigger.

Use the collected dataset to retrain the CNN model.

Update the gesture-to-action mapping in config.py to reflect the new gestures.

Gesture Mapping

Each recognized gesture is mapped to a corresponding Spotify function.

PyAutoGUI is used to send keyboard shortcuts that control Spotify.

Example mappings:

✋ Open palm → Play/Pause

👆 One finger → Next track

✌️ Two fingers → Previous track

🤘 Rock sign → Increase volume

👊 Fist → Decrease volume

Dependencies

Python 3.x

OpenCV

TensorFlow/Keras

PyAutoGUI

NumPy

How to Run

Install dependencies:

pip install opencv-python tensorflow keras numpy pyautogui

Run the gesture recognition script:

python main.py

Use predefined hand gestures to control Spotify.

Future Improvements

Improve model accuracy with a larger dataset.

Support for custom gesture mappings.

Integration with more media applications beyond Spotify.

Conclusion

This project demonstrates how computer vision and machine learning can be combined with GUI automation to create an intuitive, touchless way to control media playback. It serves as an excellent example of how AI can enhance user experience in everyday applications.
