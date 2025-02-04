# Gesture-Based music Controller

## Introduction

This project is my final year capstone for the Machine Learning specialization at Holberton School. The goal is to develop a gesture-based system that allows users to control Spotify or any music player using hand movements. By leveraging computer vision and deep learning, the model recognizes gestures and maps them to Spotify functions without requiring an API, relying instead on GUI automation.

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
## Future Improvements
Enhance model accuracy with a larger dataset.
Support for custom gesture mappings.
Integration with additional media applications beyond Spotify.



## Accuracy Evaluation

To assess the model's performance, we conducted tests using a separate validation dataset. The results are as follows:
![image](https://github.com/user-attachments/assets/c3a3f511-382e-48e8-9893-38b3d7ecd1be)



## Data Collection Techniques

To enhance the robustness of the model, we employed several data augmentation techniques:

- **Flipping**: Horizontal flipping of images to simulate different orientations.
- **Blurring**: Applying Gaussian blur to images to mimic varying focus conditions.
- **Noise Addition**: Introducing random noise to images to simulate real-world imperfections.

These techniques enriched the dataset, allowing the model to generalize better to diverse real-world scenarios.

![Uploading image.png…]()






## Conclusion
This project demonstrates how computer vision and machine learning can be combined with GUI automation to create an intuitive, touchless way to control media playback. It serves as an excellent example of how AI can enhance user experience in everyday applications.
