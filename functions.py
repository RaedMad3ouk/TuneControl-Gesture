import pickle
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp

# Load the trained model
model = keras.models.load_model('model.keras')

# Load the label binarizer
lb = pickle.load(open('label_binarizer.pkl', 'rb'))

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Updated labels_dict for the new gestures
labels_dict = {0: 'stop', 1: 'start', 2: 'increase', 3: 'decrease', 4: 'swipe', 5: 'swipe back'}

cap = cv2.VideoCapture(0)

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        continue

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        if data_aux:
            # Padding the input data to match the expected input shape (84 features)
            while len(data_aux) < 84:  # Ensure the input is 84 features
                data_aux.append(0)

            # Prepare the input for prediction
            data_input = np.asarray(data_aux).reshape(1, -1)
            prediction_probabilities = model.predict(data_input)

            # Get the predicted class and the probability
            predicted_class = np.argmax(prediction_probabilities, axis=1)[0]
            confidence = prediction_probabilities[0][predicted_class] * 100
            predicted_character = lb.classes_[predicted_class]

            # Display the gesture prediction and confidence
            cv2.putText(frame, f'{predicted_character} ({confidence:.2f}%)', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        else:
            # If no gesture is recognized, print "Unknown Gesture"
            cv2.putText(frame, 'Unknown Gesture', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    else:
        # If no hand is detected, print "No Hand"
        cv2.putText(frame, 'No Hand Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Sign Language Gesture Recognition', frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
