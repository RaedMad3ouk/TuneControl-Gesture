import os
import cv2
import random

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 6

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam")
    exit()

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}. Press "Q" to capture, "N" for next class, "ESC" to exit.')

    counter = len(os.listdir(class_dir))  # Continue from last saved image
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break

        # Randomly flip the frame with 50% chance
        if random.random() > 0.5:
            frame = cv2.flip(frame, 1)  # Flip horizontally (1)

        cv2.putText(frame, f'Class {j}: Press "Q" to capture', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Capture image
            cv2.imwrite(os.path.join(class_dir, f'{counter}.jpg'), frame)
            print(f'Captured image {counter} for class {j}')
            counter += 1
        elif key == ord('n'):  # Move to next class
            break
        elif key == 27:  # ESC key to exit
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()
