import cv2
import numpy as np
from hand_detection import HandDetector


# Initialize HandDetector
hand_detector = HandDetector()

# Labels to display (initially empty)
labels_info = [""] * 3

# Start capturing video from the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Process the frame with the hand detector (this is the "layer")
    frame = hand_detector.process(frame)

    # Here you can modify labels_info based on the results of the hand detection
    # For example, you could display the number of hands detected
    # This is just a placeholder for where you would update the labels
    labels_info[0] = "Number of Hands: ..."
    labels_info[1] = "Hand 1 Position: ..."
    labels_info[2] = "Hand 2 Position: ..."

    # Display the resulting frame
    cv2.imshow("MediaPipe Hands", frame)

    # Update the information window
    text_window_height = 100 + 30 * len(labels_info)
    text_window = 255 * np.ones((text_window_height, 300, 3), dtype=np.uint8)
    for i, text in enumerate(labels_info):
        cv2.putText(
            text_window,
            text,
            (10, 30 + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            1,
        )
    cv2.imshow("Information", text_window)

    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
