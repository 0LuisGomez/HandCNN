import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Function to calculate the angle with respect to the horizontal axis
def calculate_horizontal_angle(p1, p2):
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi


# Function to calculate the angle with respect to the vertical axis
def calculate_vertical_angle(p1, p2):
    return np.arctan2(p2[0] - p1[0], p2[1] - p1[1]) * 180 / np.pi


# Start capturing from the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            frame_width, frame_height = frame.shape[1], frame.shape[0]

            # Get landmarks for WRIST and INDEX_FINGER_TIP
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            wrist_pos = (int(wrist.x * frame_width), int(wrist.y * frame_height))
            index_tip_pos = (
                int(index_tip.x * frame_width),
                int(index_tip.y * frame_height),
            )

            # Calculate horizontal and vertical angles
            horizontal_angle = calculate_horizontal_angle(wrist_pos, index_tip_pos)
            vertical_angle = calculate_vertical_angle(wrist_pos, index_tip_pos)

            # Draw a line indicating the pointing direction
            cv2.line(frame, wrist_pos, index_tip_pos, (0, 255, 0), 3)
            cv2.putText(
                frame,
                f"Horizontal Angle: {horizontal_angle:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Vertical Angle: {vertical_angle:.2f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand Pointing Direction", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
