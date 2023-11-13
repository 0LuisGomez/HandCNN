import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# Function to determine if a finger is extended based on the direction of flexing
def is_finger_extended(finger_tip, finger_pip, finger_mcp, hand_orientation):
    if hand_orientation == "right":
        # For a hand pointing to the right, check if the tip is to the right of the PIP joint
        return finger_tip.x > finger_pip.x
    elif hand_orientation == "left":
        # For a hand pointing to the left, check if the tip is to the left of the PIP joint
        return finger_tip.x < finger_pip.x
    elif hand_orientation == "up":
        # For a hand pointing up, check if the tip is above the PIP joint
        return finger_tip.y < finger_pip.y
    elif hand_orientation == "down":
        # For a hand pointing down, check if the tip is below the PIP joint
        return finger_tip.y > finger_pip.y


# Function to determine if the thumb is extended
def is_thumb_extended(thumb_tip, thumb_ip, thumb_cmc):
    # Check if the thumb tip is further out from the base joint (CMC) compared to the IP joint
    return (thumb_tip.x > thumb_ip.x and thumb_tip.y < thumb_ip.y) or (
        thumb_tip.x < thumb_ip.x and thumb_tip.y > thumb_ip.y
    )


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
            # Assume hand orientation is 'right' for simplicity; you may determine this dynamically
            hand_orientation = "right"
            extended_fingers = 0
            finger_names = ["index", "middle", "ring", "pinky"]
            finger_landmarks = [
                [
                    mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.INDEX_FINGER_PIP,
                    mp_hands.HandLandmark.INDEX_FINGER_MCP,
                ],
                [
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                ],
                [
                    mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_PIP,
                    mp_hands.HandLandmark.RING_FINGER_MCP,
                ],
                [
                    mp_hands.HandLandmark.PINKY_TIP,
                    mp_hands.HandLandmark.PINKY_PIP,
                    mp_hands.HandLandmark.PINKY_MCP,
                ],
            ]

            # Check each finger
            for i, finger in enumerate(finger_names):
                tip = hand_landmarks.landmark[finger_landmarks[i][0]]
                pip = hand_landmarks.landmark[finger_landmarks[i][1]]
                mcp = hand_landmarks.landmark[finger_landmarks[i][2]]

                if is_finger_extended(tip, pip, mcp, hand_orientation):
                    extended_fingers += 1

            # Special case for thumb
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
            thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]

            if is_thumb_extended(thumb_tip, thumb_ip, thumb_cmc):
                extended_fingers += 1

            # Display the number of extended fingers on the frame
            cv2.putText(
                frame,
                f"Extended Fingers: {extended_fingers}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Draw hand landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Finger Extension Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
