import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False,
    max_num_hands=2,
)


# Function to calculate the horizontal angle
def calculate_horizontal_angle(p1, p2):
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi


# Function to calculate the vertical angle
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
            # Calculate the hand's center
            lm_list = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            center_x = int(
                sum([lm[0] for lm in lm_list]) / len(lm_list) * frame.shape[1]
            )
            center_y = int(
                sum([lm[1] for lm in lm_list]) / len(lm_list) * frame.shape[0]
            )
            hand_center = (center_x, center_y)

            # Determine the hand's position in the frame
            frame_height, frame_width, _ = frame.shape
            if center_x < frame_width / 2 and center_y < frame_height / 2:
                position = "top left"
            elif center_x >= frame_width / 2 and center_y < frame_height / 2:
                position = "top right"
            elif center_x < frame_width / 2 and center_y >= frame_height / 2:
                position = "bottom left"
            else:
                position = "bottom right"

            # Calculate horizontal and vertical angles for the direction the hand is pointing
            # Using the WRIST and INDEX_FINGER_TIP landmarks to define the pointing direction
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            wrist_pos = (wrist.x * frame_width, wrist.y * frame_height)
            index_tip_pos = (index_tip.x * frame_width, index_tip.y * frame_height)

            horizontal_angle = calculate_horizontal_angle(wrist_pos, index_tip_pos)
            vertical_angle = calculate_vertical_angle(wrist_pos, index_tip_pos)

            # Identify extended fingers
            extended_fingers = []
            finger_tips = [
                mp_hands.HandLandmark.THUMB_TIP,
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP,
            ]
            for fingertip in finger_tips:
                tip = hand_landmarks.landmark[fingertip]
                tip_pos = (tip.x * frame_width, tip.y * frame_height)
                # Simple heuristic to identify if a finger is extended
                if tip_pos[1] < hand_center[1]:
                    extended_fingers.append(fingertip)

            # Determine palm or back of the hand
            thumb_cmc = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
            pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]
            thumb_cmc_pos = (thumb_cmc.x * frame_width, thumb_cmc.y * frame_height)
            pinky_mcp_pos = (pinky_mcp.x * frame_width, pinky_mcp.y * frame_height)
            hand_orientation = "palm" if thumb_cmc_pos[0] < pinky_mcp_pos[0] else "back"

            # Display the results
            cv2.putText(
                frame,
                f"Position: {position}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Horizontal Angle: {horizontal_angle:.2f}",
                (10, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Vertical Angle: {vertical_angle:.2f}",
                (10, 230),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Extended Fingers: {len(extended_fingers)}",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"Hand Orientation: {hand_orientation}",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
