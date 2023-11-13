import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    static_image_mode=False,
    max_num_hands=1,
)


# Function to calculate the horizontal angle
def calculate_horizontal_angle(p1, p2):
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi


# Function to calculate the vertical angle
def calculate_vertical_angle(p1, p2):
    return np.arctan2(p2[0] - p1[0], p2[1] - p1[1]) * 180 / np.pi


# Function to draw a line between two points
def draw_line(frame, p1, p2, color=(0, 255, 0), thickness=2):
    cv2.line(frame, p1, p2, color, thickness)


# Function to calculate the distance from a point to a line defined by two points
def point_to_line_distance(point, line_point1, line_point2):
    # Line equation coefficients A*x + B*y + C = 0
    A = line_point2[1] - line_point1[1]
    B = line_point1[0] - line_point2[0]
    C = line_point2[0] * line_point1[1] - line_point1[0] * line_point2[1]
    # Perpendicular distance from point to line
    distance = np.abs(A * point[0] + B * point[1] + C) / np.sqrt(A**2 + B**2)
    return distance


# Function to determine if a finger is extended
def is_finger_extended(tip, pip, mcp):
    # Convert to pixel coordinates
    tip_pos = (int(tip.x * frame_width), int(tip.y * frame_height))
    pip_pos = (int(pip.x * frame_width), int(pip.y * frame_height))
    mcp_pos = (int(mcp.x * frame_width), int(mcp.y * frame_height))
    # Draw the line for visual debugging
    draw_line(frame, pip_pos, mcp_pos, color=(0, 0, 255))
    # Calculate the distance of the fingertip to the line between PIP and MCP
    distance = point_to_line_distance(tip_pos, pip_pos, mcp_pos)
    # Threshold for considering a finger extended
    threshold = 10  # This threshold may need to be adjusted
    return distance > threshold


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

            extended_fingers = 0
            frame_width, frame_height = frame.shape[1], frame.shape[0]

            # Fingers
            finger_tips = [
                mp_hands.HandLandmark.THUMB_TIP,
                mp_hands.HandLandmark.INDEX_FINGER_TIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                mp_hands.HandLandmark.RING_FINGER_TIP,
                mp_hands.HandLandmark.PINKY_TIP,
            ]
            finger_pips = [
                mp_hands.HandLandmark.THUMB_IP,  # Use IP joint for thumb
                mp_hands.HandLandmark.INDEX_FINGER_PIP,
                mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                mp_hands.HandLandmark.RING_FINGER_PIP,
                mp_hands.HandLandmark.PINKY_PIP,
            ]
            finger_mcps = [
                mp_hands.HandLandmark.THUMB_CMC,  # Use CMC joint for thumb
                mp_hands.HandLandmark.INDEX_FINGER_MCP,
                mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
                mp_hands.HandLandmark.RING_FINGER_MCP,
                mp_hands.HandLandmark.PINKY_MCP,
            ]

            # Check each finger
            for fingertip in finger_tips:
                tip = hand_landmarks.landmark[fingertip]
                # Determine the corresponding PIP and MCP landmarks
                if fingertip == mp_hands.HandLandmark.THUMB_TIP:
                    pip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
                    mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC]
                else:
                    pip = hand_landmarks.landmark[fingertip - 2]
                    mcp = hand_landmarks.landmark[fingertip - 3]

                if is_finger_extended(tip, pip, mcp):
                    extended_fingers += 1

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
                f"Extended Fingers: {extended_fingers}",
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
