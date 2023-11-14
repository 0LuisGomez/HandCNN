import cv2
import mediapipe as mp
import numpy as np
import math


# Function to calculate the angle between two vectors
def angle_between_vectors_degrees(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    angle_rad = np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    return np.degrees(angle_rad)


previous_normal_vector = np.array(
    [0, 0, 1]
)  # Initial normal vector assuming palm facing away
confidence_threshold_degrees = (
    30.0  # Angle threshold in degrees for confidence checking
)


# Define a function to calculate the cross product to find the normal vector
def calculate_normal_vector(p1, p2, p3):
    vector1 = np.array([p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]])
    vector2 = np.array([p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]])
    normal = np.cross(vector1, vector2)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return normal
    return normal / norm


def get_pointing_direction(green_vector):
    pointing_description = ""
    pointing_tolerance = 0.2  # Adjust as needed for sensitivity

    # Normalized vectors for the eight primary and secondary directions
    direction_vectors = {
        "Down": np.array([0, 1, 0]),
        "Up": np.array([0, -1, 0]),
        "Left": np.array([-1, 0, 0]),
        "Right": np.array([1, 0, 0]),
        "Down-Right": np.array([1, 1, 0]) / np.sqrt(2),
        "Down-Left": np.array([-1, 1, 0]) / np.sqrt(2),
        "Up-Right": np.array([1, -1, 0]) / np.sqrt(2),
        "Up-Left": np.array([-1, -1, 0]) / np.sqrt(2),
        "Towards": np.array([0, 0, -1]),
        "Away": np.array([0, 0, 1]),
    }

    max_dot = -1.0
    for direction, vector in direction_vectors.items():
        dot_product = np.dot(green_vector, vector)
        if dot_product > max_dot:
            max_dot = dot_product
            if dot_product > 1 - pointing_tolerance:
                pointing_description = "Pointing " + direction

    if not pointing_description:
        pointing_description = "Pointing Direction Unclear"

    return pointing_description


# Function to provide a description based on the normal vector
def get_hand_description(normal_vector):
    description = ""
    # Increased tolerance for up and down detection
    tolerance_up_down = 0.5  # Adjust this value as needed
    tolerance_left_right_forward_backward = 0.2  # Keep this more sensitive

    # Predefined direction vectors with corrected up and down vectors
    up_vector = np.array([0, -1, 0])  # Now pointing down in the image
    down_vector = np.array([0, 1, 0])  # Now pointing up in the image
    left_vector = np.array([-1, 0, 0])
    right_vector = np.array([1, 0, 0])
    towards_vector = np.array([0, 0, -1])
    away_vector = np.array([0, 0, 1])

    # Compare normal vector with predefined direction vectors using dot product
    # Check for up/down orientation with increased tolerance
    if np.dot(normal_vector, up_vector) > 1 - tolerance_up_down:
        description += "Palm Facing Up "
    elif np.dot(normal_vector, down_vector) > 1 - tolerance_up_down:
        description += "Palm Facing Down "
    # Check for other orientations with regular tolerance
    elif np.dot(normal_vector, left_vector) > 1 - tolerance_left_right_forward_backward:
        description += "Palm Facing Left "
    elif (
        np.dot(normal_vector, right_vector) > 1 - tolerance_left_right_forward_backward
    ):
        description += "Palm Facing Right "
    elif (
        np.dot(normal_vector, towards_vector)
        > 1 - tolerance_left_right_forward_backward
    ):
        description += "Palm Facing Towards "
    elif np.dot(normal_vector, away_vector) > 1 - tolerance_left_right_forward_backward:
        description += "Palm Facing Away "
    else:
        description += "Orientation Not Clear "

    return description


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Process the frame with MediaPipe Hands
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks with x, y, and z coordinates
            points = [
                (
                    int(landmark.x * frame.shape[1]),  # X coordinate
                    int(landmark.y * frame.shape[0]),  # Y coordinate
                    landmark.z
                    * frame.shape[1],  # Z coordinate scaled similarly to X and Y
                )
                for landmark in hand_landmarks.landmark
            ]

            # Define points for normal vector calculation (base of index, base of pinkie, wrist)
            base_index = points[5]
            base_pinky = points[17]
            wrist = points[0]
            middle_fingertip = points[12]
            middle_knuckle = points[9]

            # Calculate the centroid of the triangle
            centroid = (
                int((base_index[0] + base_pinky[0] + wrist[0]) / 3),
                int((base_index[1] + base_pinky[1] + wrist[1]) / 3),
                (base_index[2] + base_pinky[2] + wrist[2]) / 3,
            )

            # Calculate the normal vector using the centroid
            normal_vector = calculate_normal_vector(base_index, base_pinky, centroid)

            green_vector = np.subtract(middle_knuckle, wrist)
            green_vector_normalized = green_vector / np.linalg.norm(green_vector)

            pointing_direction = get_pointing_direction(green_vector_normalized)

            # Visualize the centroid
            cv2.circle(frame, (centroid[0], centroid[1]), 5, (0, 255, 255), -1)

            # Visualize the normal vector from the centroid
            normal_point = (
                int(centroid[0] + normal_vector[0] * 50),
                int(centroid[1] + normal_vector[1] * 50),
            )
            cv2.line(frame, (centroid[0], centroid[1]), normal_point, (0, 0, 255), 2)

            # Visualize the green line
            cv2.line(
                frame,
                (wrist[0], wrist[1]),
                (middle_knuckle[0], middle_knuckle[1]),
                (0, 255, 0),
                2,
            )

            # Get hand description based on the normal vector
            hand_description = get_hand_description(normal_vector)

            hand_description += " and is " + pointing_direction

            # Display the hand description
            cv2.putText(
                frame,
                hand_description,
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Display the frame
            cv2.imshow("Hand Orientation", frame)
            print(hand_description)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
