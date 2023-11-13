import cv2
import mediapipe as mp
import numpy as np
import math


def calculate_angle(p1, p2):
    # Calculate the angle relative to the horizontal
    angle_radians = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    angle_degrees = math.degrees(angle_radians)
    return angle_degrees


# Function to provide a description based on angles
def get_hand_description(angle_green, angle_knuckle_centroid):
    description = ""
    tolerance = 10  # degrees of tolerance for angle detection

    # Determine the vertical orientation for angle_green
    if -90 - tolerance < angle_green < -90 + tolerance:
        description += "Raised "
    elif 90 - tolerance < angle_green < 90 + tolerance:
        description += "Lowered "
    elif -tolerance < angle_green < tolerance:
        description += "Pointing Right "
    elif 180 - tolerance < angle_green or angle_green < -180 + tolerance:
        description += "Pointing Left "

    # Determine the palm direction for angle_knuckle_centroid
    if -tolerance < angle_knuckle_centroid < tolerance:
        description += "with Palm Facing Left"
    elif 90 - tolerance < angle_knuckle_centroid < 90 + tolerance:
        description += "with Palm Facing Up"
    elif -90 - tolerance < angle_knuckle_centroid < -90 + tolerance:
        description += "with Palm Facing Down"
    elif (
        abs(angle_knuckle_centroid) > 180 - tolerance
        or abs(angle_knuckle_centroid) < -180 + tolerance
    ):
        description += "with Palm Facing Right"

    return description if description else "Uncertain Orientation"


mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmarks for drawing lines and points
            points = [
                (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                for landmark in hand_landmarks.landmark
            ]
            wrist_point = points[0]  # Wrist landmark
            middle_fingertip = points[12]  # Middle finger tip landmark
            thumb_tip = points[4]  # Thumb tip landmark
            pinkie_tip = points[20]  # Pinkie tip landmark
            index_tip = points[8]  # Index finger tip landmark
            middle_knuckle = points[9]  # Middle finger knuckle landmark
            thumb_knuckle = points[1]  # Middle finger knuckle landmark
            index_knuckle = points[5]  # Middle finger knuckle landmark
            ring_knuckle = points[13]  # Middle finger knuckle landmark
            pinkie_knuckle = points[17]  # Middle finger knuckle landmark
            ring_tip = points[16]
            pinkie_tip = points[20]

            # Draw line from wrist to middle finger tip
            cv2.line(frame, wrist_point, middle_knuckle, (0, 255, 0), 3)

            # Draw line from thumb tip to pinkie tip
            # cv2.line(frame, thumb_tip, pinkie_tip, (255, 0, 0), 3)

            # Draw box structure
            # cv2.line(frame, index_tip, thumb_tip, (229, 68, 235), 2)
            # cv2.line(frame, thumb_tip, wrist_point, (229, 68, 235), 2)
            # cv2.line(frame, wrist_point, pinkie_tip, (229, 68, 235), 2)
            # cv2.line(frame, pinkie_tip, index_tip, (229, 68, 235), 2)

            # cv2.line(
            #     frame, thumb_knuckle, thumb_tip, (255, 165, 0), 2
            # )  # Orange line for thumb
            # cv2.line(
            #     frame, index_knuckle, index_tip, (255, 0, 255), 2
            # )  # Purple line for index finger
            # cv2.line(
            #     frame, middle_knuckle, middle_fingertip, (0, 255, 0), 2
            # )  # Green line for middle finger
            # cv2.line(
            #     frame, ring_knuckle, ring_tip, (0, 0, 255), 2
            # )  # Blue line for ring finger
            # cv2.line(
            #     frame, pinkie_knuckle, pinkie_tip, (255, 0, 0), 2
            # )  # Red line for pinkie

            # Additional lines
            # cv2.line(frame, thumb_tip, middle_knuckle, (216, 227, 59), 2)
            # cv2.line(frame, middle_fingertip, middle_knuckle, (216, 227, 59), 2)
            # cv2.line(frame, wrist_point, middle_knuckle, (216, 227, 59), 2)
            # cv2.line(frame, pinkie_tip, middle_knuckle, (216, 227, 59), 2)

            # Calculate the centroid of the pink square
            centroid_x = int(
                (
                    index_tip[0]
                    + thumb_tip[0]
                    + wrist_point[0]
                    + pinkie_tip[0]
                    + ring_tip[0]
                )
                / 5
            )
            centroid_y = int(
                (
                    index_tip[1]
                    + thumb_tip[1]
                    + wrist_point[1]
                    + pinkie_tip[1]
                    + ring_tip[1]
                )
                / 5
            )
            centroid_point = (centroid_x, centroid_y)

            # Draw the centroid on the frame
            cv2.circle(frame, centroid_point, 5, (0, 255, 255), -1)

            # Line from centroid to middle knuckle (white line)
            cv2.line(frame, centroid_point, middle_knuckle, (255, 255, 255), 2)

            # Draw specific landmarks
            for landmark_point in [
                wrist_point,
                middle_knuckle,
                thumb_tip,
                pinkie_tip,
                index_tip,
            ]:
                cv2.circle(frame, landmark_point, 5, (0, 0, 255), -1)

            # Calculate angles
            angle_green = calculate_angle(wrist_point, middle_knuckle)
            angle_knuckle_centroid = calculate_angle(centroid_point, middle_knuckle)

            # Get hand description
            hand_description = get_hand_description(angle_green, angle_knuckle_centroid)

            # Display the frame with lines and landmarks
            cv2.imshow("Hand with Lines and Specific Landmarks", frame)
            print(hand_description)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
