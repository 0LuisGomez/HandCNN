import cv2
import mediapipe as mp
import numpy as np

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
            middle_nuckle = points[9]  # Middle finger knuckle landmark

            # Draw line from wrist to middle finger tip
            cv2.line(frame, wrist_point, middle_fingertip, (0, 255, 0), 3)

            # Draw line from thumb tip to pinkie tip
            cv2.line(frame, thumb_tip, pinkie_tip, (255, 0, 0), 3)

            # Draw box structure
            cv2.line(frame, index_tip, thumb_tip, (229, 68, 235), 2)
            cv2.line(frame, thumb_tip, wrist_point, (229, 68, 235), 2)
            cv2.line(frame, wrist_point, pinkie_tip, (229, 68, 235), 2)
            cv2.line(frame, pinkie_tip, index_tip, (229, 68, 235), 2)

            # Additional lines
            cv2.line(frame, thumb_tip, middle_nuckle, (216, 227, 59), 2)
            cv2.line(frame, middle_fingertip, middle_nuckle, (216, 227, 59), 2)
            cv2.line(frame, wrist_point, middle_nuckle, (216, 227, 59), 2)
            cv2.line(frame, pinkie_tip, middle_nuckle, (216, 227, 59), 2)

            # Calculate the centroid of the pink square
            centroid_x = int(
                (index_tip[0] + thumb_tip[0] + wrist_point[0] + pinkie_tip[0]) / 4
            )
            centroid_y = int(
                (index_tip[1] + thumb_tip[1] + wrist_point[1] + pinkie_tip[1]) / 4
            )
            centroid_point = (centroid_x, centroid_y)

            # Draw the centroid on the frame
            cv2.circle(frame, centroid_point, 5, (0, 255, 255), -1)

            cv2.line(frame, centroid_point, middle_nuckle, (255, 255, 255), 2)

            # Draw specific landmarks
            for landmark_point in [
                wrist_point,
                middle_fingertip,
                thumb_tip,
                pinkie_tip,
                index_tip,
                middle_nuckle,
            ]:
                cv2.circle(frame, landmark_point, 5, (0, 0, 255), -1)

            # Display the frame with lines and landmarks
            cv2.imshow("Hand with Lines and Specific Landmarks", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
