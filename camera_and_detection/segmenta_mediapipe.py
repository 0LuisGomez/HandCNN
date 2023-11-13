import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils
teal_color = (255, 228, 225)  # RGB color for teal

cap = cv2.VideoCapture(0)


def extend_point(p1, p2, length):
    direction = np.array([p2[0] - p1[0], p2[1] - p1[1]])
    norm_direction = direction / np.linalg.norm(direction)
    extended_point = p2 + norm_direction * length
    return extended_point.astype(int)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw standard hand landmarks
            for landmark in hand_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, teal_color, -1)

            # Extract landmarks for segmentation and extension
            points = [
                (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                for landmark in hand_landmarks.landmark
            ]
            wrist_point = points[0]
            center_point = np.mean(points, axis=0).astype(int)
            extended_point = extend_point(center_point, wrist_point, 50)
            points.append(tuple(extended_point))

            # Create a polygon for segmentation
            polygon = np.array([points], dtype=np.int32)
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, polygon, 255)
            segmented_hand = cv2.bitwise_and(frame, frame, mask=mask)

            cv2.imshow("Segmented Hand", segmented_hand)

    cv2.imshow("MediaPipe Hands", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
