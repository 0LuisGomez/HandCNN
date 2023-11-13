import cv2
import mediapipe as mp

# Initialize MediaPipe Hands module.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Initialize MediaPipe drawing module.
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam.
cap = cv2.VideoCapture(0)


def get_hand_metadata(hand_landmarks, image_width, image_height):
    """
    Process hand landmarks to extract metadata such as hand orientation,
    position relative to the center of the frame, and scale.
    """
    # Example of processing landmarks to extract metadata.
    # For instance, you might calculate the hand's center point.
    landmarks = hand_landmarks.landmark
    palm_center = landmarks[0]  # Assuming the wrist landmark as palm center

    # Calculate the relative position of the palm center.
    relative_x = (
        palm_center.x - 0.5
    )  # 0.5 is the center of the frame in the relative coordinates
    relative_y = palm_center.y - 0.5  # Same for Y

    # Calculate the scale using the distance between two specific landmarks, e.g., wrist to middle fingertip.
    scale = (
        (
            landmarks[mp_hands.HandLandmark.WRIST].x
            - landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
        )
        ** 2
        + (
            landmarks[mp_hands.HandLandmark.WRIST].y
            - landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
        )
        ** 2
    ) ** 0.5

    # This is a simple example. You might need to define "orientation" and refine "scale" based on your requirements.

    return relative_x, relative_y, scale


try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Process the image and detect hands.
        results = hands.process(image)

        # Convert back to BGR for OpenCV.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Get additional hand metadata.
                metadata = get_hand_metadata(hand_landmarks, *image.shape[1::-1])
                print(
                    metadata
                )  # For debugging, you can print out or store the metadata.

        # Show the image.
        cv2.imshow("MediaPipe Hands", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
except Exception as e:
    print(e)
finally:
    cap.release()
    cv2.destroyAllWindows()
