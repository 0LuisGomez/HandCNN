import cv2
import numpy as np


# Trackbar callback function does nothing but required for trackbar
def nothing(x):
    pass


# Function to preprocess the image
def preprocess_and_visualize(image):
    # Convert to YCrCb color space
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Use the inRange function to threshold the YCrCb image
    lower = np.array(
        [
            cv2.getTrackbarPos("Y min", "Trackbars"),
            cv2.getTrackbarPos("Cr min", "Trackbars"),
            cv2.getTrackbarPos("Cb min", "Trackbars"),
        ],
        dtype="uint8",
    )
    upper = np.array(
        [
            cv2.getTrackbarPos("Y max", "Trackbars"),
            cv2.getTrackbarPos("Cr max", "Trackbars"),
            cv2.getTrackbarPos("Cb max", "Trackbars"),
        ],
        dtype="uint8",
    )
    skinMask = cv2.inRange(ycrcb_img, lower, upper)

    # Apply the mask to get the segmented hand
    hand_segment = cv2.bitwise_and(image, image, mask=skinMask)
    return hand_segment


# Setup trackbars
cv2.namedWindow("Trackbars")
cv2.createTrackbar("Y min", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("Cr min", "Trackbars", 133, 255, nothing)
cv2.createTrackbar("Cb min", "Trackbars", 77, 255, nothing)
cv2.createTrackbar("Y max", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("Cr max", "Trackbars", 173, 255, nothing)
cv2.createTrackbar("Cb max", "Trackbars", 127, 255, nothing)

# Setup access to the webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the captured frame
        preprocessed_frame = preprocess_and_visualize(frame)

        # Display the original frame and the preprocessed frame side by side
        combined_display = np.hstack((frame, preprocessed_frame))
        cv2.imshow("Original and Preprocessed Frames", combined_display)

        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
