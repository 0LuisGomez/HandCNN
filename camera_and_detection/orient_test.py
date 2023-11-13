import cv2
import numpy as np


# Function to preprocess the image and segment the hand
def preprocess_and_visualize(image, visualize=False):
    # Convert to YCrCb color space
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Threshold the image to get the skin mask
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skinMask = cv2.inRange(ycrcb_img, lower_skin, upper_skin)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_OPEN, kernel)
    skinMask = cv2.morphologyEx(skinMask, cv2.MORPH_CLOSE, kernel)

    # Apply the mask to get the segmented hand
    hand_segment = cv2.bitwise_and(image, image, mask=skinMask)

    if visualize:
        # Display the original frame and the skin mask side by side for comparison
        combined_display = np.hstack(
            (image, cv2.cvtColor(skinMask, cv2.COLOR_GRAY2BGR))
        )
        cv2.imshow("Original and Skin Mask", combined_display)

    return hand_segment


# Function to find the hand orientation and draw the bounding box
def find_orientation_and_draw_box(image, hand_segment):
    # Convert to grayscale
    gray = cv2.cvtColor(hand_segment, cv2.COLOR_BGR2GRAY)

    # Threshold to get the contours
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Assume the largest contour is the hand
        max_contour = max(contours, key=cv2.contourArea)

        # Get the min area rectangle
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # Draw the bounding box
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        # Calculate the angle
        angle = rect[2]
        if angle < -45:
            angle += 90

        # Display the angle
        cv2.putText(
            image,
            f"Angle: {angle:.2f} degrees",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return image, angle


# Setup access to the webcam
cap = cv2.VideoCapture(0)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Preprocess the captured frame
        hand_segment = preprocess_and_visualize(frame, visualize=True)

        # Find the orientation and draw the bounding box
        oriented_frame, hand_angle = find_orientation_and_draw_box(
            frame.copy(), hand_segment
        )

        # Display the original frame with the orientation bounding box
        cv2.imshow("Hand Orientation", oriented_frame)

        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
