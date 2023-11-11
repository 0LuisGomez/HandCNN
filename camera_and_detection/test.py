import cv2
import numpy as np
import os
import time  # Import the time module


# Function to preprocess the image
def preprocess_and_visualize(image, visualize=False):
    # Convert to YCrCb color space
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Split into 3 channels
    y, cr, cb = cv2.split(ycrcb_img)

    # Adaptive Thresholding for Y channel
    binary_y = cv2.adaptiveThreshold(
        y, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Thresholding Cr channel
    _, binary_cr = cv2.threshold(cr, 133, 173, cv2.THRESH_BINARY)
    # Thresholding Cb channel
    _, binary_cb = cv2.threshold(cb, 77, 127, cv2.THRESH_BINARY)

    # Morphological operations, consider experimenting with the kernel size and shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph_y = cv2.morphologyEx(binary_y, cv2.MORPH_CLOSE, kernel)
    morph_cr = cv2.morphologyEx(binary_cr, cv2.MORPH_CLOSE, kernel)
    morph_cb = cv2.morphologyEx(binary_cb, cv2.MORPH_CLOSE, kernel)

    # Combining the masks
    combined_binary = cv2.bitwise_and(morph_cr, morph_cb)
    combined_binary = cv2.bitwise_and(combined_binary, morph_y)

    # More morphology on combined mask if needed
    combined_morph = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel)

    # Apply the mask to get the segmented hand
    hand_segment = cv2.bitwise_and(image, image, mask=combined_morph)

    return hand_segment


# Setup access to the webcam
cap = cv2.VideoCapture(0)

# Define the paths for saving images and masks
raw_image_path = "path/to/raw_images"
mask_path = "path/to/masks"
os.makedirs(raw_image_path, exist_ok=True)
os.makedirs(mask_path, exist_ok=True)

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the captured frame
        preprocessed_frame = preprocess_and_visualize(frame, visualize=False)

        # Display the original frame and the preprocessed frame side by side
        combined_display = np.hstack((frame, preprocessed_frame))
        cv2.imshow("Original and Preprocessed Frames", combined_display)

        # Save the raw image and the preprocessed mask
        if cv2.waitKey(1) & 0xFF == ord("s"):  # Press 's' to save the images
            timestamp = int(time.time())
            img_name = f"raw_{timestamp}.png"
            mask_name = f"mask_{timestamp}.png"

            cv2.imwrite(os.path.join(raw_image_path, img_name), frame)
            cv2.imwrite(os.path.join(mask_path, mask_name), preprocessed_frame)

        # Break the loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()
