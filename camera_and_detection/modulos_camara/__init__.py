import cv2
import numpy as np
from keras.preprocessing.image import img_to_array


def preprocess_image(image):
    # Convert to YCrCb color space
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Assuming you have a background image to subtract
    # background_img = cv2.imread('path_to_background_image.jpg')
    # background_ycrcb = cv2.cvtColor(background_img, cv2.COLOR_BGR2YCrCb)
    # ycrcb_img = cv2.absdiff(ycrcb_img, background_ycrcb)

    # Split into 3 channels
    y, cr, cb = cv2.split(ycrcb_img)

    # Thresholding each channel to get binary masks
    _, binary_y = cv2.threshold(y, 80, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, binary_cr = cv2.threshold(cr, 133, 173, cv2.THRESH_BINARY)
    _, binary_cb = cv2.threshold(cb, 77, 127, cv2.THRESH_BINARY)

    # Morphological operations
    kernel = np.ones((3, 3), np.uint8)
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

    # Optional: Canny edge detection for contouring
    # edges = cv2.Canny(hand_segment, 100, 200)

    # Resize for CNN input and normalize
    resized = cv2.resize(
        hand_segment, (224, 224)
    )  # Example size, adjust to your CNN input
    cnn_input = img_to_array(resized)
    cnn_input = np.expand_dims(cnn_input, axis=0)
    cnn_input /= 255.0

    return cnn_input


# Example usage
# Load an image
image = cv2.imread("path_to_your_image.jpg")
preprocessed_image = preprocess_image(image)

# Now you can feed 'preprocessed_image' to your CNN for further processing
