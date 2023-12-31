import numpy as np
import cv2


# Preprocessing Function
def preprocess_and_visualize(
    image,
    visualize=True,
    blur_kernel_size=(11, 11),
    adaptive_thresh_block_size=11,
    adaptive_thresh_C=3,
    morph_kernel_size=(3, 3),
    area_threshold=1000,
):
    # Optional Gaussian Blur for Noise Reduction
    if blur_kernel_size is not None:
        image = cv2.GaussianBlur(image, blur_kernel_size, 0)

    # Convert to YCrCb color space
    ycrcb_img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # Split into 3 channels
    y, cr, cb = cv2.split(ycrcb_img)

    # Adaptive Thresholding for Y channel
    binary_y = cv2.adaptiveThreshold(
        y,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        adaptive_thresh_block_size,
        adaptive_thresh_C,
    )

    # Thresholding Cr and Cb channels
    _, binary_cr = cv2.threshold(
        cr, 133, 173, cv2.THRESH_BINARY
    )  # Consider making these thresholds parameters
    _, binary_cb = cv2.threshold(
        cb, 77, 127, cv2.THRESH_BINARY
    )  # Consider making these thresholds parameters

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel_size)
    morph_y = cv2.morphologyEx(binary_y, cv2.MORPH_CLOSE, kernel)
    morph_cr = cv2.morphologyEx(binary_cr, cv2.MORPH_CLOSE, kernel)
    morph_cb = cv2.morphologyEx(binary_cb, cv2.MORPH_CLOSE, kernel)

    # Combining the masks
    combined_binary = cv2.bitwise_and(morph_cr, morph_cb)
    combined_binary = cv2.bitwise_and(combined_binary, morph_y)

    # More morphology on combined mask
    combined_morph = cv2.morphologyEx(combined_binary, cv2.MORPH_CLOSE, kernel)

    # Optional step: Remove small contours to reduce noise
    contours, _ = cv2.findContours(
        combined_morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    for cnt in contours:
        if cv2.contourArea(cnt) < area_threshold:
            cv2.drawContours(combined_morph, [cnt], -1, 0, -1)

    # Apply the mask to get the segmented hand
    hand_segment = cv2.bitwise_and(image, image, mask=combined_morph)

    # Visualization for debugging
    if visualize:
        cv2.imshow("Preprocessed Image", hand_segment)
        cv2.imshow("Mask", combined_morph)
    return hand_segment


def detect_hand_orientation_contour_convex_hull(image):
    # Preprocessing the image to segment the hand
    preprocessed_image = preprocess_and_visualize(image, visualize=True)

    # Convert to grayscale
    gray = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2GRAY)

    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Filtrar contornos pequeños, asumiendo que el contorno más grande es la mano
        largest_contour = max(contours, key=cv2.contourArea)
        if (
            cv2.contourArea(largest_contour) < 1000
        ):  # Ajusta este valor según sea necesario
            return preprocessed_image

        # [Código anterior para obtener la caja delimitadora y la orientación]

        # Dibujar el contorno más grande
        cv2.drawContours(preprocessed_image, [largest_contour], -1, (0, 255, 0), 2)

        # Encontrar y dibujar el convex hull
        hull = cv2.convexHull(largest_contour)
        cv2.drawContours(preprocessed_image, [hull], -1, (0, 0, 255), 2)

        # Calcular y dibujar los defectos de convexidad
        hull_indices = cv2.convexHull(largest_contour, returnPoints=False)
        if len(hull_indices) > 3:  # Verificar si el convex hull es adecuado
            defects = cv2.convexityDefects(largest_contour, hull_indices)
            if defects is not None:
                for i in range(defects.shape[0]):
                    start_point, end_point, far_point, _ = defects[i, 0]
                    start = tuple(largest_contour[start_point][0])
                    end = tuple(largest_contour[end_point][0])
                    far = tuple(largest_contour[far_point][0])
                    cv2.line(preprocessed_image, start, end, (255, 0, 0), 2)
                    cv2.circle(preprocessed_image, far, 5, (0, 255, 0), -1)
                else:
                    pass

    return preprocessed_image


# Configuración de la cámara web
cap = cv2.VideoCapture(0)  # 0 es generalmente la cámara web predeterminada

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        try:
            processed_frame = detect_hand_orientation_contour_convex_hull(frame)
            cv2.imshow("Hand Processing", processed_frame)
        except Exception as e:
            pass

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
except Exception as e:
    pass
finally:
    cap.release()
    cv2.destroyAllWindows()
