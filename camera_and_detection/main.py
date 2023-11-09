import cv2

# Start capturing from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the captured image to YCrCb color space
    YCrCb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    # Display the resulting frame
    cv2.imshow("Webcam in YCrCb", YCrCb_frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
