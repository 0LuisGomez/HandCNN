import cv2

# Configuración de la cámara web
cap = cv2.VideoCapture(0)  # 0 es generalmente la cámara web predeterminada

try:
    while True:
        # Captura cuadro por cuadro
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Verificar si el cuadro capturado está vacío
        if frame is None:
            print("Empty frame captured from the camera")
            continue

        # Resto del procesamiento del cuadro
        # ...
finally:
    # Liberar la captura cuando todo esté hecho
    cap.release()
    cv2.destroyAllWindows()
