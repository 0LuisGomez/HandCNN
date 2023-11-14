import tkinter as tk
import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from hand_detection import HandDetector
from threading import Thread
import time


class Application(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("HCI Detection Interface")
        self.geometry("1366x768")  # Fullscreen for a 1366x768 resolution

        # Initialize HandDetector
        self.hand_detector = HandDetector()

        # Setup the camera capture
        self.video_capture = cv2.VideoCapture(0)

        # Create frames for each section
        left_frame = ctk.CTkFrame(master=self, width=200, height=768)
        left_frame.pack(side="left", fill="y")

        camera_frame = ctk.CTkFrame(
            master=self, width=926, height=768
        )  # Adjust the size to better fit the window
        camera_frame.pack(side="left", fill="both", expand=True)

        right_frame = ctk.CTkFrame(master=self, width=240, height=768)
        right_frame.pack(side="left", fill="y")

        # Dropdown menu for layer selection, positioned at the bottom of left_frame
        layer_selector_label = ctk.CTkLabel(master=left_frame, text="Layer Selector")
        layer_selector_label.pack(side="bottom", padx=10, pady=10)

        layer_selector = ctk.CTkOptionMenu(
            master=left_frame, values=["Layer 1", "Layer 2", "Camera Only"]
        )
        layer_selector.pack(side="bottom", padx=10, pady=10)

        # Title and Information labels in the right_frame
        title_label = ctk.CTkLabel(master=right_frame, text="Title", font=("Arial", 24))
        title_label.pack(pady=10)

        information_label = ctk.CTkLabel(
            master=right_frame, text="Information", font=("Arial", 16)
        )
        information_label.pack(pady=10)

        # Camera Label
        self.camera_label = ctk.CTkLabel(master=camera_frame)
        self.camera_label.pack(expand=True)

        self.video_loop()  # Start the video loop

    def video_loop(self):
        ret, frame = self.video_capture.read()
        if ret:
            # Process the frame through HandDetector
            frame = self.hand_detector.process(frame)
            # Convert the frame to a format Tkinter can use
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)
            # Update the GUI with the new image
            self.camera_label.configure(image=img)
            self.camera_label.image = img

    # Schedule the next 'video_loop' call
    self.after(10, self.video_loop)  # 10 ms delay for roughly 100 frames per second

    def update_image(self, img):
        # Update the camera_label with the new image
        self.camera_label.configure(image=img)
        self.camera_label.image = img

    def on_closing(self):
        self.video_capture.release()
        self.hand_detector.release()
        self.destroy()


# Create the main window and run the application
app = Application()
app.protocol("WM_DELETE_WINDOW", app.on_closing)  # Ensure the app closes cleanly
app.mainloop()
