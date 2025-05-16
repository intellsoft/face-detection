import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
from ultralytics import YOLO
import os
import threading
from datetime import datetime
import torch
import atexit
import shutil
import queue

class VideoPlayer:
    def __init__(self, parent, video_path):
        self.parent = parent
        self.video_path = video_path
        self.cap = None
        self.playing = False
        self.fullscreen = False
        self.stop_flag = False
        self.setup_ui()

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(self.main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        self.btn_play = ttk.Button(control_frame, text="▶", command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=5)

        self.btn_fullscreen = ttk.Button(control_frame, text="⤢", command=self.toggle_fullscreen)
        self.btn_fullscreen.pack(side=tk.LEFT, padx=5)

        self.video_label = ttk.Label(self.main_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.config(text="⏸" if self.playing else "▶")
        if self.playing:
            self.stop_flag = False
            self.play_video()
        else:
            self.stop_video()

    def play_video(self):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.video_path)
            self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.playing and not self.stop_flag:
            ret, frame = self.cap.read()
            if ret:
                max_width = self.parent.winfo_width()
                max_height = self.parent.winfo_height()
                ratio = min(max_width/self.original_width, max_height/self.original_height)
                new_width = int(self.original_width * ratio)
                new_height = int(self.original_height * ratio)
                frame = cv2.resize(frame, (new_width, new_height))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.config(image=imgtk)
                self.video_label.image = imgtk
                self.video_label.after(25, self.play_video)
            else:
                self.stop_video()

    def stop_video(self):
        self.playing = False
        self.stop_flag = True
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_play.config(text="▶")

    def toggle_fullscreen(self):
        if self.fullscreen:
            self.parent.master.attributes('-fullscreen', False)
            self.fullscreen = False
        else:
            self.parent.master.attributes('-fullscreen', True)
            self.fullscreen = True

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Face Detection")
        self.root.state('zoomed')

        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.load_model()

        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue()
        self.temp_dir = "temp_frames"
        self.output_dir = ""
        self.thumbnails = []
        self.video_path = ""
        self.current_video_player = None

        self.setup_ui()
        self.root.after(100, self.process_queue)
        atexit.register(self.cleanup)

    def load_model(self):
        model_path = "yolov8n-face.pt"
        if not os.path.exists(model_path):
            try:
                YOLO("yolov8n.pt").export(format="pt", name="yolov8n-face")
            except Exception as e:
                messagebox.showerror("Error", f"خطا در دریافت مدل: {str(e)}")
                exit()
        self.model = YOLO(model_path).to(self.device)

    def setup_ui(self):
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        self.gallery_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.gallery_frame, weight=7)

        self.video_container = ttk.Frame(self.main_paned)
        self.main_paned.add(self.video_container, weight=3)

        control_frame = ttk.Frame(self.gallery_frame)
        control_frame.pack(fill=tk.X, pady=5)

        self.btn_select = ttk.Button(control_frame, text="انتخاب ویدئو", command=self.select_video)
        self.btn_select.pack(side=tk.LEFT, padx=5)

        self.btn_process = ttk.Button(control_frame, text="شروع پردازش", command=self.toggle_processing)
        self.btn_process.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(control_frame, text="توقف", state=tk.DISABLED, command=self.stop_processing)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(self.gallery_frame, bg='#2e2e2e')
        self.scrollbar = ttk.Scrollbar(self.gallery_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.scrollable_frame.bind("<Configure>", self.update_scrollregion)

        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.root.bind("<Configure>", self.on_window_resize)

    def on_window_resize(self, event):
        self.resize_gallery()
        if self.current_video_player:
            self.current_video_player.play_video()

    def resize_gallery(self):
        gallery_width = self.gallery_frame.winfo_width()
        thumbnail_size = 200
        cols = max(1, gallery_width // (thumbnail_size + 20))
        self.arrange_thumbnails(cols)

    def arrange_thumbnails(self, cols):
        for i, child in enumerate(self.scrollable_frame.winfo_children()):
            row = i // cols
            col = i % cols
            child.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')

    def update_scrollregion(self, event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def select_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")])
        if self.video_path:
            self.status_var.set(f"Selected: {os.path.basename(self.video_path)}")
            if self.current_video_player:
                self.current_video_player.main_frame.destroy()
            self.current_video_player = VideoPlayer(self.video_container, self.video_path)

    def toggle_processing(self):
        if not hasattr(self, 'processing_thread') or not self.processing_thread.is_alive():
            self.start_processing()
        else:
            self.stop_processing()

    def start_processing(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Please select a video file first!")
            return

        self.output_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.stop_event.clear()
        self.btn_process.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
        self.processing_thread.start()

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_count = 0

        while cap.isOpened() and not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.predict(frame, conf=0.5, device=self.device, verbose=False)
            if len(results[0].boxes) > 0:
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                filepath = os.path.join(self.output_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{timestamp:.2f}s.jpg")
                cv2.imwrite(filepath, frame)
                self.frame_queue.put((frame.copy(), filepath, timestamp))
            frame_count += 1
            self.status_var.set(f"Processed: {frame_count} frames")

        cap.release()
        self.status_var.set(f"Completed! Saved in '{self.output_dir}'")
        self.stop_event.set()

    def process_queue(self):
        try:
            while not self.frame_queue.empty():
                frame, filepath, timestamp = self.frame_queue.get()
                thumbnail = self.create_thumbnail(frame, filepath, timestamp)
                frame_label = ttk.Frame(self.scrollable_frame)

                img_label = ttk.Label(frame_label, image=thumbnail)
                img_label.image = thumbnail
                img_label.bind("<Button-1>", lambda e, t=timestamp: self.play_from_timestamp(t))
                img_label.grid(row=0, column=0)

                time_label = ttk.Label(frame_label, text=f"Time: {timestamp:.2f}s")
                time_label.grid(row=1, column=0)

                frame_label.grid()
                self.resize_gallery()
        finally:
            self.root.after(100, self.process_queue)

    def create_thumbnail(self, frame, filepath, timestamp):
        thumbnail = cv2.resize(frame, (200, 200))
        img_pil = Image.fromarray(cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        draw.text((10, 10), f"{timestamp:.2f}s", fill=(255, 0, 0), font=font)
        img_pil.save(os.path.join(self.temp_dir, os.path.basename(filepath)))
        return ImageTk.PhotoImage(img_pil)

    def play_from_timestamp(self, timestamp):
        if self.current_video_player:
            # Stop current playback and release resources
            self.current_video_player.stop_video()
            
            # Reset video capture with new position
            self.current_video_player.cap = cv2.VideoCapture(self.video_path)
            self.current_video_player.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            
            # Allow time for the capture to settle
            self.root.after(100, lambda: self.start_playback())

    def start_playback(self):
        if self.current_video_player:
            self.current_video_player.playing = True
            self.current_video_player.stop_flag = False
            self.current_video_player.btn_play.config(text="⏸")
            self.current_video_player.play_video()

    def cleanup(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def stop_processing(self):
        self.stop_event.set()
        self.btn_process.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceDetectionApp(root)
    root.mainloop()