# Simpan sebagai run_local.py
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import imutils
import numpy as np
import os
import threading
import time
import platform


# --- Muat Model (di luar loop agar tidak lambat) ---
print("Memuat model...")
model_path = 'deployment/Req/best.pt'

# Check if model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file tidak ditemukan di {model_path}")
    exit()

try:
    yolo_model = YOLO(model_path)  # Gunakan file .pt untuk performa terbaik di lokal
    print("Model berhasil dimuat.")
except Exception as e:
    print(f"Error saat memuat model: {e}")
    exit()

# Create output directory for saving frames
output_dir = "output_frames"
os.makedirs(output_dir, exist_ok=True)

class LicensePlateGUI:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        
        # Set window size
        self.window.geometry("1200x700")
        
        # Create main frame
        main_frame = ttk.Frame(window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Deteksi Plat Nomor Real-Time", 
                                font=("Helvetica", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=3, pady=10)
        
        # Video display canvas (left - unchanged)
        self.canvas = tk.Canvas(main_frame, width=800, height=500, bg="black")
        self.canvas.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Cropped plate preview (right - new)
        preview_frame = ttk.LabelFrame(main_frame, text="Cropped Plate", padding="5")
        preview_frame.grid(row=1, column=2, rowspan=2, sticky=(tk.N), padx=10)
        self.crop_canvas = tk.Canvas(preview_frame, width=300, height=150, bg="black")
        self.crop_canvas.grid(row=0, column=0, pady=5)
        self.crop_label = ttk.Label(preview_frame, text="-", font=("Helvetica", 12, "bold"))
        self.crop_label.grid(row=1, column=0, pady=5)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Status", padding="10")
        status_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Frame counter
        self.frame_label = ttk.Label(status_frame, text="Frame: 0")
        self.frame_label.grid(row=0, column=0, sticky=tk.W, padx=5)
        
        # Detection status
        self.detection_label = ttk.Label(status_frame, text="Deteksi: -")
        self.detection_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        
        # Count of detected plates
        self.plate_label = ttk.Label(status_frame, text="Jumlah Plat Terdeteksi: 0", 
                                     font=("Helvetica", 12, "bold"))
        self.plate_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Mulai", command=self.start_detection)
        self.start_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_detection, 
                                      state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)
        
        self.snapshot_button = ttk.Button(button_frame, text="Ambil Snapshot", 
                                          command=self.take_snapshot, state=tk.DISABLED)
        self.snapshot_button.grid(row=0, column=2, padx=5)
        
        # Variables
        self.cap = None
        self.running = False
        self.frame_count = 0
        self.current_frame = None
        self.last_plate_text = "-"
        self.last_plate_crop = None
        self.crop_photo = None
        
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Auto-start detection shortly after the window is ready
        self.window.after(200, self.start_detection)

    def preprocess_plate(self, plate_crop: np.ndarray) -> np.ndarray:
        """Enhance plate ROI for better OCR."""
        try:
            img = plate_crop.copy()
            if img is None or img.size == 0:
                return plate_crop
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Contrast enhancement (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            # Denoise while preserving edges
            gray = cv2.bilateralFilter(gray, d=7, sigmaColor=75, sigmaSpace=75)
            # Slight sharpen
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], dtype=np.float32)
            sharp = cv2.filter2D(gray, -1, kernel)
            # Upscale to help OCR (limit to avoid huge images)
            h, w = sharp.shape[:2]
            scale = 2 if max(h, w) < 200 else 1
            resized = cv2.resize(sharp, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            # Adaptive threshold to improve text/background separation
            bin_img = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 31, 7)
            # Morphological open to remove small noise
            kernel_m = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            clean = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel_m, iterations=1)
            return clean
        except Exception:
            return plate_crop

    def extract_plate_region(self, img_bgr: np.ndarray) -> np.ndarray:
        """Ekstraksi area plat mengikuti metode:
        grayscale -> bilateralFilter -> Canny -> findContours -> approxPolyDP -> mask -> crop.
        Mengembalikan citra ter-crop (grayscale). Jika gagal, kembalikan grayscale asli.
        """
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        bfilter = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None
        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break
        if location is not None:
            mask = np.zeros(gray.shape, np.uint8)
            cv2.drawContours(mask, [location], 0, 255, -1)
            _ = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray[x1:x2+1, y1:y2+1]
            return cropped_image
        else:
            return gray

    def open_camera(self):
        """Try to open the camera robustly across OS and indices"""
        system = platform.system()
        # API backends preference per OS
        if system == "Windows":
            apis = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
        elif system == "Darwin":
            apis = [cv2.CAP_AVFOUNDATION, 0]
        else:
            apis = [cv2.CAP_V4L2, 0]

        indices = [0, 1, 2, 3]
        for api in apis:
            for idx in indices:
                cap = cv2.VideoCapture(idx, api) if api != 0 else cv2.VideoCapture(idx)
                if cap.isOpened():
                    return cap
                cap.release()
        return None

    def start_detection(self):
        """Start video capture and detection"""
        if self.running:
            return

        self.detection_label.config(text="Mencoba membuka kamera...")
        self.cap = self.open_camera()

        if not self.cap or not self.cap.isOpened():
            self.detection_label.config(text="Error: Tidak bisa membuka kamera")
            return

        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.snapshot_button.config(state=tk.NORMAL)

        # Start video loop in separate thread
        self.video_thread = threading.Thread(target=self.video_loop, daemon=True)
        self.video_thread.start()
    
    def stop_detection(self):
        """Stop video capture"""
        self.running = False
        if self.cap:
            self.cap.release()
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.snapshot_button.config(state=tk.DISABLED)
        self.detection_label.config(text="Deteksi: Stopped")
    
    def take_snapshot(self):
        """Save current frame"""
        if self.current_frame is not None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"snapshot_{timestamp}.jpg")
            cv2.imwrite(output_path, self.current_frame)
            self.detection_label.config(text=f"Snapshot disimpan: {output_path}")
            print(f"✓ Snapshot disimpan: {output_path}")
    
    def video_loop(self):
        """Main video processing loop"""
        while self.running:
            success, frame = self.cap.read()
            
            if not success:
                self.window.after(0, lambda: self.detection_label.config(text="Error: Tidak bisa membaca frame"))
                break
            
            self.frame_count += 1
            
            try:
                # Run YOLO detection
                results = yolo_model(frame, conf=0.4)[0]
                
                detection_found = False
                detected_plates = []
                
                if results.boxes:
                    for box in results.boxes:
                        detection_found = True
                        x1, y1, x2, y2 = [int(i) for i in box.xyxy[0]]
                        
                        plate_crop = frame[y1:y2, x1:x2]
                        
                        if plate_crop.size > 0 and plate_crop.shape[0] > 10 and plate_crop.shape[1] > 10:
                            # Gambar kotak tanpa OCR dan simpan crop terakhir untuk preview
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            # Opsional: tampilkan label generik
                            cv2.putText(frame, "Plat", (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            self.last_plate_crop = plate_crop.copy()
                            detected_plates.append(1)
                
                self.current_frame = frame.copy()
                
                # Update GUI using window.after() for thread safety
                self.window.after(0, lambda f=frame, d=detection_found, p=detected_plates: self.update_gui(f, d, p))
                
            except Exception as detection_error:
                print(f"Error saat deteksi frame {self.frame_count}: {detection_error}")
            
            # Add small delay to prevent excessive CPU usage
            time.sleep(0.03)  # ~30 FPS
        
        # Cleanup
        if self.cap:
            self.cap.release()
    
    def update_gui(self, frame, detection_found, detected_plates):
        """Update GUI with new frame and detection info"""
        # Resize frame to fit canvas
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        
        # Calculate scaling to fit canvas
        canvas_width = 800
        canvas_height = 500
        scale = min(canvas_width/w, canvas_height/h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        frame_resized = cv2.resize(frame_rgb, (new_w, new_h))
        
        # Convert to PIL Image
        img = Image.fromarray(frame_resized)
        self.photo = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo, anchor=tk.CENTER)
        
        # Update labels
        self.frame_label.config(text=f"Frame: {self.frame_count}")
        
        if detection_found:
            count = len(detected_plates) if isinstance(detected_plates, (list, tuple)) else int(detected_plates) if detected_plates else 0
            self.detection_label.config(text=f"Deteksi: ✓ Plat ditemukan ({count})")
            self.plate_label.config(text=f"Jumlah Plat Terdeteksi: {count}")
        else:
            self.detection_label.config(text="Deteksi: Mencari plat...")
            self.plate_label.config(text="Jumlah Plat Terdeteksi: 0")
        
        # Update cropped plate preview on the right (if available)
        if self.last_plate_crop is not None and self.last_plate_crop.size > 0:
            crop_bgr = self.last_plate_crop
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            ch, cw = crop_rgb.shape[:2]
            target_w, target_h = 300, 150
            scale_c = min(target_w / cw, target_h / ch)
            new_cw = max(1, int(cw * scale_c))
            new_ch = max(1, int(ch * scale_c))
            crop_resized = cv2.resize(crop_rgb, (new_cw, new_ch))
            crop_img = Image.fromarray(crop_resized)
            self.crop_photo = ImageTk.PhotoImage(image=crop_img)
            # Clear and draw centered
            self.crop_canvas.delete("all")
            self.crop_canvas.create_image(target_w//2, target_h//2, image=self.crop_photo, anchor=tk.CENTER)
            # Update label for crop preview (no OCR text)
            self.crop_label.config(text="-")
        
    def on_closing(self):
        """Handle window closing"""
        self.running = False
        if self.cap:
            self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateGUI(root, "Indonesian License Plate Detection")
    root.mainloop()