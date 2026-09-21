import serial
import struct
import numpy as np
import time
import cv2
import open3d as o3d
import threading
import queue
import os
from datetime import datetime
import statistics
import sys

# =====================================================
# CONFIGURATION
# =====================================================
# Hardware
SERIAL_PORT = 'COM8'  # Adjust for your Pi (e.g., /dev/ttyUSB0)
BAUD_RATE = 921600

# Sensor Geometry
START_ANGLE = -48.0
ANGULAR_RES = 0.3516
MIN_RANGE_M = 0.10
MAX_RANGE_M = 4.0  # Extended slightly for road width

# Calibration
CALIBRATION_FRAMES = 4500
GRID_CELL_SIZE = 0.05
MAX_NEIGHBOR_JUMP = 0.15

# Physics (Speed)
# Updated to 25 km/h for road testing to minimize Z-axis compression
VEHICLE_SPEED_KMPH = 25.0
VEHICLE_SPEED_MPS = VEHICLE_SPEED_KMPH / 3.6

# Detection Logic
# [FIX 1] Persistence: Vehicle must be seen for N frames to confirm it's real
REQUIRED_PERSISTENCE = 3
TRIGGER_THRESHOLD = 15  # Keep sensitive, but rely on Persistence to filter noise
IDLE_TIMEOUT = 0.5  # Reduced to 0.5s for faster cut-off on roads

# Image Generation
X_IMG_RANGE = (-1, 1)
Y_IMG_RANGE = (-1, 1)
GRID_RES = 0.005  # 5mm per pixel
MAX_DIST_INTENSITY = 50.0

# Queues
# frame_queue: Raw data from Sensor -> Main Loop
# processing_queue: Extracted Object -> Image Generator
frame_queue = queue.Queue()
processing_queue = queue.Queue()


# =====================================================
# HELPER: ROBUST MEDIAN (MAD)
# =====================================================
def robust_median(distances):
    if len(distances) < 5:
        return None
    median = statistics.median(distances)
    deviations = [abs(d - median) for d in distances]
    mad = statistics.median(deviations)
    if mad == 0:
        return median
    filtered = [d for d in distances if abs(d - median) <= 3 * mad]
    return statistics.median(filtered) if filtered else None


# =====================================================
# WORKER: BACKGROUND IMAGE PROCESSOR
# =====================================================
def background_processor():
    print("🧵 Background Processor Started")
    while True:
        try:
            extracted_stack, ts = processing_queue.get()

            # Safety Check: Too few points crash Open3D
            if len(extracted_stack) < 30:
                print(f"⚠️ Vehicle {ts} Skipped: Too few points ({len(extracted_stack)})")
                processing_queue.task_done()
                continue

            folder = f"vehicle_{ts}"
            os.makedirs(folder, exist_ok=True)

            # 1. Point Cloud Cleaning
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.array(extracted_stack, dtype=np.float64))

            # Statistical Removal (Snow/Dust)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=25, std_ratio=0.8)
            if len(pcd.points) == 0:
                processing_queue.task_done();
                continue

            # Radius Removal (Flying Pixels)
            pcd, _ = pcd.remove_radius_outlier(nb_points=12, radius=0.06)
            if len(pcd.points) == 0:
                processing_queue.task_done();
                continue

            # 2. Save Side View PCD
            # Rotate: Look from side
            R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64)
            pts_side = (R @ np.asarray(pcd.points).T).T

            pcd_side = o3d.geometry.PointCloud()
            pcd_side.points = o3d.utility.Vector3dVector(pts_side)
            o3d.io.write_point_cloud(f"{folder}/side_view.pcd", pcd_side)

            # 3. Generate Image
            mask = (
                    (pts_side[:, 0] >= X_IMG_RANGE[0]) & (pts_side[:, 0] <= X_IMG_RANGE[1]) &
                    (pts_side[:, 1] >= Y_IMG_RANGE[0]) & (pts_side[:, 1] <= Y_IMG_RANGE[1])
            )
            img_points = pts_side[mask]

            if len(img_points) > 0:
                x_bins = int((X_IMG_RANGE[1] - X_IMG_RANGE[0]) / GRID_RES)
                y_bins = int((Y_IMG_RANGE[1] - Y_IMG_RANGE[0]) / GRID_RES)
                bev = np.zeros((y_bins, x_bins), dtype=np.float32)

                # Map points to grid
                xi = ((img_points[:, 0] - X_IMG_RANGE[0]) / GRID_RES).astype(np.int32)
                yi = ((img_points[:, 1] - Y_IMG_RANGE[0]) / GRID_RES).astype(np.int32)

                valid = (xi >= 0) & (xi < x_bins) & (yi >= 0) & (yi < y_bins)
                xi, yi = xi[valid], yi[valid]

                # Intensity Mapping
                bev[yi, xi] = 1.0  # Binary occupancy for clearer shape

                # Convert to Image
                bev_img = (bev * 255).astype(np.uint8)
                bev_img = np.flipud(bev_img)

                # [FIX 2] Gap Filling (Dilation)
                # Stretches pixels horizontally to connect the "Barcode" lines
                # Kernel is (Height=1, Width=5) -> Horizontal stretch only
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
                bev_img = cv2.dilate(bev_img, kernel, iterations=2)

                # Smooth edges
                bev_img = cv2.GaussianBlur(bev_img, (3, 3), 0)

                cv2.imwrite(f"{folder}/side_view_image.png", bev_img)

            print(f"✅ Vehicle {ts} Saved")
            processing_queue.task_done()

        except Exception as e:
            print(f"❌ Error in Image Proc: {e}")
            processing_queue.task_done()


# =====================================================
# MAIN CLASS
# =====================================================
class TollPlazaSystem:
    def __init__(self):
        # [FIX 3] Threaded Serial Reader Setup
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            self.ser.reset_input_buffer()
        except Exception as e:
            print(f"❌ Serial Port Error: {e}")
            sys.exit(1)

        self.background_matrix = set()
        self.running = True

        # Detection State
        self.is_stacking = False
        self.current_full_stack = []
        self.current_extracted = []
        self.last_detection_time = time.time()
        self.start_capture_time = None
        self.consecutive_triggers = 0  # [FIX 4] Persistence Counter
        self.pre_trigger_buffer = []  # To save the frames *before* trigger confirmed

    # -------------------------------------------------
    # THREAD 1: SERIAL READER (High Speed)
    # -------------------------------------------------
    def serial_worker(self):
        print("⚡ Serial Worker Thread Live")
        internal_buffer = bytearray()

        while self.running:
            try:
                if self.ser.in_waiting > 0:
                    # Read large chunks (up to 4KB) at once
                    chunk = self.ser.read(min(self.ser.in_waiting, 4096))
                    internal_buffer.extend(chunk)
                else:
                    time.sleep(0.001)
                    continue

                # Parse Buffer for Packets
                while len(internal_buffer) > 8:
                    header_idx = internal_buffer.find(b'\xfc\xfd\xfe\xff')

                    if header_idx == -1:
                        internal_buffer = internal_buffer[-3:]
                        break

                    if header_idx > 0:
                        internal_buffer = internal_buffer[header_idx:]

                    if len(internal_buffer) < 6: break

                    size = struct.unpack('<H', internal_buffer[4:6])[0]
                    total_len = 4 + 2 + size + 2

                    if len(internal_buffer) < total_len: break

                    packet = internal_buffer[:total_len]
                    internal_buffer = internal_buffer[total_len:]

                    # Extract Payload
                    body = packet[6:-2]
                    if len(body) >= 2 and struct.unpack('<H', body[:2])[0] == 50011:
                        # Push raw body to main loop
                        frame_queue.put(body[3:])

            except Exception as e:
                print(f"❌ Serial Error: {e}")
                time.sleep(1)

    # -------------------------------------------------
    # HELPER: PARSE BYTES
    # -------------------------------------------------
    def parse_raw_bytes_to_xy(self, data):
        pts = []
        for i in range(len(data) // 2):
            d_mm = struct.unpack('<H', data[i * 2:i * 2 + 2])[0]
            if 100 < d_mm < 3500:  # Filter 10cm to 3.5m
                dist = d_mm / 1000.0
                ang = np.radians(START_ANGLE + i * ANGULAR_RES)
                pts.append([dist * np.cos(ang), dist * np.sin(ang)])
        return pts

    # -------------------------------------------------
    # CALIBRATION
    # -------------------------------------------------
    def calibrate(self):
        print(f"⌛ Robust Calibration ({CALIBRATION_FRAMES} frames)...")
        while not frame_queue.empty(): frame_queue.get()  # Flush old data

        frames = 0
        beam_history = {}

        while frames < CALIBRATION_FRAMES:
            try:
                raw_data = frame_queue.get(timeout=1.0)
                data = raw_data  # It's already the body bytes
                for i in range(len(data) // 2):
                    d = struct.unpack('<H', data[i * 2:i * 2 + 2])[0]
                    if d > 0:
                        beam_history.setdefault(i, []).append(d)
                frames += 1
                if frames % 500 == 0: print(f"    {frames}/{CALIBRATION_FRAMES}")
            except queue.Empty:
                continue

        print("    Computing Median & Dilating...")
        idxs = sorted(beam_history.keys())
        for idx in idxs:
            robust_mm = robust_median(beam_history[idx])
            if robust_mm is None: continue
            dist_m = robust_mm / 1000.0

            if not (MIN_RANGE_M <= dist_m <= MAX_RANGE_M): continue

            angle = np.radians(START_ANGLE + idx * ANGULAR_RES)
            x, y = dist_m * np.cos(angle), dist_m * np.sin(angle)

            ix, iy = int(np.floor(x / GRID_CELL_SIZE)), int(np.floor(y / GRID_CELL_SIZE))

            # [FIX 5] Dilation: Mark cell AND neighbors as background
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    self.background_matrix.add((ix + dx, iy + dy))

        print(f"✅ Calibration Done. Background Cells: {len(self.background_matrix)}")

    # -------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------
    def run_forever(self):
        # Start Threads
        threading.Thread(target=background_processor, daemon=True).start()
        threading.Thread(target=self.serial_worker, daemon=True).start()

        # Allow serial to settle
        time.sleep(1.0)
        self.calibrate()
        print("🚀 System Live (Persistence Mode Enabled)")

        while True:
            try:
                raw_data = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Parse
            raw_xy = self.parse_raw_bytes_to_xy(raw_data)

            # Calculate Z (Time dimension)
            t = time.time()
            if self.is_stacking:
                z = (t - self.start_capture_time) * VEHICLE_SPEED_MPS
            else:
                z = 0

            raw_xyz = [[p[0], p[1], z] for p in raw_xy]

            # Background Subtraction
            extracted = []
            for p in raw_xy:
                ix, iy = int(np.floor(p[0] / GRID_CELL_SIZE)), int(np.floor(p[1] / GRID_CELL_SIZE))

                # Check neighbors (Robust Background Check)
                is_background = False
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if (ix + dx, iy + dy) in self.background_matrix:
                            is_background = True
                            break
                    if is_background: break

                if not is_background:
                    extracted.append([p[0], p[1], z])

            # =========================================
            # [FIX 1 Logic] PERSISTENCE CHECK
            # =========================================
            if len(extracted) > TRIGGER_THRESHOLD:
                # Potential Object Detected
                self.consecutive_triggers += 1

                # Buffer this frame so we don't lose the start of the car
                self.pre_trigger_buffer.append((raw_xyz, extracted))
                if len(self.pre_trigger_buffer) > 10:
                    self.pre_trigger_buffer.pop(0)  # Keep buffer small

                # Only START if we see it for N consecutive frames
                if self.consecutive_triggers >= REQUIRED_PERSISTENCE:
                    if not self.is_stacking:
                        print("🚗 Vehicle Confirmed (Persistence Met)")
                        self.is_stacking = True
                        self.start_capture_time = time.time() - (
                                    0.02 * len(self.pre_trigger_buffer))  # Adjust start time back
                        self.current_full_stack = []
                        self.current_extracted = []

                        # Add the buffered frames that happened before confirmation
                        for b_xyz, b_ext in self.pre_trigger_buffer:
                            # Re-adjust Z for buffered frames
                            # (Simplified: just add them, precise Z correction is minor here)
                            self.current_full_stack.extend(b_xyz)
                            self.current_extracted.extend(b_ext)
                        self.pre_trigger_buffer = []

                    # Standard Recording
                    self.current_full_stack.extend(raw_xyz)
                    self.current_extracted.extend(extracted)
                    self.last_detection_time = time.time()

            else:
                # No object seen in this frame
                self.consecutive_triggers = 0  # Reset persistence counter
                self.pre_trigger_buffer = []  # Clear buffer

                # If we were recording, check for timeout
                if self.is_stacking and (time.time() - self.last_detection_time > IDLE_TIMEOUT):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    processing_queue.put((list(self.current_extracted), ts))
                    self.is_stacking = False
                    print(f"🏁 Vehicle {ts} Finished")


if __name__ == "__main__":
    sys = TollPlazaSystem()
    try:
        sys.run_forever()
    except KeyboardInterrupt:
        sys.running = False