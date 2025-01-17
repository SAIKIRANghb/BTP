import os
from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import numpy as np
from collections import deque
import time
from flask_sock import Sock
import io

app = Flask(__name__)
sock = Sock(app)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

class GolfBallTracker:
    def __init__(self, buffer_size=32, target_fps=30):
        self.frame_height = 480  # Default height, will be updated with first frame
        self.min_radius = int(self.frame_height * 0.015)
        self.max_radius = int(self.frame_height * 0.035)
        self.positions = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        self.velocity_smoothing = 0.7
        self.smoothed_velocity_x = 0
        self.smoothed_velocity_y = 0
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps

    def calculate_velocity_components(self, positions, timestamps):
        if len(positions) < 2:
            return 0, 0
        velocities_x = []
        velocities_y = []
        for i in range(1, len(positions)):
            time_diff = timestamps[i] - timestamps[i - 1]
            if time_diff > 0:
                velocity_x = (positions[i][0] - positions[i - 1][0]) / time_diff
                velocity_y = (positions[i][1] - positions[i - 1][1]) / time_diff
                velocities_x.append(velocity_x)
                velocities_y.append(velocity_y)
        if not velocities_x:
            return 0, 0
        current_velocity_x = np.mean(velocities_x)
        current_velocity_y = np.mean(velocities_y)
        self.smoothed_velocity_x = (
            self.velocity_smoothing * self.smoothed_velocity_x
            + (1 - self.velocity_smoothing) * current_velocity_x
        )
        self.smoothed_velocity_y = (
            self.velocity_smoothing * self.smoothed_velocity_y
            + (1 - self.velocity_smoothing) * current_velocity_y
        )
        return self.smoothed_velocity_x, self.smoothed_velocity_y

    def detect_golf_ball(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=25,
            minRadius=self.min_radius,
            maxRadius=self.max_radius,
        )
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            return circles[0]
        return None

    def draw_tracking_info(self, frame, center, radius, velocity_x, velocity_y):
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        cv2.circle(frame, center, 2, (0, 0, 255), -1)
        scale = 1
        end_point_x = (int(center[0] + velocity_x * scale), center[1])
        cv2.arrowedLine(frame, center, end_point_x, (0, 0, 255), 2)
        end_point_y = (center[0], int(center[1] + velocity_y * scale))
        cv2.arrowedLine(frame, center, end_point_y, (255, 0, 0), 2)
        end_point_total = (
            int(center[0] + velocity_x * scale),
            int(center[1] + velocity_y * scale),
        )
        cv2.arrowedLine(frame, center, end_point_total, (0, 255, 0), 2)
        cv2.putText(frame, f"Diameter: {radius*2}px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Vx: {velocity_x:.1f} px/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Vy: {velocity_y:.1f} px/s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        total_speed = np.sqrt(velocity_x**2 + velocity_y**2)
        cv2.putText(frame, f"Speed: {total_speed:.1f} px/s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def process_frame(self, frame):
        # Update frame height if needed
        if self.frame_height != frame.shape[0]:
            self.frame_height = frame.shape[0]
            self.min_radius = int(self.frame_height * 0.015)
            self.max_radius = int(self.frame_height * 0.035)

        ball = self.detect_golf_ball(frame)
        if ball is not None:
            x, y, r = ball
            current_time = time.time()
            self.positions.append((x, y))
            self.timestamps.append(current_time)
            velocity_x, velocity_y = self.calculate_velocity_components(
                list(self.positions), list(self.timestamps)
            )
            self.draw_tracking_info(frame, (x, y), r, velocity_x, velocity_y)

        return frame

    def run_video(self, video_source):
        cap = cv2.VideoCapture(video_source)
        try:
            next_frame_time = time.time()
            while True:
                current_time = time.time()
                
                # Wait until it's time for the next frame
                if current_time < next_frame_time:
                    time.sleep(next_frame_time - current_time)
                    continue
                
                ret, frame = cap.read()
                if not ret:
                    break
                    
                processed_frame = self.process_frame(frame)
                
                yield processed_frame
                next_frame_time = current_time + self.frame_time
                
        finally:
            cap.release()

@sock.route('/ws')
def websocket(ws):
    tracker = GolfBallTracker(target_fps=30)
    
    while True:
        try:
            # Receive frame from client
            frame_data = ws.receive()
            
            # Convert binary data to OpenCV frame
            frame_array = np.frombuffer(frame_data, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            
            # Process frame
            processed_frame = tracker.process_frame(frame)
            
            # Convert processed frame back to JPEG
            _, buffer = cv2.imencode('.jpg', processed_frame)
            
            # Send processed frame back to client
            ws.send(buffer.tobytes())
        except Exception as e:
            print(f"WebSocket error: {e}")
            break

def generate_frames(video_source=0):
    tracker = GolfBallTracker(target_fps=30)
    for frame in tracker.run_video(video_source):
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route("/webcam_feed")
def webcam_feed():
    return Response(generate_frames(video_source=0), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(video_source="ball_tracking.mp4"), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return redirect(url_for("home"))
    file = request.files["video"]
    if file.filename == "":
        return redirect(url_for("home"))
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    return redirect(url_for("play_uploaded_video", filename=file.filename))

@app.route("/play_uploaded/<filename>")
def play_uploaded_video(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return Response(generate_frames(video_source=file_path), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)