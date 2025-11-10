import cv2
import numpy as np
import time
import threading
import logging
import socket
import json
import os
import sys
from datetime import datetime
from collections import deque

# Core ML/AI Libraries
try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available, YOLOv5 detection disabled")

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_AVAILABLE = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_AVAILABLE = True
    except ImportError:
        TFLITE_AVAILABLE = False
        print("Warning: TensorFlow Lite not available, MobileNet detection disabled")

# GPIO Library (Raspberry Pi specific)
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: RPi.GPIO not available, running in simulation mode")

# === CONFIGURATION CONSTANTS ===
# GPIO Pin Definitions (BCM numbering)
RED_LED_PIN = 17
YELLOW_LED_PIN = 27
GREEN_LED_PIN = 22
PEDESTRIAN_IR_PIN = 23

# Camera Configuration
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# AI Model Configuration
YOLO_CONFIDENCE_THRESHOLD = 0.4
MOBILENET_CONFIDENCE_THRESHOLD = 0.5
CONTOUR_AREA_THRESHOLD = 500  # cv2.contourArea > 500px filter

# Traffic Control Parameters
MIN_GREEN_TIME = 5    # seconds
MAX_GREEN_TIME = 30   # seconds
YELLOW_TIME = 2       # seconds
RED_TIME = 5          # seconds
PEDESTRIAN_CROSSING_TIME = 10  # seconds

# Vehicle Classes (COCO dataset)
VEHICLE_CLASSES = [2, 3, 5, 7, 8]  # car, motorcycle, bus, truck, train

# === LOGGING CONFIGURATION ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/traffic_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# === MAIN TRAFFIC LIGHT CONTROLLER CLASS ===
class AITrafficLightController:
    def __init__(self):
        self.running = False
        self.current_state = "RED"
        self.vehicle_count_history = deque(maxlen=10)
        self.detection_stats = {
            'vehicles_detected': 0,
            'pedestrians_detected': 0,
            'false_positives_filtered': 0,
            'total_frames_processed': 0
        }
        
        # Initialize components
        self.setup_gpio()
        self.setup_camera()
        self.load_ai_models()
        self.setup_remote_monitoring()
        
        logger.info("AI Traffic Light Controller initialized successfully")

    def setup_gpio(self):
        """Initialize GPIO pins for LED control and IR sensor"""
        if not GPIO_AVAILABLE:
            logger.warning("GPIO not available - running in simulation mode")
            return
            
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            
            # Setup LED output pins
            for pin in [RED_LED_PIN, YELLOW_LED_PIN, GREEN_LED_PIN]:
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
            
            # Setup IR sensor input pin
            GPIO.setup(PEDESTRIAN_IR_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
            
            # Initial state: RED light
            self.set_traffic_signal(red=True, yellow=False, green=False)
            logger.info("GPIO initialized successfully")
            
        except Exception as e:
            logger.error(f"GPIO initialization failed: {e}")

    def setup_camera(self):
        """Initialize camera with optimal settings"""
        try:
            self.camera = cv2.VideoCapture(CAMERA_INDEX)
            if not self.camera.isOpened():
                raise Exception("Camera not found")
            
            # Configure camera settings
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.camera.set(cv2.CAP_PROP_FPS, FPS)
            
            # Test camera capture
            ret, frame = self.camera.read()
            if not ret:
                raise Exception("Failed to capture test frame")
                
            logger.info(f"Camera initialized: {FRAME_WIDTH}x{FRAME_HEIGHT} @ {FPS}fps")
            
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            self.camera = None

    def load_ai_models(self):
        """Load YOLOv5 and MobileNet models"""
        self.yolo_model = None
        self.mobilenet_interpreter = None
        
        # Load YOLOv5 for vehicle detection
        if TORCH_AVAILABLE:
            try:
                logger.info("Loading YOLOv5 model...")
                self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', 
                                                device='cpu', force_reload=False)
                self.yolo_model.conf = YOLO_CONFIDENCE_THRESHOLD
                logger.info("YOLOv5 model loaded successfully")
            except Exception as e:
                logger.error(f"YOLOv5 loading failed: {e}")
        
        # Load MobileNet for pedestrian detection (if available)
        if TFLITE_AVAILABLE and os.path.exists('mobilenet_v2.tflite'):
            try:
                logger.info("Loading MobileNet TFLite model...")
                self.mobilenet_interpreter = tflite.Interpreter(model_path='mobilenet_v2.tflite')
                self.mobilenet_interpreter.allocate_tensors()
                self.mobilenet_input_details = self.mobilenet_interpreter.get_input_details()
                self.mobilenet_output_details = self.mobilenet_interpreter.get_output_details()
                logger.info("MobileNet TFLite model loaded successfully")
            except Exception as e:
                logger.error(f"MobileNet loading failed: {e}")

    def setup_remote_monitoring(self):
        """Setup SSH-like remote monitoring server"""
        try:
            self.monitoring_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.monitoring_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.monitoring_socket.bind(('0.0.0.0', 8888))
            self.monitoring_socket.listen(1)
            
            # Start monitoring thread
            self.monitoring_thread = threading.Thread(target=self.handle_remote_monitoring)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            logger.info("Remote monitoring server started on port 8888")
        except Exception as e:
            logger.error(f"Remote monitoring setup failed: {e}")

    def set_traffic_signal(self, red=False, yellow=False, green=False):
        """Control traffic light LEDs"""
        if GPIO_AVAILABLE:
            try:
                GPIO.output(RED_LED_PIN, red)
                GPIO.output(YELLOW_LED_PIN, yellow)
                GPIO.output(GREEN_LED_PIN, green)
            except Exception as e:
                logger.error(f"GPIO output error: {e}")
        
        # Update current state
        if red and not yellow and not green:
            self.current_state = "RED"
        elif yellow and not red and not green:
            self.current_state = "YELLOW"
        elif green and not red and not yellow:
            self.current_state = "GREEN"
        else:
            self.current_state = "TRANSITION"
            
        logger.debug(f"Traffic signal: RED={red}, YELLOW={yellow}, GREEN={green}")

    def detect_pedestrians_ir(self):
        """Detect pedestrians using IR sensor"""
        if not GPIO_AVAILABLE:
            return False
            
        try:
            return GPIO.input(PEDESTRIAN_IR_PIN) == GPIO.HIGH
        except Exception as e:
            logger.error(f"IR sensor read error: {e}")
            return False

    def detect_vehicles_yolo(self, frame):
        """Detect and count vehicles using YOLOv5"""
        if self.yolo_model is None:
            return 0
        
        try:
            # Run inference
            results = self.yolo_model(frame)
            
            vehicle_count = 0
            detections = results.xyxy[0].cpu().numpy()
            
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                class_id = int(cls)
                
                # Filter for vehicle classes and confidence
                if conf >= YOLO_CONFIDENCE_THRESHOLD and class_id in VEHICLE_CLASSES:
                    # Additional filtering by bounding box area
                    box_area = (x2 - x1) * (y2 - y1)
                    if box_area > CONTOUR_AREA_THRESHOLD:
                        vehicle_count += 1
                        
                        # Draw bounding box for debugging
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f'Vehicle: {conf:.2f}', (int(x1), int(y1-10)),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            return vehicle_count
            
        except Exception as e:
            logger.error(f"YOLOv5 detection error: {e}")
            return 0

    def detect_pedestrians_mobilenet(self, frame):
        """Detect pedestrians using MobileNet (TensorFlow Lite)"""
        if self.mobilenet_interpreter is None:
            return False
        
        try:
            # Preprocess frame
            input_shape = self.mobilenet_input_details[0]['shape']
            resized_frame = cv2.resize(frame, (input_shape[2], input_shape[1]))
            normalized_frame = resized_frame.astype(np.float32) / 255.0
            input_data = np.expand_dims(normalized_frame, axis=0)
            
            # Run inference
            self.mobilenet_interpreter.set_tensor(self.mobilenet_input_details[0]['index'], input_data)
            self.mobilenet_interpreter.invoke()
            
            # Get output
            output_data = self.mobilenet_interpreter.get_tensor(self.mobilenet_output_details[0]['index'])
            
            # Check for pedestrian detection (assuming binary classification)
            pedestrian_confidence = output_data[0][0]  # Adjust based on model output format
            
            return pedestrian_confidence > MOBILENET_CONFIDENCE_THRESHOLD
            
        except Exception as e:
            logger.error(f"MobileNet detection error: {e}")
            return False

    def apply_opencv_filtering(self, frame):
        """Apply OpenCV-based noise and shadow filtering"""
        try:
            # Convert to grayscale for contour detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Adaptive threshold to handle varying lighting
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area (remove small noise)
            filtered_contours = [c for c in contours if cv2.contourArea(c) > CONTOUR_AREA_THRESHOLD]
            
            # Update statistics
            self.detection_stats['false_positives_filtered'] += len(contours) - len(filtered_contours)
            
            return filtered_contours
            
        except Exception as e:
            logger.error(f"OpenCV filtering error: {e}")
            return []

    def calculate_dynamic_green_time(self, vehicle_count):
        """Calculate optimal green light duration using the patent formula"""
        # Formula: green_time = min(30, max(5, vehicle_count * 2))
        green_time = min(MAX_GREEN_TIME, max(MIN_GREEN_TIME, vehicle_count * 2))
        
        # Store in history for analysis
        self.vehicle_count_history.append(vehicle_count)
        
        logger.info(f"Dynamic green time calculated: {green_time}s for {vehicle_count} vehicles")
        return green_time

    def handle_pedestrian_priority(self):
        """Handle pedestrian crossing priority when no vehicles present"""
        logger.info("Pedestrian priority mode activated")
        
        # Ensure red for vehicles
        self.set_traffic_signal(red=True, yellow=False, green=False)
        time.sleep(2)  # Safety delay
        
        # Green for pedestrians (vehicles remain red)
        logger.info("Pedestrian crossing - vehicles stop")
        time.sleep(PEDESTRIAN_CROSSING_TIME)
        
        # Return to red
        self.set_traffic_signal(red=True, yellow=False, green=False)
        
        self.detection_stats['pedestrians_detected'] += 1

    def execute_vehicle_cycle(self, green_duration):
        """Execute normal vehicle traffic cycle"""
        logger.info(f"Vehicle cycle: GREEN={green_duration}s, YELLOW={YELLOW_TIME}s, RED={RED_TIME}s")
        
        # Green phase
        self.set_traffic_signal(red=False, yellow=False, green=True)
        time.sleep(green_duration)
        
        # Yellow phase
        self.set_traffic_signal(red=False, yellow=True, green=False)
        time.sleep(YELLOW_TIME)
        
        # Red phase
        self.set_traffic_signal(red=True, yellow=False, green=False)
        time.sleep(RED_TIME)

    def process_frame(self, frame):
        """Main frame processing pipeline"""
        try:
            self.detection_stats['total_frames_processed'] += 1
            
            # Apply OpenCV filtering
            filtered_contours = self.apply_opencv_filtering(frame)
            
            # Vehicle detection with YOLOv5
            vehicle_count = self.detect_vehicles_yolo(frame)
            
            # Pedestrian detection (IR sensor + optional MobileNet)
            pedestrian_ir = self.detect_pedestrians_ir()
            pedestrian_ai = self.detect_pedestrians_mobilenet(frame)
            pedestrian_detected = pedestrian_ir or pedestrian_ai
            
            # Update statistics
            self.detection_stats['vehicles_detected'] += vehicle_count
            
            return vehicle_count, pedestrian_detected
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return 0, False

    def handle_remote_monitoring(self):
        """Handle remote monitoring connections"""
        while self.running:
            try:
                conn, addr = self.monitoring_socket.accept()
                logger.info(f"Remote monitoring connection from {addr}")
                
                # Send system status
                status = {
                    'timestamp': datetime.now().isoformat(),
                    'current_state': self.current_state,
                    'statistics': self.detection_stats,
                    'vehicle_history': list(self.vehicle_count_history)
                }
                
                response = json.dumps(status, indent=2)
                conn.send(response.encode())
                conn.close()
                
            except Exception as e:
                logger.error(f"Remote monitoring error: {e}")
                time.sleep(1)

    def run(self):
        """Main control loop"""
        self.running = True
        logger.info("Starting AI Traffic Light Control System")
        
        if self.camera is None:
            logger.error("Camera not available - cannot start system")
            return
        
        try:
            while self.running:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    continue
                
                # Process current frame
                vehicle_count, pedestrian_detected = self.process_frame(frame)
                
                # Traffic control decision logic
                if vehicle_count == 0 and pedestrian_detected:
                    # Pedestrian priority mode
                    self.handle_pedestrian_priority()
                else:
                    # Normal vehicle traffic mode
                    green_time = self.calculate_dynamic_green_time(vehicle_count)
                    self.execute_vehicle_cycle(green_time)
                
                # Optional: Save debug frame
                if logging.getLogger().level <= logging.DEBUG:
                    debug_frame_path = f"/tmp/debug_frame_{int(time.time())}.jpg"
                    cv2.imwrite(debug_frame_path, frame)
                
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
        except Exception as e:
            logger.error(f"System error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up system resources")
        
        self.running = False
        
        # Turn off all lights
        if GPIO_AVAILABLE:
            self.set_traffic_signal(red=False, yellow=False, green=False)
            GPIO.cleanup()
        
        # Release camera
        if self.camera:
            self.camera.release()
        
        # Close monitoring socket
        if hasattr(self, 'monitoring_socket'):
            self.monitoring_socket.close()
        
        # Final statistics
        logger.info("System Statistics:")
        for key, value in self.detection_stats.items():
            logger.info(f"  {key}: {value}")

# === UTILITY FUNCTIONS ===
def install_dependencies():
    """Install required dependencies if missing"""
    dependencies = [
        'torch torchvision',
        'opencv-python',
        'numpy',
        'tflite-runtime'
    ]
    
    for dep in dependencies:
        try:
            os.system(f"pip3 install {dep}")
        except Exception as e:
            logger.warning(f"Failed to install {dep}: {e}")

def system_health_check():
    """Perform system health check"""
    checks = {
        'GPIO': GPIO_AVAILABLE,
        'PyTorch': TORCH_AVAILABLE,
        'TensorFlow Lite': TFLITE_AVAILABLE,
        'Camera': os.path.exists('/dev/video0'),
        'Storage': os.path.getsize('/') > 1000000  # >1MB free space
    }
    
    logger.info("System Health Check:")
    for component, status in checks.items():
        status_str = "OK" if status else "FAILED"
        logger.info(f"  {component}: {status_str}")
    
    return all(checks.values())

# === MAIN ENTRY POINT ===
if __name__ == "__main__":
    print("=" * 60)
    print("AI-Based Adaptive Traffic Light Control System")
    print("Raspberry Pi 5 + YOLOv5 + MobileNet + TensorFlow Lite")
    print("Authors: Swastik Mohanty, Dhruv Negi")
    print("Institution: Manipal University Jaipur")
    print("=" * 60)
    
    # System initialization
    if len(sys.argv) > 1 and sys.argv[1] == '--install-deps':
        install_dependencies()
        sys.exit(0)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--health-check':
        system_health_check()
        sys.exit(0)
    
    # Create and run traffic controller
    try:
        controller = AITrafficLightController()
        controller.run()
    except Exception as e:
        logger.error(f"System startup failed: {e}")
        sys.exit(1)
