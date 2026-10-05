running yolov8s on RPI5 using pi cam



1.First enable ssh, vnc or install xrdp



2.fish makes easy to type cmds(optional)



3.Make sure that raspberry and your PC is connected to same network.



4.you can install yolo by making ssh or by using GUI terminal of pi(while installing yolo by making ssh, if it resumes go to GUI terminal or go to that place were high internet speed).



5.INSIDE TERMINAL



&#x20;   a. **sudo apt update \&\& apt upgrade** 

&#x20;   b. **sudo apt install -y python3-picamera2 python3-opencv python3-venv**

&#x20;   c. **python3 -m venv \~/yolo\_env --system-site-packages**   

&#x20;      **source \~/yolo\_env/bin/activate**  (virtual environment to keep packages isolated from your Raspberry Pi's system Python.)

&#x20;   d.Install CPU-only PyTorch

&#x20;      **python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu**



&#x20;      Verify PyTorch



&#x20; python3 -c "import torch; print('PyTorch:', torch.\_\_version\_\_); print('CUDA:', torch.cuda.is\_available())"



&#x20;    e.Then install Ultralytics

&#x20;       **python3 -m pip install ultralytics**



&#x20;     **verify**

**python3 -c "from ultralytics import YOLO; print('YOLO OK')"**





&#x20;   **f.nano \~/person\_detect.py**



**paste this**

&#x20; 





**import cv2**

**import time**

**from picamera2 import Picamera2**

**from ultralytics import YOLO**



**# -----------------------------**

**# Load YOLOv8s model**

**# -----------------------------**

**model = YOLO("/home/scarface/yolov8s.pt")**



**# -----------------------------**

**# Initialize camera**

**# -----------------------------**

**picam2 = Picamera2()**



**config = picam2.create\_preview\_configuration(**

&#x20;   **main={**

&#x20;       **"size": (640, 480),**

&#x20;       **"format": "RGB888"**

&#x20;   **}**

**)**



**picam2.configure(config)**

**picam2.start()**



**time.sleep(2)**



**print("Camera started")**

**print("YOLO person detection started")**

**print("Press Q to quit")**



**# -----------------------------**

**# Variables**

**# -----------------------------**

**frame\_count = 0**

**results = None**



**# -----------------------------**

**# Main loop**

**# -----------------------------**

**while True:**



&#x20;   **# Capture camera frame**

&#x20;   **frame = picam2.capture\_array()**

&#x20;   **frame\_count += 1**



&#x20;   **# ---------------------------------**

&#x20;   **# Run YOLO every 2nd frame**

&#x20;   **# ---------------------------------**

&#x20;   **if frame\_count % 2 == 0:**

&#x20;       **results = model(**

&#x20;           **frame,**

&#x20;           **imgsz=320,**

&#x20;           **classes=\[0],      # Person only**

&#x20;           **conf=0.40,**

&#x20;           **verbose=False**

&#x20;       **)**



&#x20;   **# ---------------------------------**

&#x20;   **# Draw detections \& count persons in THIS frame**

&#x20;   **# ---------------------------------**

&#x20;   **num\_persons = 0**



&#x20;   **if results is not None:**

&#x20;       **result = results\[0]**



&#x20;       **for box in result.boxes:**

&#x20;           **# Bounding box**

&#x20;           **x1, y1, x2, y2 = (**

&#x20;               **box.xyxy\[0]**

&#x20;               **.cpu()**

&#x20;               **.numpy()**

&#x20;               **.astype(int)**

&#x20;           **)**



&#x20;           **# Confidence**

&#x20;           **confidence = float(box.conf\[0])**



&#x20;           **# Label**

&#x20;           **label = f"Person {confidence:.2f}"**



&#x20;           **# Rectangle**

&#x20;           **cv2.rectangle(**

&#x20;               **frame,**

&#x20;               **(x1, y1),**

&#x20;               **(x2, y2),**

&#x20;               **(0, 255, 0),**

&#x20;               **2**

&#x20;           **)**



&#x20;           **# Confidence text**

&#x20;           **cv2.putText(**

&#x20;               **frame,**

&#x20;               **label,**

&#x20;               **(x1, max(y1 - 10, 20)),**

&#x20;               **cv2.FONT\_HERSHEY\_SIMPLEX,**

&#x20;               **0.6,**

&#x20;               **(0, 255, 0),**

&#x20;               **2**

&#x20;           **)**



&#x20;           **num\_persons += 1**



&#x20;   **# ---------------------------------**

&#x20;   **# Show PC count if > 0 (current frame only)**

&#x20;   **# ---------------------------------**

&#x20;   **if num\_persons > 0:**

&#x20;       **pc\_text = f"PC: {num\_persons}"**

&#x20;       **cv2.putText(**

&#x20;           **frame,**

&#x20;           **pc\_text,**

&#x20;           **(10, 30),**

&#x20;           **cv2.FONT\_HERSHEY\_SIMPLEX,**

&#x20;           **0.8,**

&#x20;           **(0, 255, 255),**

&#x20;           **2**

&#x20;       **)**



&#x20;   **# ---------------------------------**

&#x20;   **# Display**

&#x20;   **# ---------------------------------**

&#x20;   **cv2.imshow(**

&#x20;       **"YOLOv8s Person Detection",**

&#x20;       **frame**

&#x20;   **)**



&#x20;   **# Q = quit**

&#x20;   **if cv2.waitKey(1) \& 0xFF == ord("q"):**

&#x20;       **break**



**# -----------------------------**

**# Cleanup**

**# -----------------------------**

**picam2.stop()**

**cv2.destroyAllWindows()**



**print("Detection stopped")**

















**Ctrl + O → Enter → Ctrl + X**





**source \~/yolo\_env/bin/activate.fish**



**python3 \~/person\_detect.py**

