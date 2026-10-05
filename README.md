# Real-Time Person Detection on Raspberry Pi 5 (YOLOv8 + Pi Camera)

Detect and count people in real time using a **Raspberry Pi 5**, a **Pi Camera**, and **YOLOv8s** (Ultralytics). Detected people are shown with green boxes and confidence scores, and a live person count is shown on screen.

## Features

- Live camera feed using `Picamera2`
- Person-only detection (COCO class `0`)
- Runs on CPU (no GPU needed)
- Runs YOLO on every 2nd frame at 320 px to keep the frame rate smooth
- Live on-screen person count (`PC: N`)
- Press **Q** to quit

## Requirements

| Item | Details |
|------|---------|
| Board | Raspberry Pi 5 |
| Camera | Raspberry Pi Camera Module (connected and enabled) |
| OS | Raspberry Pi OS (Bookworm or newer, 64-bit recommended) |
| Display | Monitor, or VNC / xrdp remote desktop (the preview window needs a desktop session) |
| Internet | Needed during install (PyTorch and Ultralytics are large downloads) |

## Step 1: Connect to your Pi

1. Make sure the Pi and your PC are on the **same network**.
2. Enable remote access on the Pi: **SSH**, **VNC**, or install **xrdp**.
3. *(Optional)* Install the `fish` shell for easier typing of commands.

> **Tip:** If the install over SSH gets interrupted, continue from the Pi's own desktop terminal, or move to a place with faster internet.

## Step 2: Install dependencies

Run these in the Pi terminal.

**1. Update the system and install system packages**

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-picamera2 python3-opencv python3-venv
```

**2. Create and activate a virtual environment**

`--system-site-packages` lets the venv use the system `picamera2` and `opencv`.

```bash
python3 -m venv ~/yolo_env --system-site-packages
source ~/yolo_env/bin/activate
```

> Using fish? Activate with `source ~/yolo_env/bin/activate.fish`

**3. Install CPU-only PyTorch**

```bash
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify:

```bash
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

(`CUDA: False` is expected on a Pi.)

**4. Install Ultralytics**

```bash
python3 -m pip install ultralytics
```

Verify:

```bash
python3 -c "from ultralytics import YOLO; print('YOLO OK')"
```

## Step 3: Create the detection script

Create the file:

```bash
nano ~/person_detect.py
```

Paste the code below, then save with **Ctrl+O → Enter → Ctrl+X**.

```python
import cv2
import time
from picamera2 import Picamera2
from ultralytics import YOLO

# -----------------------------
# Load YOLOv8s model
# (downloads automatically on first run if not found)
# -----------------------------
model = YOLO("yolov8s.pt")

# -----------------------------
# Initialize camera
# -----------------------------
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (640, 480), "format": "RGB888"}
)

picam2.configure(config)
picam2.start()
time.sleep(2)  # let the camera warm up

print("Camera started")
print("YOLO person detection started")
print("Press Q to quit")

# -----------------------------
# Variables
# -----------------------------
frame_count = 0
results = None

# -----------------------------
# Main loop
# -----------------------------
while True:
    # Capture camera frame
    frame = picam2.capture_array()
    frame_count += 1

    # Run YOLO on every 2nd frame (saves CPU)
    if frame_count % 2 == 0:
        results = model(
            frame,
            imgsz=320,       # smaller image = faster
            classes=[0],     # person only
            conf=0.40,       # minimum confidence
            verbose=False,
        )

    # Draw detections and count persons
    num_persons = 0

    if results is not None:
        result = results[0]

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0])
            label = f"Person {confidence:.2f}"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Label
            cv2.putText(
                frame, label, (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

            num_persons += 1

    # Show person count (PC = Person Count)
    if num_persons > 0:
        cv2.putText(
            frame, f"PC: {num_persons}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
        )

    # Display
    cv2.imshow("YOLOv8s Person Detection", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# -----------------------------
# Cleanup
# -----------------------------
picam2.stop()
cv2.destroyAllWindows()
print("Detection stopped")
```

## Step 4: Run it

```bash
source ~/yolo_env/bin/activate        # fish: source ~/yolo_env/bin/activate.fish
python3 ~/person_detect.py
```

A window opens with the live camera feed. Press **Q** to quit.

## Tuning for speed or accuracy

| Setting | Where | Effect |
|---------|-------|--------|
| `imgsz=320` | `model(...)` call | Higher (e.g. `640`) = more accurate, slower |
| `conf=0.40` | `model(...)` call | Higher = fewer false positives, may miss people |
| `frame_count % 2` | main loop | Use `% 3` to run YOLO less often (faster) or `% 1` for every frame |
| `yolov8s.pt` | model load | Use `yolov8n.pt` for a faster, lighter model on the Pi |

## Troubleshooting

- **`ModuleNotFoundError: picamera2`**: the venv must be created with `--system-site-packages`, and `python3-picamera2` must be installed via `apt`.
- **Window doesn't open / Qt or display error**: run from the Pi desktop or a VNC/xrdp session (not plain SSH), or use `ssh -X`.
- **Camera not detected**: check the ribbon cable, then test with `rpicam-hello`.
- **Slow frame rate**: switch to `yolov8n.pt`, lower `imgsz`, or run YOLO on fewer frames.
- **Install keeps failing over SSH**: finish the install from the Pi's desktop terminal on a faster connection.

## Built with

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Picamera2](https://github.com/raspberrypi/picamera2)
- [OpenCV](https://opencv.org/)
- [PyTorch](https://pytorch.org/)
