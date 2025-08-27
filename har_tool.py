# har_tool.py
import os
import cv2
import numpy as np
import torch
from collections import deque

from ultralytics import YOLO

# --------------- Display setup (back to normal window) ------------------
WINDOW_NAME = "YOLO + HAR"
FULLSCREEN = False           # keep False to match your earlier windowed size
SHOW_OBSTACLE_BANNER = True

# If you want a fixed window size, set these (else OpenCV picks camera size)
TARGET_WINDOW_W, TARGET_WINDOW_H = 1280, 720

# --------------- YOLO settings (CPU-friendly) ---------------------------
MODEL_PATH = "yolov8n.pt"    # small & fast; switch to yolov8s.pt if you have GPU
DEVICE = "cpu"               # "cuda" if you have a working CUDA setup
IMG_SIZE = 416               # try 320–512 (higher = slower but sometimes more stable)
CONF_THRES = 0.35
IOU_THRES = 0.45
MAX_DET = 10000
PERSON_CLASS_ID = 0

# Classes that suggest "studying" when near a sitting person
STUDY_CUE_LABELS = {"laptop", "book", "cell phone", "keyboard", "mouse"}
# Objects you may want to warn about when they’re close
OBSTACLE_LABELS = {
    "chair", "couch", "bed", "bench",
    "dining table", "tv", "laptop",
    "bottle", "cup", "backpack", "handbag",
    "suitcase", "refrigerator", "microwave",
    "oven", "toaster", "sink"
}

# --------------- Optional TFLite HAR (used only if file exists) ---------
TFLITE_PATH = "har_model.tflite"
USE_TFLITE = os.path.exists(TFLITE_PATH)
har = None
if USE_TFLITE:
    # Import TensorFlow only if the file is present (prevents the warning spam)
    import tensorflow as tf
    har = tf.lite.Interpreter(model_path=TFLITE_PATH)
    har.allocate_tensors()
    har_in = har.get_input_details()[0]["index"]
    har_out = har.get_output_details()[0]["index"]

# If your TFLite model uses different labels, change here:
HAR_LABELS = ["Walking", "Running", "Sitting", "Standing", "Unknown"]

# Heuristic thresholds used when TFLite is absent
MOTION_THR = 18.0     # mean abs diff (0–255) inside person box
ASPECT_STAND = 1.6    # h/w > this → tall → standing-ish

# --------------- Helpers ------------------------------------------------
def put_text(img, text, org, color=(0, 255, 0), scale=0.9, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0: return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter + 1e-6)

def close_obstacle(box, frame_shape, area_ratio=0.08):
    x1, y1, x2, y2 = box
    H, W = frame_shape[:2]
    barea = max(0, (x2 - x1)) * max(0, (y2 - y1))
    return (barea / float(W * H + 1e-6)) > area_ratio

def har_tflite_on_crop(bgr):
    if bgr.size == 0: return "Unknown"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
    rgb = np.expand_dims(rgb, 0)
    har.set_tensor(har_in, rgb)
    har.invoke()
    probs = har.get_tensor(har_out)[0]
    idx = int(np.argmax(probs))
    return HAR_LABELS[idx] if 0 <= idx < len(HAR_LABELS) else "Unknown"

def har_heuristic(bgr, prev_gray_crop, motion_thr=MOTION_THR):
    if bgr.size == 0:
        return "Unknown", None, 0.0
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    motion = 0.0
    if prev_gray_crop is not None:
        # same size for diff
        g2 = cv2.resize(gray, (prev_gray_crop.shape[1], prev_gray_crop.shape[0]))
        motion = float(np.mean(cv2.absdiff(prev_gray_crop, g2)))
    # aspect ratio: tall → standing, wide → sitting
    h, w = gray.shape[:2]
    ar = h / max(w, 1)
    if motion > motion_thr:
        act = "Walking" if ar >= ASPECT_STAND else "Moving"
    else:
        act = "Standing" if ar >= ASPECT_STAND else "Sitting"
    return act, gray, motion

# --------------- Very light tracker for stable multi-person labels ------
class Track:
    __slots__ = ("tid","box","age","misses","activity_hist","prev_gray")
    def __init__(self, tid, box):
        self.tid = tid
        self.box = box
        self.age = 0
        self.misses = 0
        self.activity_hist = deque(maxlen=8)
        self.prev_gray = None

    def update(self, box):
        self.box = box
        self.age += 1
        self.misses = 0

    def vote(self, default="Unknown"):
        if not self.activity_hist: return default
        # majority vote with last item bias
        vals, counts = np.unique(self.activity_hist, return_counts=True)
        return vals[np.argmax(counts)]

class IoUTracker:
    def __init__(self, iou_thr=0.5, max_misses=10):
        self.iou_thr = iou_thr
        self.max_misses = max_misses
        self.tracks = []
        self.next_id = 1

    def step(self, dets):
        # dets: list of (x1,y1,x2,y2)
        unmatched = set(range(len(dets)))
        # greedy match
        for tr in self.tracks:
            best_j, best_iou = -1, 0.0
            for j in list(unmatched):
                i = iou(tr.box, dets[j])
                if i > best_iou:
                    best_iou, best_j = i, j
            if best_j >= 0 and best_iou >= self.iou_thr:
                tr.update(dets[best_j])
                unmatched.discard(best_j)
            else:
                tr.misses += 1

        # new tracks
        for j in unmatched:
            self.tracks.append(Track(self.next_id, dets[j]))
            self.next_id += 1

        # drop old tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return self.tracks

# --------------- Init camera & model -----------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Ask the camera for a sensible size (keeps windowed look)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WINDOW_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_WINDOW_H)

yolo = YOLO(MODEL_PATH)
yolo.to(DEVICE)
torch.set_grad_enabled(False)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
if FULLSCREEN:
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
else:
    cv2.resizeWindow(WINDOW_NAME, TARGET_WINDOW_W, TARGET_WINDOW_H)

tracker = IoUTracker(iou_thr=0.45, max_misses=12)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # YOLO inference
    with torch.no_grad():
        res = yolo.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            max_det=MAX_DET,
            device=DEVICE,
            verbose=False
        )[0]

    names = yolo.names
    person_boxes = []
    other_boxes = []
    other_names = []

    obstacle_close = False

    if res.boxes is not None and len(res.boxes) > 0:
        xyxy = res.boxes.xyxy.cpu().numpy().astype(int)
        cls = res.boxes.cls.cpu().numpy().astype(int)
        conf = res.boxes.conf.cpu().numpy()

        for (x1,y1,x2,y2), c, s in zip(xyxy, cls, conf):
            label = names.get(c, str(c))
            if c == PERSON_CLASS_ID:
                person_boxes.append((x1,y1,x2,y2))
            else:
                other_boxes.append((x1,y1,x2,y2))
                other_names.append(label)
                if label in OBSTACLE_LABELS and SHOW_OBSTACLE_BANNER and close_obstacle((x1,y1,x2,y2), frame.shape):
                    obstacle_close = True

    # Update tracks with current persons
    tracks = tracker.step(person_boxes)

    # For study cues, precompute which objects sit inside each person
    def overlaps_study_cue(pbox):
        for (bx1,by1,bx2,by2), lbl in zip(other_boxes, other_names):
            if lbl in STUDY_CUE_LABELS and iou(pbox, (bx1,by1,bx2,by2)) > 0.10:
                return True
        return False

    # Draw persons with activity
    for t in tracks:
        x1,y1,x2,y2 = t.box
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(frame.shape[1]-1, x2); y2 = min(frame.shape[0]-1, y2)

        crop = frame[y1:y2, x1:x2]

        if USE_TFLITE:
            act = har_tflite_on_crop(crop)
        else:
            act, gray_now, _ = har_heuristic(crop, t.prev_gray)
            t.prev_gray = gray_now

        # If sitting & laptop/phone/book overlaps → call it "Studying"
        if act in ("Sitting",) and overlaps_study_cue((x1,y1,x2,y2)):
            act = "Studying"

        # smooth with history
        t.activity_hist.append(act)
        act_smoothed = t.vote(default=act)

        # draw
        color = (0,255,0)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 3)
        put_text(frame, f"Person #{t.tid} | {act_smoothed}", (x1, max(30, y1-10)), color, 0.9, 2)

    # Draw other objects (thin boxes)
    for (bx1,by1,bx2,by2), lbl in zip(other_boxes, other_names):
        color = (0,0,255) if lbl in OBSTACLE_LABELS else (0,255,0)
        cv2.rectangle(frame, (bx1,by1), (bx2,by2), color, 2)
        put_text(frame, lbl, (bx1, max(20, by1-6)), color, 0.7, 2)

    # Obstacle banner
    if obstacle_close and SHOW_OBSTACLE_BANNER:
        H, W = frame.shape[:2]
        cv2.rectangle(frame, (0,0), (W, 40), (0,0,255), -1)
        put_text(frame, "⚠ OBSTACLE CLOSE — proceed carefully", (10, 28), (255,255,255), 0.9, 2)

    cv2.imshow(WINDOW_NAME, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
