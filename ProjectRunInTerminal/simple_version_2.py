#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand Rehab Minimal Multi-Gesture Version
Dependencies: python 3.9+, opencv-python, mediapipe, numpy
Install: pip install --upgrade mediapipe opencv-python numpy
If downloading the task model is hard, set USE_OLD_API = True (will use legacy solutions.Hands).
"""

import os, cv2, time, math, csv, json
from datetime import datetime
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# -------- Configuration --------
MODEL_PATH = "/Users/dong/Desktop/rock_paper_scissors/team2/手部检测/task/hand_landmarker.task"
TEMPLATE_FILE = "gesture_templates.json"
CSV_FILE = "rehab_records.csv"

GESTURES = ["open", "fist", "pinch"]
GESTURE_KEYS = {ord('1'): "open", ord('2'): "fist", ord('3'): "pinch"}
DEFAULT_GESTURE = "open"

FINGERTIPS = [4, 8, 12, 16, 20]
PALM_POINTS = [0, 5, 9, 13, 17]

SHAPE_WEIGHT = 0.7
OPENNESS_WEIGHT = 0.3
DEVIATION_GREEN_THRESHOLD = 0.05

PREP_SECONDS = 2
CAPTURE_SECONDS = 8
TEMPLATE_FEEDBACK_SECONDS = 2.0

# If True uses legacy solutions.Hands (no model .task file needed)
USE_OLD_API = False

# -------- Template Management --------
def load_templates():
    if not os.path.exists(TEMPLATE_FILE):
        with open(TEMPLATE_FILE, "w") as f:
            json.dump({g: {} for g in GESTURES}, f, indent=2)
        return {g: {} for g in GESTURES}
    with open(TEMPLATE_FILE, "r") as f:
        data = json.load(f)
    changed = False
    for g in GESTURES:
        if g not in data:
            data[g] = {}; changed = True
    if changed:
        with open(TEMPLATE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    return data

def save_templates(tpl):
    with open(TEMPLATE_FILE, "w") as f:
        json.dump(tpl, f, indent=2)

gesture_templates = load_templates()

def has_template(name):
    return name in gesture_templates and len(gesture_templates[name]) > 0

# -------- Geometry / Normalization --------
def palm_center_and_width(lm_list):
    cx = sum(lm_list[i].x for i in PALM_POINTS) / len(PALM_POINTS)
    cy = sum(lm_list[i].y for i in PALM_POINTS) / len(PALM_POINTS)
    w_ref = math.dist((lm_list[5].x, lm_list[5].y),
                      (lm_list[17].x, lm_list[17].y)) or 1e-6
    return cx, cy, w_ref

def normalize_landmarks(lm_list):
    cx, cy, w_ref = palm_center_and_width(lm_list)
    return {i: ((lm.x - cx)/w_ref, (lm.y - cy)/w_ref) for i, lm in enumerate(lm_list)}

def openness_ratio(lm_list):
    cx, cy, w_ref = palm_center_and_width(lm_list)
    ds = [math.dist((lm_list[t].x, lm_list[t].y), (cx, cy))/w_ref for t in FINGERTIPS]
    avg = sum(ds)/len(ds)
    return min(1.0, avg / 0.9)

# -------- Scoring --------
def compute_devs(norm, template_dict):
    devs = {}
    for k, (tx, ty) in template_dict.items():
        idx = int(k)
        if idx in norm:
            x, y = norm[idx]
            devs[idx] = math.hypot(x - tx, y - ty)
    return devs

def finger_weight(idx, gesture):
    if gesture == "open":
        return 1.2 if idx == 4 else 1.0
    if gesture == "fist":
        return 1.1 if idx in (8, 12) else 1.0
    if gesture == "pinch":
        return 1.5 if idx in (4, 8) else 0.5
    return 1.0

def shape_score(devs, gesture):
    if not devs: return 0.0
    sims = []
    for idx, d in devs.items():
        s = math.exp(-8*d) * finger_weight(idx, gesture)
        sims.append(s)
    return (sum(sims)/len(sims))*100.0

def gesture_total(gesture, shape_part, open_part, lm_list):
    if gesture == "open":
        return SHAPE_WEIGHT*shape_part + OPENNESS_WEIGHT*open_part
    if gesture == "fist":
        closure = 100 - open_part  # Lower openness => higher closure
        return 0.6*shape_part + 0.4*closure
    if gesture == "pinch":
        pinch_score = 0.0
        if lm_list:
            pt4, pt8 = lm_list[4], lm_list[8]
            dist = math.hypot(pt4.x - pt8.x, pt4.y - pt8.y)
            _, _, w_ref = palm_center_and_width(lm_list)
            nd = dist / w_ref
            pinch_score = math.exp(-8*abs(nd - 0.15))*100.0  # target normalized distance
        mid_pref = 100 - abs(open_part - 50)  # prefer moderate openness
        return 0.5*shape_part + 0.3*pinch_score + 0.2*mid_pref
    return SHAPE_WEIGHT*shape_part + OPENNESS_WEIGHT*open_part

# -------- CSV --------
def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Time","Session","Gesture","Total","Shape","Open"])

def append_csv(ts, sess, gesture, total, shape, open_part):
    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ts, sess, gesture,
                                f"{total:.2f}", f"{shape:.2f}", f"{open_part:.2f}"])

# -------- Landmarker Init --------
def create_landmarker_tasks():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "缺少 hand_landmarker.task; 若无法下载可将 USE_OLD_API=True。\n"
            "下载命令:\n"
            "curl -L -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
    base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=base,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(opts)

def create_old_hands():
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(static_image_mode=False,
                          max_num_hands=1,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)

# -------- Main --------
def main():
    init_csv()
    gesture = DEFAULT_GESTURE
    hands = None
    landmarker = None
    if USE_OLD_API:
        hands = create_old_hands()
    else:
        landmarker = create_landmarker_tasks()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    session = 0
    template_msg = ""
    template_time = 0

    while True:
        session += 1
        # ---- Preparation Phase ----
        prep_start = time.time()
        while time.time() - prep_start < PREP_SECONDS:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            remain = PREP_SECONDS - int(time.time() - prep_start)
            cv2.putText(frame, f"Session {session} Ready:{remain}", (20,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(frame, f"Gesture:{gesture}", (20,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,255), 2)
            cv2.putText(frame, "[1]open [2]fist [3]pinch  [t]模板采集  ESC退出",
                        (20,430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180),1)
            if template_msg and time.time()-template_time < TEMPLATE_FEEDBACK_SECONDS:
                cv2.rectangle(frame,(10,10),(470,40),(0,180,255),-1)
                cv2.putText(frame, template_msg, (20,34),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(30,30,30),2)
            cv2.imshow("Hand Rehab Minimal", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cap.release(); cv2.destroyAllWindows(); return
            elif k in GESTURE_KEYS:
                gesture = GESTURE_KEYS[k]

        # ---- Capture Phase ----
        best_total = 0.0
        best_shape = 0.0
        best_open = 0.0
        best_dev = {}
        best_frame = None

        start = time.time()
        while time.time() - start < CAPTURE_SECONDS:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            lm_list = None
            if USE_OLD_API:
                results = hands.process(rgb)
                if results.multi_hand_landmarks:
                    lm_list = results.multi_hand_landmarks[0].landmark
            else:
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)
                if result.hand_landmarks:
                    lm_list = result.hand_landmarks[0]

            shape_part = 0.0
            open_part = 0.0
            devs = {}

            if lm_list:
                # Template capture
                if cv2.waitKey(1) & 0xFF == ord('t'):
                    norm_cap = normalize_landmarks(lm_list)
                    gesture_templates[gesture] = {
                        str(i): norm_cap[i] for i in FINGERTIPS if i in norm_cap
                    }
                    save_templates(gesture_templates)
                    template_msg = f"[模板更新] {gesture}"
                    template_time = time.time()

                norm = normalize_landmarks(lm_list)
                if has_template(gesture):
                    devs = compute_devs(norm, gesture_templates[gesture])
                    shape_part = shape_score(devs, gesture)
                open_part = openness_ratio(lm_list) * 100.0
                total = gesture_total(gesture, shape_part, open_part, lm_list)

                if total > best_total:
                    best_total = total
                    best_shape = shape_part
                    best_open = open_part
                    best_dev = devs.copy()
                    best_frame = frame.copy()

                # Draw skeleton
                h, w, _ = frame.shape
                CONN = [
                    (0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (5,9),(9,10),(10,11),(11,12),
                    (9,13),(13,14),(14,15),(15,16),
                    (13,17),(17,18),(18,19),(19,20),(0,17)
                ]
                for a,b in CONN:
                    ax, ay = int(lm_list[a].x*w), int(lm_list[a].y*h)
                    bx, by = int(lm_list[b].x*w), int(lm_list[b].y*h)
                    cv2.line(frame,(ax,ay),(bx,by),(0,255,0),2)
                for lm in lm_list:
                    cx_, cy_ = int(lm.x*w), int(lm.y*h)
                    cv2.circle(frame,(cx_,cy_),3,(0,165,255),-1)

                # Deviation list
                y0 = 150
                for tip in FINGERTIPS:
                    if tip in devs:
                        d = devs[tip]
                        color = (0,255,0) if d < DEVIATION_GREEN_THRESHOLD else (0,0,255)
                        cv2.putText(frame, f"{tip}:{d:.3f}", (20,y0),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
                        y0 += 16

                cv2.putText(frame, f"Shape:{shape_part:.1f}% Open:{open_part:.1f}%",
                            (20,120), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,255),2)
                cv2.putText(frame, f"Best:{best_total:.1f}%", (20,95),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)

            remain = CAPTURE_SECONDS - int(time.time() - start)
            cv2.putText(frame, f"{gesture} {remain}s", (20,35),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
            cv2.putText(frame, "[1]open [2]fist [3]pinch [t]模板 ESC退",
                        (20,430), cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)
            if template_msg and time.time()-template_time < TEMPLATE_FEEDBACK_SECONDS:
                cv2.rectangle(frame,(10,10),(300,40),(0,180,255),-1)
                cv2.putText(frame, template_msg, (18,34),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(30,30,30),2)
            cv2.imshow("Hand Rehab Minimal", frame)
            k2 = cv2.waitKey(1) & 0xFF
            if k2 == 27:
                cap.release(); cv2.destroyAllWindows(); return
            elif k2 in GESTURE_KEYS:
                gesture = GESTURE_KEYS[k2]

        # ---- Result Summary ----
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        append_csv(ts, session, gesture, best_total, best_shape, best_open)

        summary = best_frame if best_frame is not None else np.zeros((480,640,3),dtype=np.uint8)
        summary = summary.copy()
        cv2.putText(summary, f"{gesture} Done!", (20,60),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
        cv2.putText(summary, f"Total:{best_total:.1f}%  (Shape {best_shape:.1f}%  Open {best_open:.1f}%)",
                    (20,105), cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,100),2)
        y0 = 140
        for tip in FINGERTIPS:
            if tip in best_dev:
                d = best_dev[tip]
                color = (0,255,0) if d < DEVIATION_GREEN_THRESHOLD else (0,0,255)
                cv2.putText(summary, f"{tip}:{d:.3f}", (20,y0),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
                y0 += 18
        cv2.putText(summary, "Enter=Next  ESC=Quit", (20,430),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1)
        cv2.imshow("Hand Rehab Minimal", summary)
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 13:  # Enter
                break
            if k == 27:
                cap.release(); cv2.destroyAllWindows(); return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
