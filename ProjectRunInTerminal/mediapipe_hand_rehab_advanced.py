import os
import cv2
import time
import math
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
TEMPLATE_FILE = "gesture_templates.json"
CSV_FILE = "rehab_records.csv"
HISTORY_PNG = "completion_history.png"

DEFAULT_GESTURE = "open"
FINGERTIPS = [4, 8, 12, 16, 20]
PALM_POINTS = [0, 5, 9, 13, 17]

SHAPE_WEIGHT = 0.7
OPENNESS_WEIGHT = 0.3
DEVIATION_GREEN_THRESHOLD = 0.05

PREP_SECONDS = 3
CAPTURE_SECONDS = 10

history = []

def load_templates():
    if not os.path.exists(TEMPLATE_FILE):
        with open(TEMPLATE_FILE, "w") as f:
            json.dump({"open": {}}, f, indent=2)
        return {"open": {}}
    with open(TEMPLATE_FILE, "r") as f:
        return json.load(f)

def save_templates(templates):
    with open(TEMPLATE_FILE, "w") as f:
        json.dump(templates, f, indent=2)

gesture_templates = load_templates()

def has_template(name):
    return name in gesture_templates and len(gesture_templates[name]) > 0

def palm_center_and_width(lm_list):
    cx = sum(lm_list[i].x for i in PALM_POINTS) / len(PALM_POINTS)
    cy = sum(lm_list[i].y for i in PALM_POINTS) / len(PALM_POINTS)
    w_ref = math.dist(
        (lm_list[5].x, lm_list[5].y),
        (lm_list[17].x, lm_list[17].y)
    ) or 1e-6
    return cx, cy, w_ref

def normalize_landmarks_palm(lm_list):
    cx, cy, w_ref = palm_center_and_width(lm_list)
    norm = {}
    for i, lm in enumerate(lm_list):
        nx = (lm.x - cx) / w_ref
        ny = (lm.y - cy) / w_ref
        norm[i] = (nx, ny)
    return norm

def openness_ratio(lm_list):
    cx, cy, w_ref = palm_center_and_width(lm_list)
    dists = []
    for tip in FINGERTIPS:
        d = math.dist((lm_list[tip].x, lm_list[tip].y), (cx, cy)) / w_ref
        dists.append(d)
    avg = sum(dists) / len(dists)
    return min(1.0, avg / 0.9)

def compute_deviations_from_template(norm_landmarks, template_dict):
    devs = {}
    for k, (tx, ty) in template_dict.items():
        idx = int(k)
        if idx in norm_landmarks:
            x, y = norm_landmarks[idx]
            devs[idx] = math.sqrt((x - tx) ** 2 + (y - ty) ** 2)
    return devs

def finger_weight(idx):
    return 1.2 if idx == 4 else 1.0

def shape_score_from_devs(devs):
    if not devs:
        return 0.0
    sims = []
    for idx, d in devs.items():
        sim = math.exp(-8 * d)
        sim *= finger_weight(idx)
        sims.append(sim)
    return (sum(sims) / len(sims)) * 100.0

def combined_score(shape_part, open_part):
    return SHAPE_WEIGHT * shape_part + OPENNESS_WEIGHT * open_part

def plot_history():
    if len(history) < 2:
        return
    sessions = list(range(1, len(history)+1))
    plt.figure()
    plt.plot(sessions, history, marker='o')
    plt.title("Rehab Completion History")
    plt.xlabel("Session")
    plt.ylabel("Completion (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.savefig(HISTORY_PNG)
    plt.close()

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["Time", "Gesture", "Completion", "Shape", "Openness"])

def append_csv(ts, gesture, total, shape, open_part):
    with open(CSV_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([ts, gesture, f"{total:.2f}", f"{shape:.2f}", f"{open_part:.2f}"])

def generate_pdf(ts, gesture, total_score, shape_part, open_part, best_frame, best_dev):
    pdf_name = f"rehab_report_{ts}.pdf"
    c = canvas.Canvas(pdf_name, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "Rehabilitation Training Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 730, f"Time: {ts}")
    c.drawString(50, 710, f"Gesture: {gesture}")
    c.drawString(50, 690, f"Total Completion: {total_score:.2f}%")
    c.drawString(50, 670, f"Shape Part: {shape_part:.2f}%")
    c.drawString(50, 650, f"Openness Part: {open_part:.2f}%")

    if best_frame is not None:
        img_name = f"screenshot_{ts}.png"
        cv2.imwrite(img_name, best_frame)
        c.drawString(50, 630, "Best Frame:")
        c.drawImage(img_name, 50, 400, width=200, height=200)

    if os.path.exists(HISTORY_PNG):
        c.drawString(300, 630, "Completion History:")
        c.drawImage(HISTORY_PNG, 300, 400, width=200, height=200)

    y = 380
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Per-Finger Deviations:")
    y -= 20
    c.setFont("Helvetica", 11)
    for idx, d in best_dev.items():
        c.drawString(50, y, f"Idx {idx}: {d:.4f}")
        y -= 15

    c.save()
    return pdf_name

def init_landmarker():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"缺少模型文件 {MODEL_PATH}. 请先执行:\n"
            "curl -L -o hand_landmarker.task \
"
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        )
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    return vision.HandLandmarker.create_from_options(options)

def main():
    gesture = DEFAULT_GESTURE
    landmarker = init_landmarker()
    init_csv()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    session = 0
    while True:
        session += 1
        prep_start = time.time()
        while time.time() - prep_start < PREP_SECONDS:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            remain = PREP_SECONDS - int(time.time() - prep_start)
            cv2.putText(frame, f"Session {session} - Get Ready: {remain}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
            cv2.putText(frame, f"Gesture: {gesture}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,255), 2)
            cv2.putText(frame, "Keys: [t] capture template  [1] open  ESC quit",
                        (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
            cv2.imshow("Hand Rehab Advanced", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cap.release(); cv2.destroyAllWindows(); return
            elif k == ord('1'):
                gesture = "open"

        best_total = 0.0
        best_frame = None
        best_dev = {}
        best_shape = 0.0
        best_open = 0.0

        capture_start = time.time()
        while time.time() - capture_start < CAPTURE_SECONDS:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            shape_part = 0.0
            open_part = 0.0
            devs = {}

            if result.hand_landmarks:
                lm_list = result.hand_landmarks[0]

                key_now = cv2.waitKey(1) & 0xFF
                if key_now == ord('t'):
                    norm_capture = normalize_landmarks_palm(lm_list)
                    new_template = {str(i): norm_capture[i] for i in FINGERTIPS if i in norm_capture}
                    gesture_templates[gesture] = new_template
                    save_templates(gesture_templates)
                    print(f"[INFO] 模板已更新 {gesture}: {new_template}")

                norm = normalize_landmarks_palm(lm_list)

                if has_template(gesture):
                    template = gesture_templates[gesture]
                    devs = compute_deviations_from_template(norm, template)
                    shape_part = shape_score_from_devs(devs)

                open_part = openness_ratio(lm_list) * 100.0
                total_score = combined_score(shape_part, open_part)

                if total_score > best_total:
                    best_total = total_score
                    best_frame = frame.copy()
                    best_dev = devs.copy()
                    best_shape = shape_part
                    best_open = open_part

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
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    cv2.circle(frame,(cx,cy),4,(0,165,255),-1)

                y0 = 160
                for tip in FINGERTIPS:
                    if tip in devs:
                        d = devs[tip]
                        color = (0,255,0) if d < DEVIATION_GREEN_THRESHOLD else (0,0,255)
                        cv2.putText(frame, f"Idx{tip}:{d:.3f}", (20,y0),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                        y0 += 22

                cv2.putText(frame, f"Shape:{shape_part:.1f}% Open:{open_part:.1f}%", (20, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2)
                cv2.putText(frame, f"Best:{best_total:.1f}%", (20, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

            remain = CAPTURE_SECONDS - int(time.time() - capture_start)
            cv2.putText(frame, f"Capturing ({remain}s)", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(frame, f"Gesture:{gesture}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,255), 2)

            cv2.imshow("Hand Rehab Advanced", frame)
            k2 = cv2.waitKey(1) & 0xFF
            if k2 == 27:
                cap.release(); cv2.destroyAllWindows(); return
            elif k2 == ord('1'):
                gesture = "open"

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        history.append(best_total)
        append_csv(ts, gesture, best_total, best_shape, best_open)
        plot_history()
        pdf_path = generate_pdf(ts, gesture, best_total, best_shape, best_open, best_frame, best_dev)
        print(f"[INFO] 报告生成: {pdf_path}")

        display = best_frame if best_frame is not None else np.zeros((480,640,3),dtype=np.uint8)
        display = display.copy()
        cv2.putText(display, "Session Complete!", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
        cv2.putText(display, f"Total:{best_total:.1f}%", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,100), 2)
        cv2.putText(display, f"(Shape {best_shape:.1f}%  Open {best_open:.1f}%)", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)

        y0 = 190
        for tip in FINGERTIPS:
            if tip in best_dev:
                d = best_dev[tip]
                color = (0,255,0) if d < DEVIATION_GREEN_THRESHOLD else (0,0,255)
                cv2.putText(display, f"Idx{tip}:{d:.3f}", (20,y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                y0 += 24

        cv2.putText(display, "Enter=Next  ESC=Quit  [1]=open  [t]=capture template", (20, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
        cv2.imshow("Hand Rehab Advanced", display)

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 13:
                break
            elif k == 27:
                cap.release(); cv2.destroyAllWindows(); return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
