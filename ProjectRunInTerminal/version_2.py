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
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from PIL import Image

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ================= Configuration =================
MODEL_PATH = "/Users/dong/Desktop/rock_paper_scissors/team2/手部检测/task/hand_landmarker.task"
TEMPLATE_FILE = "gesture_templates.json"
CSV_FILE = "rehab_records.csv"
HISTORY_PNG = "completion_history.png"

# Supported gestures
GESTURES = ["open", "fist", "pinch"]
DEFAULT_GESTURE = "open"

# Keys mapping
GESTURE_KEYS = {
    ord('1'): "open",
    ord('2'): "fist",
    ord('3'): "pinch",
}

FINGERTIPS = [4, 8, 12, 16, 20]
PALM_POINTS = [0, 5, 9, 13, 17]

# 'pinch' scoring will rely more on thumb + index
PINCH_CRITICAL_POINTS = [4, 8]

SHAPE_WEIGHT = 0.7
OPENNESS_WEIGHT = 0.3
DEVIATION_GREEN_THRESHOLD = 0.05

PREP_SECONDS = 3
CAPTURE_SECONDS = 10

GENERATE_PER_SESSION_PDF = True
MASTER_PDF_NAME = "rehab_report_all_grouped.pdf"
THUMB_PER_PAGE = 3

history = []
session_records = []

template_message = ""
template_msg_time = 0
TEMPLATE_FEEDBACK_SECONDS = 2.5

# ================ Template Management =================
def load_templates():
    if not os.path.exists(TEMPLATE_FILE):
        with open(TEMPLATE_FILE, "w") as f:
            # initialize empty templates for all gestures
            json.dump({g: {} for g in GESTURES}, f, indent=2)
        return {g: {} for g in GESTURES}
    with open(TEMPLATE_FILE, "r") as f:
        data = json.load(f)
    # Ensure all gestures present
    changed = False
    for g in GESTURES:
        if g not in data:
            data[g] = {}
            changed = True
    if changed:
        with open(TEMPLATE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    return data

def save_templates(templates):
    with open(TEMPLATE_FILE, "w") as f:
        json.dump(templates, f, indent=2)

gesture_templates = load_templates()

def has_template(name):
    return name in gesture_templates and len(gesture_templates[name]) > 0

# ================ Geometry & Normalization =================
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
    return {
        i: ((lm.x - cx) / w_ref, (lm.y - cy) / w_ref)
        for i, lm in enumerate(lm_list)
    }

def openness_ratio(lm_list):
    """
    For 'fist' we expect low openness; adjust openness scaling so that
    'fist' does not get unfair low total score. We'll compute normal openness
    but later adapt weighting per gesture.
    """
    cx, cy, w_ref = palm_center_and_width(lm_list)
    dists = []
    for tip in FINGERTIPS:
        d = math.dist((lm_list[tip].x, lm_list[tip].y), (cx, cy)) / w_ref
        dists.append(d)
    avg = sum(dists) / len(dists)
    return min(1.0, avg / 0.9)

# ================ Scoring =================
def compute_deviations_from_template(norm_landmarks, template_dict):
    devs = {}
    for k, (tx, ty) in template_dict.items():
        idx = int(k)
        if idx in norm_landmarks:
            x, y = norm_landmarks[idx]
            devs[idx] = math.sqrt((x - tx) ** 2 + (y - ty) ** 2)
    return devs

def finger_weight(idx, gesture):
    # Different gesture importance:
    if gesture == "open":
        return 1.2 if idx == 4 else 1.0
    if gesture == "fist":
        # Fist: emphasize knuckle closure - keep all equal or slightly higher on index & middle
        return 1.1 if idx in (8, 12) else 1.0
    if gesture == "pinch":
        # Pinch: strongly emphasize thumb & index
        if idx in (4, 8):
            return 1.5
        else:
            return 0.5
    return 1.0

def shape_score_from_devs(devs, gesture):
    if not devs:
        return 0.0
    sims = []
    for idx, d in devs.items():
        sim = math.exp(-8 * d)
        sim *= finger_weight(idx, gesture)
        sims.append(sim)
    return (sum(sims) / len(sims)) * 100.0

def compute_gesture_specific_total(gesture, shape_part, open_part, lm_list, devs):
    """
    Adjust combination logic per gesture:
     - open: shape + openness (default weights)
     - fist: reward *low openness* (so invert openness ratio)
     - pinch: shape part from thumb/index; add pinch distance closeness
    """
    if gesture == "open":
        return SHAPE_WEIGHT * shape_part + OPENNESS_WEIGHT * open_part

    if gesture == "fist":
        # Expect small openness. Convert open_part (0..100) to "closure" = (100 - open_part)
        closure = 100 - open_part
        # Weighted blend: shape 60%, closure 40%
        return 0.6 * shape_part + 0.4 * closure

    if gesture == "pinch":
        # Pinch distance: distance between thumb tip (4) and index tip (8)
        pinch_score = 0.0
        if lm_list:
            pt4 = lm_list[4]
            pt8 = lm_list[8]
            dist = math.sqrt((pt4.x - pt8.x)**2 + (pt4.y - pt8.y)**2)
            # Normalize distance by palm width
            _, _, w_ref = palm_center_and_width(lm_list)
            norm_d = dist / w_ref
            # Ideal pinch distance ~ small (e.g., 0.15). Score: exp decay
            pinch_score = math.exp(-8 * abs(norm_d - 0.15)) * 100.0
        # Combine: shape(thumb/index emphasis) 50% + pinch distance 30% + openness moderate (prefer moderately low)
        mod_open = 100 - abs(open_part - 50)  # best around mid openness ~50%
        return 0.5 * shape_part + 0.3 * pinch_score + 0.2 * mod_open

    # fallback
    return SHAPE_WEIGHT * shape_part + OPENNESS_WEIGHT * open_part

# ================ History / Reports =================
def plot_history():
    if len(history) < 2:
        return
    sessions = list(range(1, len(history) + 1))
    plt.figure()
    plt.plot(sessions, history, marker='o')
    plt.title("Rehab Completion History (All Gestures)")
    plt.xlabel("Session")
    plt.ylabel("Completion (%)")
    plt.ylim(0, 100)
    plt.grid(True)
    plt.savefig(HISTORY_PNG)
    plt.close()

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["Time", "SessionIndex", "Gesture", "Total", "Shape", "Open"])

def append_csv(ts, idx, gesture, total, shape, open_part):
    with open(CSV_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([ts, idx, gesture, f"{total:.2f}", f"{shape:.2f}", f"{open_part:.2f}"])

def generate_session_pdf(ts, session_idx, gesture, total_score, shape_part, open_part, best_frame, best_dev):
    pdf_name = f"rehab_report_{gesture}_{ts}.pdf"
    c = canvas.Canvas(pdf_name, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, f"Session Report - {gesture}")
    c.setFont("Helvetica", 12)
    c.drawString(50, 730, f"Time: {ts}")
    c.drawString(50, 710, f"Session: {session_idx}")
    c.drawString(50, 690, f"Gesture: {gesture}")
    c.drawString(50, 670, f"Total Completion: {total_score:.2f}%")
    c.drawString(50, 650, f"Shape Part: {shape_part:.2f}%")
    c.drawString(50, 630, f"Openness Part: {open_part:.2f}%")

    if best_frame is not None:
        rgb = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_reader = ImageReader(pil_img)
        c.drawImage(img_reader, 50, 390, width=220, height=220)

    if os.path.exists(HISTORY_PNG):
        c.drawString(300, 630, "Overall History:")
        c.drawImage(HISTORY_PNG, 300, 390, width=220, height=220)

    y = 370
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Per-Finger Deviations:")
    y -= 20
    c.setFont("Helvetica", 11)
    for idx, d in best_dev.items():
        c.drawString(50, y, f"Idx {idx}: {d:.4f}")
        y -= 15
        if y < 60:
            c.showPage()
            y = 750
            c.setFont("Helvetica", 11)

    c.save()
    return pdf_name

def generate_master_pdf_grouped(records):
    """
    Group sessions by gesture. For each gesture create:
      - Summary table
      - Thumbnails pages
    Include a global cover page with counts.
    """
    if not records:
        return
    c = canvas.Canvas(MASTER_PDF_NAME, pagesize=letter)
    width, height = letter

    # -------- Cover Page --------
    c.setFont("Helvetica-Bold", 22)
    c.drawString(60, height - 60, "Rehabilitation Master Report (Grouped)")
    c.setFont("Helvetica", 12)
    c.drawString(60, height - 90, f"Total Sessions: {len(records)}")
    c.drawString(60, height - 110, f"Last Update: {records[-1]['ts']}")
    # Gesture counts
    counts = {g: 0 for g in GESTURES}
    for r in records:
        counts[r["gesture"]] = counts.get(r["gesture"], 0) + 1
    y_counts = height - 140
    for g in GESTURES:
        c.drawString(60, y_counts, f"{g}: {counts[g]} sessions")
        y_counts -= 18

    if os.path.exists(HISTORY_PNG):
        c.drawImage(HISTORY_PNG, width - 260, height - 320, width=200, height=200)

    c.showPage()

    # -------- Per-gesture sections --------
    for gesture in GESTURES:
        g_records = [r for r in records if r["gesture"] == gesture]
        if not g_records:
            continue

        # Summary table page
        c.setFont("Helvetica-Bold", 18)
        c.drawString(50, height - 60, f"Gesture: {gesture} (Sessions={len(g_records)})")
        headers = ["#", "Time", "Total", "Shape", "Open"]
        col_x = [50, 110, 220, 300, 370]
        c.setFont("Helvetica-Bold", 11)
        y = height - 90
        for hx, text in zip(col_x, headers):
            c.drawString(hx, y, text)
        c.line(50, y - 2, 500, y - 2)
        y -= 20
        c.setFont("Helvetica", 10)
        for rec in g_records:
            c.drawString(col_x[0], y, str(rec["session_idx"]))
            c.drawString(col_x[1], y, rec["ts"][-6:])
            c.drawString(col_x[2], y, f"{rec['total']:.1f}")
            c.drawString(col_x[3], y, f"{rec['shape']:.1f}")
            c.drawString(col_x[4], y, f"{rec['open']:.1f}")
            y -= 14
            if y < 70:
                c.showPage()
                c.setFont("Helvetica-Bold", 11)
                y = height - 60
                for hx, text in zip(col_x, headers):
                    c.drawString(hx, y, text)
                c.line(50, y - 2, 500, y - 2)
                y -= 20
                c.setFont("Helvetica", 10)
        c.showPage()

        # Thumbnails
        thumb_w = 480
        thumb_h = 170
        margin_x = 50
        start_top = height - 70
        count_in_page = 0
        current_y = start_top

        def draw_thumb(rec, y_top):
            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin_x, y_top, f"Session {rec['session_idx']}  Total:{rec['total']:.1f}% Shape:{rec['shape']:.1f}% Open:{rec['open']:.1f}%")
            if rec["frame"] is not None:
                rgb = cv2.cvtColor(rec["frame"], cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                img_reader = ImageReader(pil_img)
                c.drawImage(img_reader, margin_x, y_top - thumb_h + 10, width=thumb_w, height=thumb_h - 30)
            # deviations
            devs = rec["dev"]
            y_text = y_top - thumb_h + 10
            c.setFont("Helvetica", 8)
            dev_line = 0
            for idx, d in devs.items():
                c.drawString(margin_x + thumb_w + 10, y_text + dev_line * 10, f"Idx{idx}:{d:.3f}")
                dev_line += 1

        for rec in g_records:
            if count_in_page == THUMB_PER_PAGE:
                c.showPage()
                count_in_page = 0
                current_y = start_top
            draw_thumb(rec, current_y)
            current_y -= (thumb_h + 25)
            count_in_page += 1
        c.showPage()

    c.save()
    print(f"[INFO] 分组汇总报告已更新: {MASTER_PDF_NAME}")

# ================ Landmarker Init =================
def init_landmarker():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"缺少模型文件 {MODEL_PATH}. 请先执行:\n"
            "curl -L -o hand_landmarker.task \\\n"
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

# ================ UI Helpers =================
def overlay_template_feedback(frame):
    if template_message and (time.time() - template_msg_time) < TEMPLATE_FEEDBACK_SECONDS:
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + 560, 10 + 40), (0, 180, 255), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
        cv2.putText(frame, template_message, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    return frame

def draw_gesture_shortcuts(frame):
    cv2.putText(frame, "Gestures: [1]=open  [2]=fist  [3]=pinch  [t]=capture template  ESC=quit",
                (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180,180,180), 1)

# ================ Main Loop =================
def main():
    global template_message, template_msg_time
    gesture = DEFAULT_GESTURE
    landmarker = init_landmarker()
    init_csv()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    session_idx = 0
    while True:
        session_idx += 1
        # -------- Preparation --------
        prep_start = time.time()
        while time.time() - prep_start < PREP_SECONDS:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            remain = PREP_SECONDS - int(time.time() - prep_start)
            cv2.putText(frame, f"Session {session_idx} - Ready: {remain}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
            cv2.putText(frame, f"Gesture: {gesture}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200,200,255), 2)
            draw_gesture_shortcuts(frame)
            frame = overlay_template_feedback(frame)
            cv2.imshow("Hand Rehab Multi", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cap.release(); cv2.destroyAllWindows(); return
            elif k in GESTURE_KEYS:
                gesture = GESTURE_KEYS[k]

        # -------- Capture Phase --------
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
            lm_list = None

            if result.hand_landmarks:
                lm_list = result.hand_landmarks[0]

                key_now = cv2.waitKey(1) & 0xFF
                if key_now == ord('t'):
                    norm_capture = normalize_landmarks_palm(lm_list)
                    # For pinch gesture, we still store all finger tips; weighting adjusts scoring
                    new_template = {str(i): norm_capture[i] for i in FINGERTIPS if i in norm_capture}
                    gesture_templates[gesture] = new_template
                    save_templates(gesture_templates)
                    template_message = f"[模板已更新] {gesture} 模板采集成功"
                    template_msg_time = time.time()
                    print(template_message)

                norm = normalize_landmarks_palm(lm_list)

                if has_template(gesture):
                    template = gesture_templates[gesture]
                    devs = compute_deviations_from_template(norm, template)
                    shape_part = shape_score_from_devs(devs, gesture)

                open_part = openness_ratio(lm_list) * 100.0
                total_score = compute_gesture_specific_total(gesture, shape_part, open_part, lm_list, devs)

                if total_score > best_total:
                    best_total = total_score
                    best_frame = frame.copy()
                    best_dev = devs.copy()
                    best_shape = shape_part
                    best_open = open_part

                # Draw skeleton and info
                if lm_list:
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
                        cv2.circle(frame,(cx_,cy_),4,(0,165,255),-1)

                # Deviations list
                y0 = 170
                for tip in FINGERTIPS:
                    if tip in devs:
                        d = devs[tip]
                        color = (0,255,0) if d < DEVIATION_GREEN_THRESHOLD else (0,0,255)
                        cv2.putText(frame, f"Idx{tip}:{d:.3f}", (20,y0),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        y0 += 18

                cv2.putText(frame, f"G:{gesture}", (20, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.putText(frame, f"Shape:{shape_part:.1f}%", (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 2)
                cv2.putText(frame, f"Open:{open_part:.1f}%", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,180), 2)
                cv2.putText(frame, f"Best:{best_total:.1f}%", (20, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

            remain = CAPTURE_SECONDS - int(time.time() - capture_start)
            cv2.putText(frame, f"Capturing ({remain}s)", (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
            draw_gesture_shortcuts(frame)
            frame = overlay_template_feedback(frame)
            cv2.imshow("Hand Rehab Multi", frame)
            k2 = cv2.waitKey(1) & 0xFF
            if k2 == 27:
                cap.release(); cv2.destroyAllWindows(); return
            elif k2 in GESTURE_KEYS:
                gesture = GESTURE_KEYS[k2]

        # -------- Result Phase --------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        history.append(best_total)
        append_csv(ts, session_idx, gesture, best_total, best_shape, best_open)
        plot_history()

        session_records.append({
            "session_idx": session_idx,
            "ts": ts,
            "gesture": gesture,
            "total": best_total,
            "shape": best_shape,
            "open": best_open,
            "dev": best_dev,
            "frame": best_frame
        })

        if GENERATE_PER_SESSION_PDF:
            generate_session_pdf(ts, session_idx, gesture, best_total, best_shape, best_open, best_frame, best_dev)

        generate_master_pdf_grouped(session_records)

        display = best_frame if best_frame is not None else np.zeros((480,640,3),dtype=np.uint8)
        display = display.copy()
        cv2.putText(display, f"{gesture} Session Complete!", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 3)
        cv2.putText(display, f"Total:{best_total:.1f}% (Shape {best_shape:.1f}%  Open {best_open:.1f}%)",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,100), 2)

        y0 = 150
        for tip in FINGERTIPS:
            if tip in best_dev:
                d = best_dev[tip]
                color = (0,255,0) if d < DEVIATION_GREEN_THRESHOLD else (0,0,255)
                cv2.putText(display, f"Idx{tip}:{d:.3f}", (20,y0),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                y0 += 18

        cv2.putText(display, "Enter=Next  ESC=Quit  [1]=open [2]=fist [3]=pinch [t]=capture",
                    (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        display = overlay_template_feedback(display)
        cv2.imshow("Hand Rehab Multi", display)

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
