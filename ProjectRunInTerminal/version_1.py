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
from PIL import Image  # For in-memory embedding

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ================= Configuration =================
MODEL_PATH = "/Users/dong/Desktop/rock_paper_scissors/team2/手部检测/task/hand_landmarker.task"
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

# Toggle whether to still produce per-session individual PDF reports
GENERATE_PER_SESSION_PDF = False
MASTER_PDF_NAME = "rehab_report_all.pdf"
THUMB_PER_PAGE = 3  # number of session images per page (excluding the first summary page)

history = []
session_records = []  # store all session data including frames

# Feedback state for template capture
template_message = ""
template_msg_time = 0
TEMPLATE_FEEDBACK_SECONDS = 2.5

# ================= Template Management =================
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

# ================= Geometry & Normalization =================
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
    cx, cy, w_ref = palm_center_and_width(lm_list)
    dists = []
    for tip in FINGERTIPS:
        d = math.dist((lm_list[tip].x, lm_list[tip].y), (cx, cy)) / w_ref
        dists.append(d)
    avg = sum(dists) / len(dists)
    return min(1.0, avg / 0.9)

# ================= Scoring =================
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

# ================= History / Reports =================
def plot_history():
    if len(history) < 2:
        return
    sessions = list(range(1, len(history) + 1))
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
            csv.writer(f).writerow(["Time", "SessionIndex", "Gesture", "Completion", "Shape", "Openness"])

def append_csv(ts, idx, gesture, total, shape, open_part):
    with open(CSV_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([ts, idx, gesture, f"{total:.2f}", f"{shape:.2f}", f"{open_part:.2f}"])

def generate_session_pdf(ts, session_idx, gesture, total_score, shape_part, open_part, best_frame, best_dev):
    pdf_name = f"rehab_report_{ts}.pdf"
    c = canvas.Canvas(pdf_name, pagesize=letter)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 750, "Rehabilitation Training Report (Single Session)")
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
        c.drawImage(img_reader, 50, 400, width=220, height=220)

    if os.path.exists(HISTORY_PNG):
        c.drawString(300, 630, "Completion History:")
        c.drawImage(HISTORY_PNG, 300, 400, width=220, height=220)

    y = 380
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

def generate_master_pdf(records):
    """
    Create/overwrite a cumulative PDF containing:
      Page 1: Summary table + optional history chart
      Subsequent pages: Session best frame thumbnails (THUMB_PER_PAGE per page)
    """
    if not records:
        return
    c = canvas.Canvas(MASTER_PDF_NAME, pagesize=letter)
    width, height = letter

    # --- Page 1 Summary ---
    c.setFont("Helvetica-Bold", 20)
    c.drawString(40, height - 60, "Rehabilitation Master Report")

    c.setFont("Helvetica", 12)
    c.drawString(40, height - 90, f"Total Sessions: {len(records)}")
    c.drawString(40, height - 110, f"Latest Update: {records[-1]['ts']}")

    # Table header
    start_y = height - 140
    c.setFont("Helvetica-Bold", 11)
    headers = ["#","Time","Gesture","Total%","Shape%","Open%"]
    col_x = [40, 100, 210, 300, 370, 440]
    for hx, text in zip(col_x, headers):
        c.drawString(hx, start_y, text)
    c.setLineWidth(0.5)
    c.line(40, start_y - 2, 520, start_y - 2)

    c.setFont("Helvetica", 10)
    row_y = start_y - 20
    for rec in records:
        c.drawString(col_x[0], row_y, str(rec["session_idx"]))
        c.drawString(col_x[1], row_y, rec["ts"][-6:])  # show hhmmss part
        c.drawString(col_x[2], row_y, rec["gesture"])
        c.drawString(col_x[3], row_y, f"{rec['total']:.1f}")
        c.drawString(col_x[4], row_y, f"{rec['shape']:.1f}")
        c.drawString(col_x[5], row_y, f"{rec['open']:.1f}")
        row_y -= 14
        if row_y < 100:
            # History image if exists, place at bottom right
            if os.path.exists(HISTORY_PNG):
                c.drawImage(HISTORY_PNG, 360, 120, width=180, height=180)
            c.showPage()
            c.setFont("Helvetica-Bold", 11)
            for hx, text in zip(col_x, headers):
                c.drawString(hx, height - 60, text)
            c.line(40, height - 62, 520, height - 62)
            c.setFont("Helvetica", 10)
            row_y = height - 80

    # If history chart exists and not already placed on last partial page
    if os.path.exists(HISTORY_PNG):
        if row_y > 200:  # enough space
            c.drawImage(HISTORY_PNG, 360, 120, width=180, height=180)
    c.showPage()

    # --- Subsequent Pages: Thumbnails ---
    # Each page grid: vertical stacking (THUMB_PER_PAGE)
    thumb_w = 500
    thumb_h = 180
    margin_x = 50
    start_top = height - 80

    def draw_session_thumb(rec, y_top):
        c.setFont("Helvetica-Bold", 12)
        c.drawString(margin_x, y_top, f"Session {rec['session_idx']}  ({rec['gesture']})  Total:{rec['total']:.1f}%  Shape:{rec['shape']:.1f}%  Open:{rec['open']:.1f}%")
        if rec["frame"] is not None:
            rgb = cv2.cvtColor(rec["frame"], cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            img_reader = ImageReader(pil_img)
            c.drawImage(img_reader, margin_x, y_top - thumb_h + 10, width=thumb_w, height=thumb_h - 30)
        # deviations table (inline small)
        devs = rec["dev"]
        y_text = y_top - thumb_h + 10
        c.setFont("Helvetica", 8)
        dev_line = 0
        for idx, d in devs.items():
            c.drawString(margin_x + thumb_w + 10, y_text + dev_line * 10, f"Idx{idx}:{d:.3f}")
            dev_line += 1

    count_in_page = 0
    current_y = start_top
    for rec in records:
        if count_in_page == THUMB_PER_PAGE:
            c.showPage()
            count_in_page = 0
            current_y = start_top
        draw_session_thumb(rec, current_y)
        current_y -= (thumb_h + 20)
        count_in_page += 1

    c.save()
    print(f"[INFO] 累计报告已更新: {MASTER_PDF_NAME}")

# ================= Landmarker Init =================
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

# ================= UI Helpers =================
def overlay_template_feedback(frame):
    if template_message and (time.time() - template_msg_time) < TEMPLATE_FEEDBACK_SECONDS:
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + 540, 10 + 40), (0, 180, 255), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
        cv2.putText(frame, template_message, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)
    return frame

# ================= Main Loop =================
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
            cv2.putText(frame, f"Session {session_idx} - Get Ready: {remain}",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
            cv2.putText(frame, f"Gesture: {gesture}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,255), 2)
            cv2.putText(frame, "Keys: [t] capture template  [1] open  ESC quit",
                        (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)
            frame = overlay_template_feedback(frame)
            cv2.imshow("Hand Rehab Advanced", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cap.release(); cv2.destroyAllWindows(); return
            elif k == ord('1'):
                gesture = "open"

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

            if result.hand_landmarks:
                lm_list = result.hand_landmarks[0]

                # Template capture
                key_now = cv2.waitKey(1) & 0xFF
                if key_now == ord('t'):
                    norm_capture = normalize_landmarks_palm(lm_list)
                    new_template = {str(i): norm_capture[i] for i in FINGERTIPS if i in norm_capture}
                    gesture_templates[gesture] = new_template
                    save_templates(gesture_templates)
                    template_message = f"[模板已更新] {gesture} 指尖模板采集成功"
                    template_msg_time = time.time()
                    print(template_message)

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

                # Draw skeleton and info
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

            frame = overlay_template_feedback(frame)
            cv2.imshow("Hand Rehab Advanced", frame)
            k2 = cv2.waitKey(1) & 0xFF
            if k2 == 27:
                cap.release(); cv2.destroyAllWindows(); return
            elif k2 == ord('1'):
                gesture = "open"

        # -------- Result Phase --------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        history.append(best_total)
        append_csv(ts, session_idx, gesture, best_total, best_shape, best_open)
        plot_history()

        # Store session record for master PDF
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

        # Generate per session PDF (optional)
        if GENERATE_PER_SESSION_PDF:
            generate_session_pdf(ts, session_idx, gesture, best_total, best_shape, best_open, best_frame, best_dev)

        # Generate/refresh master cumulative PDF
        generate_master_pdf(session_records)

        # Prepare display summary
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

        cv2.putText(display, "Enter=Next  ESC=Quit", (20, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        display = overlay_template_feedback(display)
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
