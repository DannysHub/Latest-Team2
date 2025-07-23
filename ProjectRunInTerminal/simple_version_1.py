import os, cv2, time, math, csv, json
from datetime import datetime
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# -------- Config (Minimal) --------
MODEL_PATH = "/Users/dong/Desktop/rock_paper_scissors/team2/手部检测/task/hand_landmarker.task"
TEMPLATE_FILE = "gesture_templates.json"
CSV_FILE = "rehab_records.csv"

DEFAULT_GESTURE = "open"
FINGERTIPS = [4, 8, 12, 16, 20]
PALM_POINTS = [0, 5, 9, 13, 17]

SHAPE_WEIGHT = 0.7
OPENNESS_WEIGHT = 0.3
DEVIATION_GREEN_THRESHOLD = 0.05

PREP_SECONDS = 2        # 可调更快
CAPTURE_SECONDS = 8     # 可调
TEMPLATE_FEEDBACK_SECONDS = 2.0

# -------- State --------
gesture_templates = {}
template_message = ""
template_time = 0
history = []  # 存放总分（仅内存）

# -------- Template I/O --------
def load_templates():
    if not os.path.exists(TEMPLATE_FILE):
        with open(TEMPLATE_FILE, "w") as f:
            json.dump({DEFAULT_GESTURE: {}}, f, indent=2)
    with open(TEMPLATE_FILE, "r") as f:
        return json.load(f)

def save_templates():
    with open(TEMPLATE_FILE, "w") as f:
        json.dump(gesture_templates, f, indent=2)

def has_template(name):
    return name in gesture_templates and len(gesture_templates[name]) > 0

# -------- Geometry / Normalization --------
def palm_center_and_scale(lm):
    cx = sum(lm[i].x for i in PALM_POINTS) / len(PALM_POINTS)
    cy = sum(lm[i].y for i in PALM_POINTS) / len(PALM_POINTS)
    scale = math.dist((lm[5].x, lm[5].y), (lm[17].x, lm[17].y)) or 1e-6
    return cx, cy, scale

def normalize_landmarks(lm):
    cx, cy, s = palm_center_and_scale(lm)
    return {i: ((p.x - cx)/s, (p.y - cy)/s) for i, p in enumerate(lm)}

def openness_ratio(lm):
    cx, cy, s = palm_center_and_scale(lm)
    d = [math.dist((lm[t].x, lm[t].y), (cx, cy))/s for t in FINGERTIPS]
    avg = sum(d)/len(d)
    return min(1.0, avg / 0.9)

# -------- Scoring --------
def compute_devs(norm, template):
    devs = {}
    for k, (tx, ty) in template.items():
        idx = int(k)
        if idx in norm:
            x, y = norm[idx]
            devs[idx] = math.hypot(x - tx, y - ty)
    return devs

def finger_weight(idx):  # 拇指稍微重要一点
    return 1.2 if idx == 4 else 1.0

def shape_score(devs):
    if not devs: return 0.0
    sims = []
    for idx, d in devs.items():
        sim = math.exp(-8*d) * finger_weight(idx)
        sims.append(sim)
    return (sum(sims)/len(sims))*100.0

def combined(shape_part, open_part):
    return SHAPE_WEIGHT*shape_part + OPENNESS_WEIGHT*open_part

# -------- CSV --------
def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Time","Session","Gesture","Total","Shape","Open"])

def append_csv(ts, sess, gest, total, shape, open_part):
    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ts, sess, gest,
                                f"{total:.2f}", f"{shape:.2f}", f"{open_part:.2f}"])

# -------- Feedback Overlay --------
def overlay_template_info(frame):
    global template_message, template_time
    if template_message and time.time() - template_time < TEMPLATE_FEEDBACK_SECONDS:
        cv2.rectangle(frame, (10,10), (460,50), (0,180,255), -1)
        cv2.putText(frame, template_message, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 2)
    return frame

# -------- Mediapipe Init --------
def create_landmarker():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "缺少 hand_landmarker.task，请先下载:\n"
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

# -------- Main Loop --------
def main():
    global gesture_templates, template_message, template_time
    gesture_templates = load_templates()
    init_csv()
    gesture = DEFAULT_GESTURE

    landmarker = create_landmarker()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("❌ 无法打开摄像头"); return

    session = 0
    while True:
        session += 1

        # --- Preparation ---
        prep_start = time.time()
        while time.time() - prep_start < PREP_SECONDS:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            remain = PREP_SECONDS - int(time.time() - prep_start)
            cv2.putText(frame, f"Session {session} Ready:{remain}",
                        (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
            cv2.putText(frame, f"Gesture:{gesture}", (20,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,255), 2)
            cv2.putText(frame, "Keys: [t]模板  [1]open  ESC退出",
                        (20,440), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170,170,170), 1)
            overlay_template_info(frame)
            cv2.imshow("Hand Rehab (Minimal)", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27: cap.release(); cv2.destroyAllWindows(); return
            elif k == ord('1'): gesture = "open"

        # --- Capture Phase ---
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
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_img)

            shape_part = 0.0
            open_part = 0.0
            devs = {}

            if result.hand_landmarks:
                lm_list = result.hand_landmarks[0]

                key_now = cv2.waitKey(1) & 0xFF
                if key_now == ord('t'):
                    norm_cap = normalize_landmarks(lm_list)
                    tmpl = {str(i): norm_cap[i] for i in FINGERTIPS if i in norm_cap}
                    gesture_templates[gesture] = tmpl
                    save_templates()
                    template_message = f"[模板更新] {gesture}"
                    template_time = time.time()

                norm = normalize_landmarks(lm_list)
                if has_template(gesture):
                    devs = compute_devs(norm, gesture_templates[gesture])
                    shape_part = shape_score(devs)
                open_part = openness_ratio(lm_list) * 100.0
                total = combined(shape_part, open_part)

                if total > best_total:
                    best_total = total
                    best_shape = shape_part
                    best_open = open_part
                    best_dev = devs.copy()
                    best_frame = frame.copy()

                # 画骨架 & dev
                h, w, _ = frame.shape
                CONN = [(0,1),(1,2),(2,3),(3,4),
                        (0,5),(5,6),(6,7),(7,8),
                        (5,9),(9,10),(10,11),(11,12),
                        (9,13),(13,14),(14,15),(15,16),
                        (13,17),(17,18),(18,19),(19,20),(0,17)]
                for a,b in CONN:
                    ax, ay = int(lm_list[a].x*w), int(lm_list[a].y*h)
                    bx, by = int(lm_list[b].x*w), int(lm_list[b].y*h)
                    cv2.line(frame,(ax,ay),(bx,by),(0,255,0),2)
                for lm in lm_list:
                    cx_, cy_ = int(lm.x*w), int(lm.y*h)
                    cv2.circle(frame,(cx_,cy_),3,(0,165,255),-1)

                y = 150
                for tip in FINGERTIPS:
                    if tip in devs:
                        d = devs[tip]
                        color = (0,255,0) if d < DEVIATION_GREEN_THRESHOLD else (0,0,255)
                        cv2.putText(frame, f"{tip}:{d:.3f}", (20,y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        y += 18

                cv2.putText(frame, f"Shape:{shape_part:.1f}% Open:{open_part:.1f}%",
                            (20,120), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255),2)
                cv2.putText(frame, f"Best:{best_total:.1f}%", (20,95),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)

            remain = CAPTURE_SECONDS - int(time.time() - start)
            cv2.putText(frame, f"Capturing:{remain}s", (20,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
            cv2.putText(frame, f"Gesture:{gesture}", (20,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,255),1)
            overlay_template_info(frame)
            cv2.imshow("Hand Rehab (Minimal)", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                cap.release(); cv2.destroyAllWindows(); return

        # --- Result / Summary ---
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        history.append(best_total)
        append_csv(ts, session, gesture, best_total, best_shape, best_open)

        summary = best_frame if best_frame is not None else np.zeros((480,640,3),dtype=np.uint8)
        summary = summary.copy()
        cv2.putText(summary,"Session Complete!", (20,60),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
        cv2.putText(summary,f"Total:{best_total:.1f}%",(20,105),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,100),2)
        cv2.putText(summary,f"(Shape {best_shape:.1f}% Open {best_open:.1f}%)",(20,140),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),1)
        y=170
        for tip in FINGERTIPS:
            if tip in best_dev:
                d = best_dev[tip]
                color = (0,255,0) if d < DEVIATION_GREEN_THRESHOLD else (0,0,255)
                cv2.putText(summary,f"{tip}:{d:.3f}",(20,y),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
                y+=18
        cv2.putText(summary,"Enter=Next  ESC=Quit",(20,430),
                    cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1)
        overlay_template_info(summary)
        cv2.imshow("Hand Rehab (Minimal)", summary)

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 13: break
            if k == 27: cap.release(); cv2.destroyAllWindows(); return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
