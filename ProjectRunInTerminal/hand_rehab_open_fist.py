#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand Rehab Completion (Open / Fist Only)
=======================================
精简版本：仅评估“张开(Open)”与“握拳(Fist)”两个康复训练手势的完成度。

特性:
  * Mediapipe Hands (solutions) 获取 21 关键点
  * 模板采集：按 't' 记录当前帧为当前手势模板（只保存 5 个指尖归一化坐标）
  * 实时计算：
        - Shape% : 指尖位置与模板的匹配度（指数衰减 + 指尖权重）
        - Open%  : 手张开度 (五指尖到掌心的平均归一化距离)
        - Closure%: 100 - Open%
        - Total% : 
            open  手势  ->  0.7*Shape + 0.3*Open
            fist  手势  ->  0.6*Shape + 0.4*Closure
  * 滑动窗口平滑 (默认窗口=5)
  * 每个 Session (一次采集周期) 记录最佳 (Total, Shape, Open, Closure) 到 CSV
  * Session 流程：准备阶段(倒计时) -> 采集阶段(实时评分) -> 结果展示(等待 Enter / ESC)
  * 仅依赖：opencv-python, mediapipe, numpy (标准库: csv/json/time/os/math)

快捷键:
  1 / 2     切换手势 (open / fist)
  t         采集 / 覆盖该手势模板
  Enter     结束当前 Session, 进入下一次
  ESC       退出程序

文件:
  gesture_templates.json   保存两个手势模板
  rehab_open_fist_records.csv  Session 结果 CSV

可选自定义:
  - 调整 PREP_SECONDS, CAPTURE_SECONDS
  - 调整权重 SHAPE_WEIGHT_OPEN / SHAPE_WEIGHT_FIST
  - 关闭平滑: 将 SMOOTH_WINDOW = 1
"""

import os
import cv2
import time
import math
import csv
import json
from datetime import datetime
from collections import deque

import mediapipe as mp

# ----------------------------------------------------------------------------
# 配置参数
# ----------------------------------------------------------------------------
TEMPLATE_FILE = "gesture_templates.json"
CSV_FILE = "rehab_open_fist_records.csv"

GESTURES = ["open", "fist"]
GESTURE_KEYS = {ord('1'): "open", ord('2'): "fist"}
DEFAULT_GESTURE = "open"

FINGERTIPS = [4, 8, 12, 16, 20]
PALM_POINTS = [0, 5, 9, 13, 17]

# 权重（可按需要微调）
SHAPE_WEIGHT_OPEN = 0.7
OPEN_WEIGHT_OPEN = 0.3
SHAPE_WEIGHT_FIST = 0.6
CLOSURE_WEIGHT_FIST = 0.4

# 评分显示 & 平滑
DEVIATION_GREEN_THRESHOLD = 0.05
SMOOTH_WINDOW = 5  # 最近若干帧 Total 平滑

# Session 时序
PREP_SECONDS = 2
CAPTURE_SECONDS = 8

# 颜色配置
COLOR_TEXT = (255, 255, 255)
COLOR_HINT = (180, 180, 180)
COLOR_EMPH = (0, 255, 255)
COLOR_GOOD = (0, 255, 0)
COLOR_BAD = (0, 0, 255)

# ----------------------------------------------------------------------------
# 模板管理
# ----------------------------------------------------------------------------

def load_templates():
    if not os.path.exists(TEMPLATE_FILE):
        data = {g: {} for g in GESTURES}
        with open(TEMPLATE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return data
    with open(TEMPLATE_FILE, 'r') as f:
        data = json.load(f)
    changed = False
    for g in GESTURES:
        if g not in data:
            data[g] = {}
            changed = True
    if changed:
        with open(TEMPLATE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
    return data

def save_templates(tpl):
    with open(TEMPLATE_FILE, 'w') as f:
        json.dump(tpl, f, indent=2)

gesture_templates = load_templates()

def has_template(gesture: str) -> bool:
    return gesture in gesture_templates and len(gesture_templates[gesture]) > 0

# ----------------------------------------------------------------------------
# 几何 / 归一化
# ----------------------------------------------------------------------------

def palm_center_and_width(lm_list):
    """掌心近似中心 + 尺度(拇指根到小指根的距离)."""
    cx = sum(lm_list[i].x for i in PALM_POINTS) / len(PALM_POINTS)
    cy = sum(lm_list[i].y for i in PALM_POINTS) / len(PALM_POINTS)
    w_ref = math.dist((lm_list[5].x, lm_list[5].y),
                      (lm_list[17].x, lm_list[17].y)) or 1e-6
    return cx, cy, w_ref

def normalize_landmarks(lm_list):
    """21 点归一化到以掌心为中心, 以掌宽为尺度的坐标系."""
    cx, cy, w_ref = palm_center_and_width(lm_list)
    return {i: ((lm.x - cx)/w_ref, (lm.y - cy)/w_ref) for i, lm in enumerate(lm_list)}

def openness_ratio(lm_list):
    """平均指尖距掌心的归一化距离 -> 0~1."""
    cx, cy, w_ref = palm_center_and_width(lm_list)
    ds = [math.dist((lm_list[t].x, lm_list[t].y), (cx, cy))/w_ref for t in FINGERTIPS]
    avg = sum(ds)/len(ds)
    return min(1.0, avg / 0.9)

# ----------------------------------------------------------------------------
# 偏差与评分
# ----------------------------------------------------------------------------

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
        return 1.2 if idx == 4 else 1.0  # 拇指稍重要
    if gesture == "fist":
        return 1.1 if idx in (8, 12) else 1.0  # 食指/中指略加权
    return 1.0

def shape_score(devs, gesture):
    if not devs:
        return 0.0
    sims = []
    for idx, d in devs.items():
        sims.append(math.exp(-8*d) * finger_weight(idx, gesture))
    return (sum(sims)/len(sims))*100.0

def total_score(gesture, shape_part, open_part):
    if gesture == "open":
        return SHAPE_WEIGHT_OPEN*shape_part + OPEN_WEIGHT_OPEN*open_part
    if gesture == "fist":
        closure = 100 - open_part
        return SHAPE_WEIGHT_FIST*shape_part + CLOSURE_WEIGHT_FIST*closure
    return shape_part  # fallback

# ----------------------------------------------------------------------------
# CSV
# ----------------------------------------------------------------------------

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            csv.writer(f).writerow([
                'Time','Session','Gesture','Total','Shape','Open','Closure'
            ])

def append_csv(ts, session, gesture, total, shape, open_pct):
    closure = 100 - open_pct
    with open(CSV_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([
            ts, session, gesture,
            f"{total:.2f}", f"{shape:.2f}", f"{open_pct:.2f}", f"{closure:.2f}"
        ])

# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def main():
    init_csv()
    gesture = DEFAULT_GESTURE

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=1,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    session = 0
    while True:
        session += 1
        # ---------------- 准备阶段 ----------------
        prep_start = time.time()
        while time.time() - prep_start < PREP_SECONDS:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            remain = PREP_SECONDS - int(time.time() - prep_start)
            cv2.putText(frame, f"Session {session} Ready:{remain}", (20,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_EMPH, 2)
            cv2.putText(frame, f"Gesture:{gesture}", (20,90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.70, (200,200,255), 2)
            cv2.putText(frame, "[1]open [2]fist  [t]模板  ESC退出", (20,430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HINT, 1)
            cv2.imshow("Rehab Open/Fist", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:
                cap.release(); cv2.destroyAllWindows(); return
            elif k in GESTURE_KEYS:
                gesture = GESTURE_KEYS[k]

        # ---------------- 采集阶段 ----------------
        best_total = 0.0
        best_shape = 0.0
        best_open = 0.0
        best_dev = {}
        best_frame = None

        total_window = deque(maxlen=SMOOTH_WINDOW)

        start = time.time()
        while time.time() - start < CAPTURE_SECONDS:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            shape_part = 0.0
            open_part = 0.0
            devs = {}

            if results.multi_hand_landmarks:
                lm_list = results.multi_hand_landmarks[0].landmark

                # 模板采集
                key_temp = cv2.waitKey(1) & 0xFF
                if key_temp == ord('t'):
                    norm_cap = normalize_landmarks(lm_list)
                    gesture_templates[gesture] = {str(i): norm_cap[i] for i in FINGERTIPS}
                    save_templates(gesture_templates)
                    print(f"[模板更新] {gesture}")

                # 评分
                norm = normalize_landmarks(lm_list)
                if has_template(gesture):
                    devs = compute_devs(norm, gesture_templates[gesture])
                    shape_part = shape_score(devs, gesture)
                open_part = openness_ratio(lm_list) * 100.0
                total = total_score(gesture, shape_part, open_part)

                # 平滑
                total_window.append(total)
                total_disp = sum(total_window)/len(total_window)

                # 更新最佳
                if total_disp > best_total:
                    best_total = total_disp
                    best_shape = shape_part
                    best_open = open_part
                    best_dev = devs.copy()
                    best_frame = frame.copy()

                # 绘制关键点 (简单画法)
                h, w, _ = frame.shape
                # 连接骨架索引对
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
                    cv2.line(frame, (ax,ay), (bx,by), (0,255,0), 2)
                for lm in lm_list:
                    cx_, cy_ = int(lm.x*w), int(lm.y*h)
                    cv2.circle(frame, (cx_,cy_), 3, (0,165,255), -1)

                # 偏差列表
                y0 = 170
                for tip in FINGERTIPS:
                    if tip in devs:
                        d = devs[tip]
                        color = COLOR_GOOD if d < DEVIATION_GREEN_THRESHOLD else COLOR_BAD
                        cv2.putText(frame, f"{tip}:{d:.3f}", (20,y0),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                        y0 += 18

                closure = 100 - open_part
                cv2.putText(frame, f"Shape:{shape_part:.1f}% Open:{open_part:.1f}% Closure:{closure:.1f}%", (20,140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0,200,255), 2)
                cv2.putText(frame, f"Total:{total_disp:.1f}% (Best:{best_total:.1f}%)", (20,115),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2)

            remain = CAPTURE_SECONDS - int(time.time() - start)
            cv2.putText(frame, f"{gesture} {remain}s", (20,35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2)
            cv2.putText(frame, "[1]open [2]fist [t]模板 ESC退", (20,430),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_HINT, 1)
            cv2.imshow("Rehab Open/Fist", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cap.release(); cv2.destroyAllWindows(); return
            elif key in GESTURE_KEYS:
                gesture = GESTURE_KEYS[key]

        # ---------------- 结果阶段 ----------------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        append_csv(ts, session, gesture, best_total, best_shape, best_open)

        summary = best_frame if best_frame is not None else (255*0.2)*np.ones((480,640,3), dtype=np.uint8)
        if best_frame is None:
            summary = (255*0.2)*np.ones((480,640,3), dtype=np.uint8)
        else:
            summary = best_frame.copy()

        closure_best = 100 - best_open
        cv2.putText(summary, "Session Done!", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_EMPH, 2)
        cv2.putText(summary, f"Gesture:{gesture}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,100), 2)
        cv2.putText(summary, f"Best Total:{best_total:.1f}%", (20,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
        cv2.putText(summary, f"Shape:{best_shape:.1f}% Open:{best_open:.1f}% Closure:{closure_best:.1f}%", (20,175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 2)
        y0 = 210
        for tip in FINGERTIPS:
            if tip in best_dev:
                d = best_dev[tip]
                color = COLOR_GOOD if d < DEVIATION_GREEN_THRESHOLD else COLOR_BAD
                cv2.putText(summary, f"{tip}:{d:.3f}", (20,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y0 += 20
        cv2.putText(summary, "Enter下一次  ESC退出", (20,430), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HINT, 1)
        cv2.imshow("Rehab Open/Fist", summary)

        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 13:  # Enter 下一 Session
                break
            if k == 27:  # ESC 退出
                cap.release(); cv2.destroyAllWindows(); return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import numpy as np  # 放在末尾仅用于创建空白帧
    main()
