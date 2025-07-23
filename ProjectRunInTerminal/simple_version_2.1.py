#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hand Rehab Minimal Multi-Gesture Version (Commented)
===================================================
功能概述:
    一个精简、低依赖的手部康复/手势质量训练脚本。
    - 使用 MediaPipe 获取 21 个手部关键点
    - 支持三类手势: open / fist / pinch
    - 支持按 't' 即时采集模板 (只存指尖关键点的归一化坐标)
    - 实时计算: 形状匹配分 (shape)、张开度(open) 或 其他衍生指标
    - 综合得分策略根据手势类型不同而变化
    - 保存每个 Session 最佳结果到 CSV
    - 提供两种检测后端:
        * tasks API (需 hand_landmarker.task 模型文件)
        * 旧 solutions.Hands API (无需模型文件, 设置 USE_OLD_API=True)

依赖:
    pip install --upgrade mediapipe opencv-python numpy

目录 / 文件:
    - hand_landmarker.task            : 任务模型 (若使用 tasks API)
    - gesture_templates.json          : 模板(指尖)坐标 JSON
    - rehab_records.csv               : 结果记录 (追加写入)

快速运行:
    1. (推荐) 下载任务模型:
       curl -L -o hand_landmarker.task \
         https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    2. python hand_rehab_min_multi_commented.py
    3. 按键:
       1 / 2 / 3  切换手势(open / fist / pinch)
       t          记录当前帧为该手势模板
       Enter      进入下一 Session
       ESC        退出

可扩展方向(示例):
    - 增加其它手势: 在 GESTURES / GESTURE_KEYS / scoring 分支增加逻辑
    - 平滑得分: 维护一个窗口对 total 做移动平均
    - Web 接口: 用 FastAPI 或 Streamlit 包裹
    - 可视化历史: 单独脚本读取 CSV 画图, 避免主脚本引入 matplotlib
"""

import os
import cv2
import time
import math
import csv
import json
from datetime import datetime
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------------------------------------------------------------------
# 1. 基本配置区域 (可根据需求调整)
# ---------------------------------------------------------------------------
MODEL_PATH = "hand_landmarker.task"      # 如果模型放在别处可改绝对路径
TEMPLATE_FILE = "gesture_templates.json" # 手势模板保存文件
CSV_FILE = "rehab_records.csv"           # 结果记录 CSV

GESTURES = ["open", "fist", "pinch"]     # 支持的手势名称
GESTURE_KEYS = {ord('1'): "open", ord('2'): "fist", ord('3'): "pinch"}  # 切换键
DEFAULT_GESTURE = "open"                 # 初始手势

FINGERTIPS = [4, 8, 12, 16, 20]          # 五个指尖 index
PALM_POINTS = [0, 5, 9, 13, 17]          # 掌心近似采样点 (用于中心/尺度)

# 得分权重 (open 手势使用; 其他手势自己定义组合逻辑)
SHAPE_WEIGHT = 0.7
OPENNESS_WEIGHT = 0.3

DEVIATION_GREEN_THRESHOLD = 0.05         # 指尖偏差低于该值显示绿色(匹配较好)

# Session 时序参数
PREP_SECONDS = 2                         # 准备阶段秒数 (显示倒计时)
CAPTURE_SECONDS = 8                      # 采集阶段秒数 (实时评分)
TEMPLATE_FEEDBACK_SECONDS = 2.0          # 采集模板后屏幕提示显示时长

# 是否使用旧 API (若无法下载 .task 模型可设 True)
USE_OLD_API = False

# ---------------------------------------------------------------------------
# 2. 模板管理函数
# ---------------------------------------------------------------------------
def load_templates():
    """加载(或初始化)手势模板 JSON。模板存储的是每个手势下各指尖的归一化坐标。
    结构示例:
    {
        "open": {"4": [x,y], "8": [x,y], ...},
        "fist": {...},
        "pinch": {...}
    }
    """
    if not os.path.exists(TEMPLATE_FILE):
        # 首次运行创建空模板
        with open(TEMPLATE_FILE, "w") as f:
            json.dump({g: {} for g in GESTURES}, f, indent=2)
        return {g: {} for g in GESTURES}
    with open(TEMPLATE_FILE, "r") as f:
        data = json.load(f)
    # 确保新加的手势键存在
    changed = False
    for g in GESTURES:
        if g not in data:
            data[g] = {}
            changed = True
    if changed:
        with open(TEMPLATE_FILE, "w") as f:
            json.dump(data, f, indent=2)
    return data

def save_templates(tpl):
    """保存手势模板到 JSON。"""
    with open(TEMPLATE_FILE, "w") as f:
        json.dump(tpl, f, indent=2)

gesture_templates = load_templates()

def has_template(name):
    """判断某手势是否已有模板记录。"""
    return name in gesture_templates and len(gesture_templates[name]) > 0

# ---------------------------------------------------------------------------
# 3. 关键点归一化 / 几何辅助函数
# ---------------------------------------------------------------------------
def palm_center_and_width(lm_list):
    """计算掌心近似中心 & 尺度:
    - 中心: 选取 PALM_POINTS 对应关键点求平均
    - 尺度: 使用拇指基部(5) 到 小指基部(17) 的距离作为尺度因子
    返回: (cx, cy, w_ref)
    """
    cx = sum(lm_list[i].x for i in PALM_POINTS) / len(PALM_POINTS)
    cy = sum(lm_list[i].y for i in PALM_POINTS) / len(PALM_POINTS)
    w_ref = math.dist((lm_list[5].x, lm_list[5].y),
                      (lm_list[17].x, lm_list[17].y)) or 1e-6
    return cx, cy, w_ref

def normalize_landmarks(lm_list):
    """把全部 21 个关键点坐标转为以掌心为中心、掌宽(5~17)为尺度的归一化坐标。"""
    cx, cy, w_ref = palm_center_and_width(lm_list)
    return {
        i: ((lm.x - cx)/w_ref, (lm.y - cy)/w_ref)
        for i, lm in enumerate(lm_list)
    }

def openness_ratio(lm_list):
    """衡量手张开程度:
       计算五指尖到掌心的归一化距离均值, 再相对于经验上限(0.9)截断到 [0,1]。
       返回: 0(合拢) ~ 1(最大张开) 之间的比例
    """
    cx, cy, w_ref = palm_center_and_width(lm_list)
    ds = [
        math.dist((lm_list[t].x, lm_list[t].y), (cx, cy)) / w_ref
        for t in FINGERTIPS
    ]
    avg = sum(ds)/len(ds)
    return min(1.0, avg / 0.9)

# ---------------------------------------------------------------------------
# 4. 偏差与得分计算
# ---------------------------------------------------------------------------
def compute_devs(norm, template_dict):
    """计算指尖与模板的欧氏距离(均在归一化坐标系)。
    norm: {landmark_index: (x,y)}
    template_dict: {"4": [tx, ty], ...}
    返回: {index: deviation}
    """
    devs = {}
    for k, (tx, ty) in template_dict.items():
        idx = int(k)
        if idx in norm:
            x, y = norm[idx]
            devs[idx] = math.hypot(x - tx, y - ty)
    return devs

def finger_weight(idx, gesture):
    """不同手势对指尖赋予不同权重:
       - open: 拇指更关键
       - fist: 加重食指/中指 (抓握对称性)
       - pinch: 加重拇指+食指, 其他降低
    """
    if gesture == "open":
        return 1.2 if idx == 4 else 1.0
    if gesture == "fist":
        return 1.1 if idx in (8, 12) else 1.0
    if gesture == "pinch":
        return 1.5 if idx in (4, 8) else 0.5
    return 1.0

def shape_score(devs, gesture):
    """形状匹配分:
       对每个指尖偏差 d 做指数衰减: exp(-8*d)
       再乘以手势相关权重 finger_weight
       取平均后乘 100 转为百分比。
    """
    if not devs:
        return 0.0
    sims = []
    for idx, d in devs.items():
        s = math.exp(-8*d) * finger_weight(idx, gesture)
        sims.append(s)
    return (sum(sims)/len(sims))*100.0

def gesture_total(gesture, shape_part, open_part, lm_list):
    """根据手势类型组合不同的指标为总分:
       open  : 0.7 * shape + 0.3 * openness
       fist  : 0.6 * shape + 0.4 * (100 - openness)
       pinch : 0.5 * shape + 0.3 * pinch_score + 0.2 * mid_pref
               - pinch_score: 拇指与食指间距离与目标值(0.15)的接近程度
               - mid_pref   : openness 距离 50% 越近越好 (便于保持适度张开)
    """
    if gesture == "open":
        return SHAPE_WEIGHT*shape_part + OPENNESS_WEIGHT*open_part
    if gesture == "fist":
        closure = 100 - open_part
        return 0.6*shape_part + 0.4*closure
    if gesture == "pinch":
        pinch_score = 0.0
        if lm_list:
            pt4, pt8 = lm_list[4], lm_list[8]
            dist = math.hypot(pt4.x - pt8.x, pt4.y - pt8.y)
            _, _, w_ref = palm_center_and_width(lm_list)
            nd = dist / w_ref  # 归一化捏合距离
            # 目标距离 ~0.15, 用指数衰减评估接近程度
            pinch_score = math.exp(-8*abs(nd - 0.15))*100.0
        mid_pref = 100 - abs(open_part - 50)  # openness 越接近 50 越优
        return 0.5*shape_part + 0.3*pinch_score + 0.2*mid_pref
    # 默认回退
    return SHAPE_WEIGHT*shape_part + OPENNESS_WEIGHT*open_part

# ---------------------------------------------------------------------------
# 5. CSV 结果记录
# ---------------------------------------------------------------------------
def init_csv():
    """若 CSV 不存在则写入表头。"""
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["Time","Session","Gesture","Total","Shape","Open"])

def append_csv(ts, sess, gesture, total, shape, open_part):
    """追加一条最佳结果记录。"""
    with open(CSV_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            ts, sess, gesture,
            f"{total:.2f}", f"{shape:.2f}", f"{open_part:.2f}"
        ])

# ---------------------------------------------------------------------------
# 6. 检测后端初始化
# ---------------------------------------------------------------------------
def create_landmarker_tasks():
    """创建 tasks API 的 HandLandmarker 实例 (需要模型文件)."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "缺少 hand_landmarker.task；无法创建 tasks HandLandmarker。\n"
            "下载命令:\n"
            "curl -L -o hand_landmarker.task "
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task\n"
            "或设置 USE_OLD_API=True 退回旧 API。"
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
    """创建旧 solutions.Hands 实例 (无需模型文件)."""
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(static_image_mode=False,
                          max_num_hands=1,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)

# ---------------------------------------------------------------------------
# 7. 主流程
# ---------------------------------------------------------------------------
def main():
    # 初始化 CSV 表头
    init_csv()

    gesture = DEFAULT_GESTURE  # 当前手势
    hands = None               # 旧 API 句柄
    landmarker = None          # tasks API 句柄

    # 根据开关初始化检测器
    if USE_OLD_API:
        hands = create_old_hands()
    else:
        landmarker = create_landmarker_tasks()

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    session = 0
    template_msg = ""   # 最近一次模板采集提示文本
    template_time = 0   # 提示开始时间戳

    while True:
        session += 1

        # --------------------------- 准备阶段 ---------------------------
        prep_start = time.time()
        while time.time() - prep_start < PREP_SECONDS:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            remain = PREP_SECONDS - int(time.time() - prep_start)

            # 基本文本提示
            cv2.putText(frame, f"Session {session} Ready:{remain}",
                        (20,50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(frame, f"Gesture:{gesture}",
                        (20,90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,255), 2)
            cv2.putText(frame, "[1]open [2]fist [3]pinch  [t]模板采集  ESC退出",
                        (20,430), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180),1)

            # 模板采集反馈条 (短时间显示)
            if template_msg and time.time()-template_time < TEMPLATE_FEEDBACK_SECONDS:
                cv2.rectangle(frame,(10,10),(470,40),(0,180,255),-1)
                cv2.putText(frame, template_msg, (20,34),
                            cv2.FONT_HERSHEY_SIMPLEX,0.55,(30,30,30),2)

            cv2.imshow("Hand Rehab Minimal", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC
                cap.release()
                cv2.destroyAllWindows()
                return
            elif k in GESTURE_KEYS:  # 切换手势
                gesture = GESTURE_KEYS[k]

        # --------------------------- 采集阶段 ---------------------------
        best_total = 0.0
        best_shape = 0.0
        best_open = 0.0
        best_dev = {}
        best_frame = None

        start = time.time()
        while time.time() - start < CAPTURE_SECONDS:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 取得 landmarks 列表 (长度 21)
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
                # (1) 按 't' 采集模板 (只存指尖)
                if cv2.waitKey(1) & 0xFF == ord('t'):
                    norm_cap = normalize_landmarks(lm_list)
                    gesture_templates[gesture] = {
                        str(i): norm_cap[i] for i in FINGERTIPS if i in norm_cap
                    }
                    save_templates(gesture_templates)
                    template_msg = f"[模板更新] {gesture}"
                    template_time = time.time()

                # (2) 计算归一化坐标 + 偏差
                norm = normalize_landmarks(lm_list)
                if has_template(gesture):
                    devs = compute_devs(norm, gesture_templates[gesture])
                    shape_part = shape_score(devs, gesture)

                # (3) 张开度 (0~1) *100 转百分比
                open_part = openness_ratio(lm_list) * 100.0

                # (4) 综合总分 (手势特异策略)
                total = gesture_total(gesture, shape_part, open_part, lm_list)

                # (5) 若刷新最佳分则保存
                if total > best_total:
                    best_total = total
                    best_shape = shape_part
                    best_open = open_part
                    best_dev = devs.copy()
                    best_frame = frame.copy()

                # (6) 绘制骨架与关键点
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

                # (7) 指尖偏差显示
                y0 = 150
                for tip in FINGERTIPS:
                    if tip in devs:
                        d = devs[tip]
                        color = (0,255,0) if d < DEVIATION_GREEN_THRESHOLD else (0,0,255)
                        cv2.putText(frame, f"{tip}:{d:.3f}", (20,y0),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
                        y0 += 16

                # (8) 分数信息
                cv2.putText(frame, f"Shape:{shape_part:.1f}% Open:{open_part:.1f}%",
                            (20,120), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,255),2)
                cv2.putText(frame, f"Best:{best_total:.1f}%",
                            (20,95), cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)

            # (9) 顶部状态行 & 提示
            remain = CAPTURE_SECONDS - int(time.time() - start)
            cv2.putText(frame, f"{gesture} {remain}s",
                        (20,35), cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
            cv2.putText(frame, "[1]open [2]fist [3]pinch [t]模板 ESC退",
                        (20,430), cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)

            # 模板更新提示
            if template_msg and time.time()-template_time < TEMPLATE_FEEDBACK_SECONDS:
                cv2.rectangle(frame,(10,10),(300,40),(0,180,255),-1)
                cv2.putText(frame, template_msg, (18,34),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,(30,30,30),2)

            cv2.imshow("Hand Rehab Minimal", frame)

            # (10) 采集阶段按键处理
            k2 = cv2.waitKey(1) & 0xFF
            if k2 == 27:  # ESC 立即退出
                cap.release()
                cv2.destroyAllWindows()
                return
            elif k2 in GESTURE_KEYS:
                gesture = GESTURE_KEYS[k2]

        # --------------------------- 结果展示阶段 ---------------------------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        append_csv(ts, session, gesture, best_total, best_shape, best_open)

        summary = best_frame if best_frame is not None else np.zeros((480,640,3),dtype=np.uint8)
        summary = summary.copy()
        cv2.putText(summary, f"{gesture} Done!",
                    (20,60), cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
        cv2.putText(summary,
                    f"Total:{best_total:.1f}%  (Shape {best_shape:.1f}%  Open {best_open:.1f}%)",
                    (20,105), cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,100),2)

        # 显示最佳指尖偏差列表
        y0 = 140
        for tip in FINGERTIPS:
            if tip in best_dev:
                d = best_dev[tip]
                color = (0,255,0) if d < DEVIATION_GREEN_THRESHOLD else (0,0,255)
                cv2.putText(summary, f"{tip}:{d:.3f}", (20,y0),
                            cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
                y0 += 18

        cv2.putText(summary, "Enter=Next  ESC=Quit",
                    (20,430), cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),1)
        cv2.imshow("Hand Rehab Minimal", summary)

        # 等待用户决定: Enter 下一轮; ESC 退出
        while True:
            k = cv2.waitKey(1) & 0xFF
            if k == 13:   # Enter
                break
            if k == 27:   # ESC
                cap.release()
                cv2.destroyAllWindows()
                return

    # 正常结束释放资源
    cap.release()
    cv2.destroyAllWindows()

# ---------------------------------------------------------------------------
# 8. 入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
