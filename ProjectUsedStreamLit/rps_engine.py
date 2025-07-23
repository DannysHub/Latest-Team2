import os
import csv
import json
import math
import random
import datetime
import socket

import cv2
import mediapipe as mp

# ==== 常量 ====
TEMPLATE_FILE = "gesture_templates.json"
CSV_FILE      = "rps_open_fist_completion.csv"
DEFAULT_PORT  = 65432

# 手指关键点与掌心参考
FINGERTIPS = [8, 12, 16, 20]
PALM_POINTS = [0, 5, 9, 13, 17]

# ==== 模板与日志 ====

def load_templates():
    if not os.path.exists(TEMPLATE_FILE):
        data = {"open": {}, "fist": {}, "scissors": {}}
        with open(TEMPLATE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return data
    with open(TEMPLATE_FILE, 'r') as f:
        data = json.load(f)
    for k in ("open", "fist", "scissors"):
        data.setdefault(k, {})
    return data


def save_templates(templates):
    """保存手势模板到 JSON 文件"""
    with open(TEMPLATE_FILE, 'w') as f:
        json.dump(templates, f, indent=2)


def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["Time", "Mode", "Gesture", "BestCompletion"])


def append_csv(time, mode, gesture, best_completion):
    with open(CSV_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([time, mode, gesture, f"{best_completion:.2f}"])

# ==== 手势处理 辅助函数 ====

def palm_center_and_width(lm):
    cx = sum(lm[i].x for i in PALM_POINTS) / len(PALM_POINTS)
    cy = sum(lm[i].y for i in PALM_POINTS) / len(PALM_POINTS)
    w  = math.dist((lm[5].x, lm[5].y), (lm[17].x, lm[17].y)) or 1e-6
    return cx, cy, w


def normalize_landmarks(lm):
    cx, cy, w = palm_center_and_width(lm)
    return {i: ((pt.x-cx)/w, (pt.y-cy)/w) for i, pt in enumerate(lm)}


def openness_ratio(lm):
    cx, cy, w = palm_center_and_width(lm)
    ds = [math.dist((lm[t].x, lm[t].y), (cx, cy))/w for t in FINGERTIPS]
    return min(1.0, sum(ds)/len(ds)/0.9)


def compute_devs(norm, tpl):
    devs = {}
    for k, (tx, ty) in tpl.items():
        idx = int(k)
        if idx in norm:
            x, y = norm[idx]
            devs[idx] = math.hypot(x-tx, y-ty)
    return devs


def shape_score(devs, base):
    if not devs:
        return 0.0
    sims = []
    for i, d in devs.items():
        bonus = 1.0
        if base == "open"  and i == 4:    bonus = 1.2
        if base == "fist"  and i in (8, 12): bonus = 1.1
        sims.append(math.exp(-8*d) * bonus)
    return sum(sims)/len(sims) * 100


def total_score(base, shape, open_pct):
    if base == "rock":
        return 0.6 * shape + 0.4 * (100 - open_pct)
    if base == "paper":
        return 0.7 * shape + 0.3 * open_pct
    # scissors
    return 0.5 * shape + 0.5 * (100 - abs(open_pct - 50))


def classify_rps(lm)->str:
    fingers = [1 if lm[t].y < lm[t-2].y else 0 for t in FINGERTIPS]
    c = sum(fingers)
    if c == 0:    return "rock"
    if c == 2:    return "scissors"
    if c >= 4:    return "paper"
    return "unknown"

def judge(player: str, ai: str)->str:
    if player == ai:
        return "Draw"
    win = (player=="rock"     and ai=="scissors") or \
          (player=="scissors" and ai=="paper")    or \
          (player=="paper"    and ai=="rock")
    return "You Win!" if win else "You Lose!"

# ==== 后端主类 ====

class RPSBackend:
    def __init__(
        self,
        mode="local",
        host_ip="",
        cam_idx=0,
        templates=None
    ):
        init_csv()
        self.mode     = mode
        self.host     = host_ip
        self.port     = DEFAULT_PORT
        self.score    = 0
        self.wins     = 0
        self.losses   = 0
        self.draws    = 0
        self.round_id = 0
        self.templates= templates or load_templates()
        self.last_norm= None

        # 摄像头与 Mediapipe Hands
        self.cap   = cv2.VideoCapture(cam_idx)
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # 网络通信初始化
        self.sock = None
        self.conn = None
        if self.mode == "host":
            self.sock = socket.socket()
            self.sock.bind(("", self.port))
            self.sock.listen(1)
            self.conn, _ = self.sock.accept()
        elif self.mode == "client":
            self.sock = socket.socket()
            self.sock.connect((self.host, self.port))

    def start_round(self):
        self.round_id += 1

    def process_frame(self):
        """返回 {'frame':bgr_image, 'gesture':str, 'shape':float, 'open':float, 'total':float}"""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera fail")
        frame = cv2.flip(frame,1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = self.hands.process(rgb)

        # 初始化数据
        data = {"frame": frame, "gesture": None, "shape": 0.0, "open": 0.0, "total": None}
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            # 识别手势
            g = classify_rps(lm)
            data["gesture"] = g
            # 保存规范化 landmarks 以备采集模板
            self.last_norm = normalize_landmarks(lm)
            # 计算完成度
            mapping = {"rock": "fist", "paper": "open", "scissors": "scissors"}
            base = mapping.get(g)
            if base in self.templates:
                norm = self.last_norm
                tpl  = self.templates.get(base, {})
                devs = compute_devs(norm, tpl)
                shp  = shape_score(devs, base)
                op_pct = openness_ratio(lm) * 100
                tot  = total_score(base, shp, op_pct)
                data.update({"shape": shp, "open": op_pct, "total": tot})
        return data

    def ai_choice(self, player):
        if self.mode in ("host", "client"):
            # 网络模式，直接传输手势
            if self.mode == "host":
                self.conn.sendall(player.encode())
                return self.conn.recv(1024).decode()
            else:
                self.sock.sendall(player.encode())
                return self.sock.recv(1024).decode()
        # 本地 vs AI：随机或根据得分动态调整
        return random.choice(["rock","paper","scissors"])

    def end_round(self, player, best_total=None):
        """结束一轮，返回 (结果, ai_gesture)"""
        ai = self.ai_choice(player)
        self.last_ai = ai

        res = judge(player, ai)
        if res == "You Win!":
            self.wins   += 1
            self.score  += 3
        elif res == "You Lose!":
            self.losses += 1
            self.score  -= 1
        else:
            self.draws  += 1

        append_csv(
            datetime.datetime.now().replace(microsecond=0).isoformat(),
            self.mode,
            player,
            best_total or getattr(self, 'last_total', 0)
        )
        return res, ai

    def release(self):
        self.cap.release()
        if self.conn: self.conn.close()
        if self.sock: self.sock.close()
        self.hands.close()
