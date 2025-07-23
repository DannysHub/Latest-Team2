import os
import csv
import json
import math
import random
import datetime
import socket
from typing import Dict, List, Tuple, Any, Optional

import cv2
import mediapipe as mp
import numpy as np

# ==== 配置常量 ====
TEMPLATE_FILE = "gesture_templates.json"
CSV_FILE      = "rps_open_fist_completion.csv"

# 模式
MODE_LOCAL  = "local"
MODE_HOST   = "host"
MODE_CLIENT = "client"

# 三种手势加权
SHAPE_WEIGHT_OPEN   = 0.7
OPEN_WEIGHT_OPEN    = 0.3
SHAPE_WEIGHT_FIST   = 0.6
CLOSURE_WEIGHT_FIST = 0.4

FINGERTIPS  = [8,12,16,20]
PALM_POINTS = [0,5,9,13,17]

# AI 难度参数
ASSIST_INIT_WIN = 0.30
ASSIST_DRAW     = 0.30
ASSIST_MIN_WIN  = 0.20

HIGH_INIT_WIN   = 0.35
HIGH_DRAW       = 0.30
HIGH_MIN_WIN    = 0.30
HIGH_MAX_WIN    = 0.45

DEFAULT_PORT = 65432

# ==== 模板与日志 ====

def load_templates() -> Dict[str, Dict[str, List[float]]]:
    if not os.path.exists(TEMPLATE_FILE):
        data = {"open": {}, "fist": {}, "scissors": {}}
        with open(TEMPLATE_FILE, 'w') as f:
            json.dump(data, f, indent=2)
        return data
    with open(TEMPLATE_FILE, 'r') as f:
        data = json.load(f)
    for k in ("open","fist","scissors"):
        data.setdefault(k,{})
    return data

def init_csv() -> None:
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            csv.writer(f).writerow([
                "Time","Round","Mode","Player","Opponent","Result",
                "Total","Shape","Open","Score","Wins","Losses","Draws"
            ])

def append_csv(time, mode, gesture, best_completion):
    with open(CSV_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([time, mode, gesture, f"{best_completion:.2f}"])

# ==== 手势处理 辅助函数 ====

def palm_center_and_width(lm) -> Tuple[float,float,float]:
    cx = sum(lm[i].x for i in PALM_POINTS) / len(PALM_POINTS)
    cy = sum(lm[i].y for i in PALM_POINTS) / len(PALM_POINTS)
    w  = math.dist((lm[5].x,lm[5].y), (lm[17].x,lm[17].y)) or 1e-6
    return cx, cy, w

def normalize_landmarks(lm) -> Dict[int,Tuple[float,float]]:
    cx, cy, w = palm_center_and_width(lm)
    return {i:((pt.x-cx)/w,(pt.y-cy)/w) for i,pt in enumerate(lm)}

def openness_ratio(lm) -> float:
    cx, cy, w = palm_center_and_width(lm)
    ds = [math.dist((lm[t].x,lm[t].y),(cx,cy))/w for t in FINGERTIPS]
    return min(1.0, sum(ds)/len(ds)/0.9)

def compute_devs(norm: Dict[int,Tuple[float,float]], tpl: Dict[str,List[float]]):
    devs = {}
    for k,(tx,ty) in tpl.items():
        idx = int(k)
        if idx in norm:
            x,y = norm[idx]
            devs[idx] = math.hypot(x-tx, y-ty)
    return devs

def shape_score(devs: Dict[int,float], base: str) -> float:
    if not devs:
        return 0.0
    sims = []
    for i,d in devs.items():
        bonus = 1.0
        if base=="open"  and i==4:   bonus = 1.2
        if base=="fist"  and i in (8,12): bonus = 1.1
        sims.append(math.exp(-8*d)*bonus)
    return sum(sims)/len(sims)*100

def total_open(shape:float, open_pct:float)->float:
    return SHAPE_WEIGHT_OPEN*shape + OPEN_WEIGHT_OPEN*open_pct

def total_fist(shape:float, open_pct:float)->float:
    return SHAPE_WEIGHT_FIST*shape + CLOSURE_WEIGHT_FIST*(100-open_pct)

def total_scissors(shape:float, open_pct:float)->float:
    return 0.5*shape + 0.5*(100-abs(open_pct-50))

def classify_rps(lm) -> str:
    tips   = [8,12,16,20]
    fingers= [1 if lm[t].y < lm[t-2].y else 0 for t in tips]
    c = sum(fingers)
    if c==0:    return "rock"
    if c==2:    return "scissors"
    if c>=4:    return "paper"
    return "unknown"

def judge(player:str, ai:str) -> str:
    if player == ai:
        return "Draw"
    win = (player=="rock" and ai=="scissors") or \
          (player=="scissors" and ai=="paper") or \
          (player=="paper" and ai=="rock")
    return "You Win!" if win else "You Lose!"


# ==== 后端主类 ====

class RPSBackend:
    def __init__(
        self,
        mode: str = MODE_LOCAL,
        host_ip: str = "",
        cam_idx: int = 0,
        templates: Optional[Dict] = None
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

        # 摄像头与 Mediapipe Hands
        self.cap   = cv2.VideoCapture(cam_idx)
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.templates = templates or load_templates()

        # 网络通信初始化
        self.sock = None
        self.conn = None
        if self.mode == MODE_HOST:
            self.sock = socket.socket()
            self.sock.bind(("", self.port))
            self.sock.listen(1)
            self.conn, _ = self.sock.accept()
        elif self.mode == MODE_CLIENT:
            self.sock = socket.socket()
            self.sock.connect((self.host, self.port))

    def start_round(self):
        self.round_id += 1

    def process_frame(self) -> Dict[str,Any]:
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Camera fail")
        frame = cv2.flip(frame,1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = self.hands.process(rgb)

        out = {"frame": frame, "gesture": None, "shape": 0.0, "open": 0.0, "total": None}

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            g  = classify_rps(lm)
            out["gesture"] = g

            # 如果识别到的是未知手势，就直接返回，不做后续模板匹配
            mapping = {"rock":"fist", "paper":"open", "scissors":"scissors"}
            if g not in mapping:
                return out

            # 已知手势，继续计算分数
            norm = normalize_landmarks(lm)
            base = mapping[g]
            tpl  = self.templates.get(base, {})
            devs = compute_devs(norm, tpl)
            shp  = shape_score(devs, base)
            op   = openness_ratio(lm) * 100

            if g == "rock":
                tot = total_fist(shp, op)
            elif g == "paper":
                tot = total_open(shp, op)
            else:  # scissors
                tot = total_scissors(shp, op)

            out.update(shape=shp, open=op, total=tot)

        return out

    def ai_choice(self, player: str) -> str:
        # Assist 模式
        if self.mode == MODE_LOCAL and self.score < -3:
            wr = max(ASSIST_MIN_WIN, ASSIST_INIT_WIN - 0.05*self.losses)
            dr = ASSIST_DRAW

        # High 模式
        elif self.mode == MODE_LOCAL and self.score >= 10:
            wr = min(HIGH_MAX_WIN, max(HIGH_MIN_WIN,
                     HIGH_INIT_WIN + 0.05*self.wins - 0.03*self.losses))
            dr = HIGH_DRAW

        else:
            return random.choice(["rock","paper","scissors"])

        # 网络模式走 net_exchange
        if self.mode in (MODE_HOST, MODE_CLIENT):
            return self.net_exchange(player)

        lr = 1 - wr - dr
        r  = random.random()
        if r < wr:
            # AI 出能打 player's 手势
            return {"rock":"paper","paper":"scissors","scissors":"rock"}.get(player,"rock")
        elif r < wr + dr:
            return player
        else:
            return {"rock":"scissors","paper":"rock","scissors":"paper"}.get(player,"rock")

    def net_exchange(self, player: str) -> str:
        msg = player.encode()
        if self.mode == MODE_HOST:
            self.conn.sendall(msg)
            opp = self.conn.recv(1024).decode()
        else:
            opp = self.sock.recv(1024).decode()
            self.sock.sendall(msg)
        return opp

    def end_round(self, player: str) -> str:
        # 获得对手动作
        if self.mode in (MODE_HOST, MODE_CLIENT):
            ai = self.net_exchange(player)
        else:
            ai = self.ai_choice(player)

        # 判定
        res = judge(player, ai)
        if res == "You Win!":
            self.wins += 1
            self.score += 3
        elif res == "You Lose!":
            self.losses += 1
            self.score -= 1
        else:
            self.draws += 1

        # 再捕获一帧数据用于日志
        last = self.process_frame()
        rec = [
            datetime.datetime.now().replace(microsecond=0).isoformat(),
            self.round_id, self.mode, player, ai, res,
            f"{last['total']:.2f}", f"{last['shape']:.2f}", f"{last['open']:.2f}",
            self.score, self.wins, self.losses, self.draws
        ]
        append_csv(rec)
        return res

    def release(self):
        self.cap.release()
        if self.conn:
            self.conn.close()
        if self.sock:
            self.sock.close()
        self.hands.close()
