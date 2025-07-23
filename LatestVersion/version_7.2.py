#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
version_8_full.py

整合 version_6.4 与 version_7.1，补全所有功能：
  * 模式选择弹窗 (Local / Host / Client)
  * Host 等待连接弹窗
  * Client 输入 IP 弹窗 + 连接失败弹窗 (Retry/Back)
  * 网络握手 & 错误提示
  * 回合开始/结束弹窗，同步本地与网络
  * MediaPipe Hands 手势识别 + 关键点绘制
  * 模板采集 (按 't')
  * 手势完成度计算：shape, open, total
  * 指尖偏差数值叠加
  * 屏幕上 shape/open/total 文本展示
  * 自适应难度 (Assist/Normal/High)
  * 回合结果展示 3 秒
  * CSV 记录：Time, Round, Mode, Gestures, Result, Total, Shape, Open, Score, Wins, Losses, Draws, HighActive, HighProb, AssistActive, AssistProb
"""
import cv2
import mediapipe as mp
import random
import time
import os
import csv
import json
import math
import datetime
import socket
import sys
import numpy as np
from collections import deque

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
TEMPLATE_FILE = "gesture_templates.json"
CSV_FILE      = "rps_open_fist_completion.csv"
SMOOTH_WINDOW = 5
CAPTURE_SECONDS = 5
PREP_SECONDS    = 2
DEVIATION_GREEN_THRESHOLD = 0.05

# 自适应难度参数
ASSIST_ENTER_SCORE         = -3
ASSIST_EXIT_SCORE          = 3
ASSIST_AI_WIN_PROB_INITIAL = 0.30
ASSIST_AI_WIN_DEC          = 0.05
ASSIST_AI_WIN_PROB_MIN     = 0.20
ASSIST_DRAW_FIXED          = 0.30

HIGH_ENTER_SCORE           = 10
HIGH_AI_WIN_PROB_INITIAL   = 0.35
HIGH_AI_WIN_INC            = 0.05
HIGH_AI_WIN_DEC            = 0.03
HIGH_AI_WIN_MIN            = 0.30
HIGH_AI_WIN_MAX            = 0.45

HOST_LISTEN = '0.0.0.0'
PORT = 65432
WINDOW_NAME = "RPS Completion Adaptive"

# -----------------------------------------------------------------------------
# 模板管理 & CSV 日志
# -----------------------------------------------------------------------------

def load_templates():
    if not os.path.exists(TEMPLATE_FILE):
        tpl = {"open": {}, "fist": {}, "scissors": {}}
        with open(TEMPLATE_FILE, 'w') as f:
            json.dump(tpl, f, indent=2)
        return tpl
    with open(TEMPLATE_FILE, 'r') as f:
        return json.load(f)


def save_templates(tpl):
    with open(TEMPLATE_FILE, 'w') as f:
        json.dump(tpl, f, indent=2)


gesture_templates = load_templates()

def has_template(name):
    return name in gesture_templates and len(gesture_templates[name]) > 0


def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE, 'w', newline='') as f:
            csv.writer(f).writerow([
                "Time", "Round", "Mode", "PlayerGesture", "OpponentGesture", "Result",
                "Total", "Shape", "Open", "Score", "Wins", "Losses", "Draws",
                "HighActive", "HighProb", "AssistActive", "AssistProb"
            ])


def append_csv(round_id, mode, pg, og, res, total, shape, open_pct,
               score, wins, losses, draws, high_active, high_prob, assist_active, assist_prob):
    with open(CSV_FILE, 'a', newline='') as f:
        csv.writer(f).writerow([
            datetime.datetime.now().isoformat(), round_id, mode, pg, og, res,
            f"{total:.2f}", f"{shape:.2f}", f"{open_pct:.2f}",
            score, wins, losses, draws,
            int(high_active), f"{high_prob:.2f}" if high_active else "",
            int(assist_active), f"{assist_prob:.2f}" if assist_active else ""
        ])

# -----------------------------------------------------------------------------
# 手势完成度计算辅助
# -----------------------------------------------------------------------------
FINGERTIPS = [4, 8, 12, 16, 20]
PALM_POINTS = [0, 5, 9, 13, 17]


def palm_center_and_width(lm):
    cx = sum(lm[i].x for i in PALM_POINTS) / len(PALM_POINTS)
    cy = sum(lm[i].y for i in PALM_POINTS) / len(PALM_POINTS)
    w = math.dist((lm[5].x, lm[5].y), (lm[17].x, lm[17].y)) or 1e-6
    return cx, cy, w


def normalize_landmarks(lm):
    cx, cy, w = palm_center_and_width(lm)
    return {i: ((pt.x - cx) / w, (pt.y - cy) / w) for i, pt in enumerate(lm)}


def openness_ratio(lm):
    cx, cy, w = palm_center_and_width(lm)
    ds = [math.dist((lm[t].x, lm[t].y), (cx, cy)) / w for t in FINGERTIPS]
    return min(1.0, sum(ds) / len(ds) / 0.9)


def shape_score(devs, base):
    if not devs:
        return 0.0
    sims = []
    for i, d in devs.items():
        bonus = 1.2 if base == 'open' and i == 4 else 1.1 if base == 'fist' and i in (8, 12) else 1.0
        sims.append(math.exp(-8 * d) * bonus)
    return sum(sims) / len(sims) * 100


def total_score(base, shape, open_pct):
    if base == 'rock':
        return 0.6 * shape + 0.4 * (100 - open_pct)
    if base == 'paper':
        return 0.7 * shape + 0.3 * open_pct
    return 0.5 * shape + 0.5 * (100 - abs(open_pct - 50))


def compute_devs(norm, tpl):
    devs = {}
    for k, v in tpl.items():
        idx = int(k)
        if idx in norm:
            devs[idx] = math.hypot(norm[idx][0] - v[0], norm[idx][1] - v[1])
    return devs

# -----------------------------------------------------------------------------
# 手势分类 & 判定
# -----------------------------------------------------------------------------

def classify_rps(lm):
    # 使用指尖关键点 8,12,16,20 进行分类，不包含拇指
    tips = [8, 12, 16, 20]
    # 手指伸展：当指尖高于对应指根时认为伸展
    fingers = [1 if lm[t].y < lm[t - 2].y else 0 for t in tips]
    c = sum(fingers)
    if c == 0:
        return 'rock'
    if c == 2:
        return 'scissors'
    if c == 4:
        return 'paper'
    return 'unknown'


def judge(player, opp):
    if player == opp:
        return 'Draw'
    win = (player == 'rock' and opp == 'scissors') or \
          (player == 'scissors' and opp == 'paper') or \
          (player == 'paper' and opp == 'rock')
    return 'You Win!' if win else 'You Lose!'

# -----------------------------------------------------------------------------
# 弹窗：模式选择 / Host / Client
# -----------------------------------------------------------------------------

def popup_mode_selection(title="Mode Selection"):
    opts = ["Local", "Host", "Client"]
    idx = 0
    while True:
        frm = np.zeros((300, 600, 3), dtype='uint8')
        cv2.putText(frm, "Select Mode:", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        for i, o in enumerate(opts):
            col = (0, 255, 255) if i == idx else (255, 255, 255)
            text = "> " + o if i == idx else "  " + o
            cv2.putText(frm, text, (50, 100 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cv2.putText(frm, "W/S: move   Enter: select   Esc: quit", (30, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow(title, frm)
        k = cv2.waitKey(100) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit(0)
        if k in (ord('w'), ord('W'), 82):
            idx = (idx - 1) % len(opts)
        if k in (ord('s'), ord('S'), 84):
            idx = (idx + 1) % len(opts)
        if k in (13, 10):
            cv2.destroyWindow(title)
            return idx + 1

# -----------------------------------------------------------------------------
# Host 等待连接弹窗
# -----------------------------------------------------------------------------

def setup_host():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((HOST_LISTEN, PORT))
    srv.listen(1)
    ip = get_local_ip()
    title = "Waiting for Client"
    while True:
        frm = np.zeros((240, 500, 3), dtype='uint8')
        cv2.putText(frm, "Waiting for Connection...", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frm, f"IP: {ip}:{PORT}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(title, frm)
        srv.settimeout(0.5)
        try:
            conn, addr = srv.accept()
            cv2.destroyWindow(title)
            return conn
        except socket.timeout:
            if cv2.waitKey(1) & 0xFF == 27:
                cv2.destroyWindow(title)
                sys.exit(0)

# -----------------------------------------------------------------------------
# Client 输入 IP & 连接失败弹窗
# -----------------------------------------------------------------------------

def client_error_popup():
    opts = ["Retry", "Back"]
    idx = 0
    title = "Connection Failed"
    while True:
        frm = np.zeros((180, 400, 3), dtype='uint8')
        cv2.putText(frm, "❌ Connection Failed", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        for i, o in enumerate(opts):
            col = (0, 255, 255) if i == idx else (255, 255, 255)
            cv2.putText(frm, "> " + o, (50, 100 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)
        cv2.imshow(title, frm)
        k = cv2.waitKey(100) & 0xFF
        if k in (ord('w'), ord('W'), 82):
            idx = (idx - 1) % len(opts)
        if k in (ord('s'), ord('S'), 84):
            idx = (idx + 1) % len(opts)
        if k in (13, 10):
            cv2.destroyWindow(title)
            return opts[idx]


def setup_client():
    import time
    while True:
        ip_chars = ""
        cursor = True
        last = time.time()
        win = "Enter Host IP"
        while True:
            frm = np.zeros((200, 600, 3), dtype='uint8')
            cv2.putText(frm, "Enter Host IP:", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            disp = ip_chars + ("_" if cursor else "")
            cv2.putText(frm, disp, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frm, "Enter: OK   Back: ESC   Del: Backspace", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
            cv2.imshow(win, frm)
            if time.time() - last > 0.5:
                cursor = not cursor
                last = time.time()
            k = cv2.waitKey(100)
            if k == 27:
                cv2.destroyWindow(win)
                return None
            if k in (13, 10):
                break
            if k in (8, 127):
                ip_chars = ip_chars[:-1]
            elif 32 <= k < 127:
                ip_chars += chr(k)
        cv2.destroyWindow(win)
        try:
            cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            cli.connect((ip_chars.strip(), PORT))
            return cli
        except:
            choice = client_error_popup()
            if choice == "Back":
                return None
            # Retry继续
            
# -----------------------------------------------------------------------------
# 回合同步与弹窗
# -----------------------------------------------------------------------------

def wait_both_ready(name, sock, is_host, network, lines, timeout=120):
    start = time.time()
    local_ready = False
    peer_ready = False
    while True:
        frame = np.zeros((300, 640, 3), dtype='uint8')
        for i, t in enumerate(lines):
            cv2.putText(frame, t, (20, 60 + i*40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow(name, frame)
        k = cv2.waitKey(100) & 0xFF
        if k in (13, 10) and not local_ready:
            local_ready = True
            if network and sock:
                try:
                    sock.sendall(b"READY")
                except:
                    pass
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit(0)
        if local_ready and (not network or peer_ready):
            return True
        if network and not peer_ready:
            try:
                data = sock.recv(16)
                if data.decode() == "READY":
                    peer_ready = True
            except:
                pass
        if time.time() - start > timeout:
            return False


def show_next_round_popup(name, rnd, score, last_res, last_best, network, sock, is_host):
    lines = [
        f"Round {rnd}",
        f"Score: {score}",
        "Press ENTER to start",
        "Press ESC to quit"
    ]
    return wait_both_ready(name, sock, is_host, network, lines)

# -----------------------------------------------------------------------------
# AI 策略
# -----------------------------------------------------------------------------
def assisted_ai_choice(player, ai_prob):
    p_ai = ai_prob
    p_draw = ASSIST_DRAW_FIXED
    r = random.random()
    mapping = {
        'rock': ('paper','rock','scissors'),
        'paper': ('scissors','paper','rock'),
        'scissors': ('rock','scissors','paper'),
    }
    a,p,s = mapping.get(player,('rock','paper','scissors'))
    if r < p_ai:
        return a
    if r < p_ai + p_draw:
        return p
    return s


def highmode_ai_choice(player, ai_prob):
    p_ai = ai_prob
    rem = 1 - p_ai
    p_draw = rem / 2
    r = random.random()
    mapping = {
        'rock': ('paper','rock','scissors'),
        'paper': ('scissors','paper','rock'),
        'scissors': ('rock','scissors','paper'),
    }
    a,p,s = mapping.get(player,('rock','paper','scissors'))
    if r < p_ai:
        return a
    if r < p_ai + p_draw:
        return p
    return s

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return '127.0.0.1'

# -----------------------------------------------------------------------------
# 主流程
# -----------------------------------------------------------------------------
def main():
    # 选择模式
    mode = popup_mode_selection()
    sock = None
    is_host = False
    network = False

    if mode == 2:
        sock = setup_host()
        is_host = True
        network = sock is not None
    elif mode == 3:
        cli = setup_client()
        if cli is None:
            return main()
        sock = cli
        network = True

    init_csv()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1,
                            min_detection_confidence=0.7,
                            min_tracking_confidence=0.5)

    wins = losses = draws = 0
    score = 0
    rnd = 0
    assist_active = False
    assist_prob = ASSIST_AI_WIN_PROB_INITIAL
    high_active = False
    high_prob = HIGH_AI_WIN_PROB_INITIAL
    last_res = None
    last_best = None

    while True:
        rnd += 1
        # 回合弹窗
        cont = show_next_round_popup(
            WINDOW_NAME, rnd, score, last_res or "", last_best or 0.0,
            network, sock, is_host
        )
        if not cont:
            break

        # 难度切换
        if score >= HIGH_ENTER_SCORE and not high_active:
            high_active = True
            high_prob = HIGH_AI_WIN_PROB_INITIAL
            assist_active = False
        if high_active and score < HIGH_ENTER_SCORE:
            high_active = False
        if score < ASSIST_ENTER_SCORE and not assist_active and not high_active:
            assist_active = True
            assist_prob = ASSIST_AI_WIN_PROB_INITIAL
        if assist_active and score >= ASSIST_EXIT_SCORE:
            assist_active = False

        # 准备阶段
        start_prep = time.time()
        while time.time() - start_prep < PREP_SECONDS:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            remain = int(PREP_SECONDS - (time.time() - start_prep))
            cv2.putText(frame, f"Ready: {remain}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                return

        # 采集阶段
        frame_buf = deque(maxlen=SMOOTH_WINDOW)
        best_total = 0.0
        best_shape = 0.0
        best_open = 0.0
        player_gesture = "None"
        capture_start = time.time()

        while time.time() - capture_start < CAPTURE_SECONDS:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark
                # 分类
                g = classify_rps(lm)
                if g in ("rock", "paper", "scissors"):
                    player_gesture = g
                # 模板采集
                if cv2.waitKey(1) & 0xFF == ord('t') and player_gesture != "None":
                    key_map = {"rock": "fist", "paper": "open", "scissors": "scissors"}
                    norm = normalize_landmarks(lm)
                    gesture_templates[key_map[g]] = {str(i): norm[i] for i in FINGERTIPS}
                    save_templates(gesture_templates)
                # 计算完成度
                norm = normalize_landmarks(lm)
                base_map = {"rock": "fist", "paper": "open", "scissors": "scissors"}
                base = base_map.get(g, None)
                shape_val = 0.0
                open_val = openness_ratio(lm) * 100
                if base and has_template(base):
                    devs = compute_devs(norm, gesture_templates[base])
                    shape_val = shape_score(devs, base)
                    # 绘制偏差
                    y0 = 180
                    for tip, dval in devs.items():
                        color = (0, 255, 0) if dval < DEVIATION_GREEN_THRESHOLD else (0, 0, 255)
                        cv2.putText(frame, f"{tip}:{dval:.3f}", (20, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        y0 += 20
                total_val = total_score(g, shape_val, open_val)
                frame_buf.append(total_val)
                avg_total = sum(frame_buf) / len(frame_buf)
                if avg_total > best_total:
                    best_total, best_shape, best_open = avg_total, shape_val, open_val
                # 绘制完成度
                cv2.putText(frame, f"Shape:{shape_val:.1f}% Open:{open_val:.1f}% Total:{avg_total:.1f}%", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                # 绘制关键点
                mp.solutions.drawing_utils.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == 27:
                return

        # AI 对手/网络交换
        if player_gesture in ("rock", "paper", "scissors"):
            if network:
                try:
                    sock.sendall(player_gesture.encode())
                    opp = sock.recv(16).decode().strip()
                except:
                    opp = random.choice(["rock", "paper", "scissors"] )
            else:
                if high_active:
                    opp = highmode_ai_choice(player_gesture, high_prob)
                elif assist_active:
                    opp = assisted_ai_choice(player_gesture, assist_prob)
                else:
                    opp = random.choice(["rock", "paper", "scissors"] )
            res_text = judge(player_gesture, opp)
            if res_text == "You Win!":
                score += 3
                wins += 1
            elif res_text == "You Lose!":
                score -= 1
                losses += 1
            else:
                draws += 1
        else:
            opp = "None"
            res_text = "No gesture!"

        # 结果显示
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, f"You: {player_gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Opp: {opp}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, res_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        cv2.imshow(WINDOW_NAME, frame)
        cv2.waitKey(3000)

        # 日志
        append_csv(rnd, "Local" if mode==1 else ("Host" if mode==2 else "Client"),
                   player_gesture, opp, res_text,
                   best_total, best_shape, best_open,
                   score, wins, losses, draws,
                   high_active, high_prob, assist_active, assist_prob)
        last_res = res_text
        last_best = best_total

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
