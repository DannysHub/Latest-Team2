#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
version_4.4 Popup Ready Edition
================================
功能汇总(含新增弹窗机制)：
  * Rock / Paper / Scissors 手势识别 (规则: 手指伸展数量)
  * 仅对 Rock(Fist)/Paper(Open) 计算完成度 (Shape + Open/Closure)
  * 模板采集: 采集 5 指尖归一化坐标 (按 't')
  * 自适应难度: Assist(<-3)、Normal(-3~9)、High(>=10) 三段 + 动态 AI 赢率调整
  * 联机(Host/Client) + 握手 (HELLO_V1/WELCOME_V1) + 方向化手势交换
  * 失败降级: 网络异常自动退回本地 AI
  * CSV 回合日志: 完成度 + 模式状态
  * 新增: 回合结束后延迟3秒 -> 弹出“下一局准备”弹窗
       - 弹窗显示: Round N, (若 N>1 则显示上一局 Result + 上一局最佳完成度), 提示 "Press ENTER to start"
       - 联机模式: 双方都按 Enter 才继续; 一方先按后显示 Waiting 状态
       - 首回合不显示上一局信息
       - 全部用 ASCII 英文避免 OpenCV 中文变问号
使用说明:
  1) 运行脚本后选择模式 1/2/3
  2) 首回合: 若联机需握手 -> 弹出 Round 1 Ready 弹窗, 双方 Enter
  3) 采集阶段内可按 't' 更新模板 (仅 rock/paper)
  4) 回合结果显示后 3 秒 -> 自动进入下一局弹窗
  5) Enter 继续, ESC 随时退出
依赖: opencv-python, mediapipe, numpy
"""
from __future__ import annotations
import cv2, mediapipe as mp, random, time, os, csv, json, math, datetime, socket, sys
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional

# =============================================================================
# 1. 配置常量
# =============================================================================
TEMPLATE_FILE = "gesture_templates.json"
CSV_FILE      = "rps_open_fist_completion.csv"
SMOOTH_WINDOW = 5
CAPTURE_SECONDS = 5
PREP_SECONDS    = 2   # (保留: 采集前仍有微短准备动画; 首回合前由弹窗控制)

# 完成度权重
SHAPE_WEIGHT_OPEN   = 0.7
OPEN_WEIGHT_OPEN    = 0.3
SHAPE_WEIGHT_FIST   = 0.6
CLOSURE_WEIGHT_FIST = 0.4

DEVIATION_GREEN_THRESHOLD = 0.05
FINGERTIPS  = [4, 8, 12, 16, 20]
PALM_POINTS = [0, 5, 9, 13, 17]

# Assist 模式参数
ASSIST_ENTER_SCORE        = -3
ASSIST_EXIT_SCORE         = 3
ASSIST_AI_WIN_PROB_INITIAL= 0.30
ASSIST_AI_WIN_DEC         = 0.05
ASSIST_AI_WIN_PROB_MIN    = 0.20
ASSIST_DRAW_FIXED         = 0.30

# High 模式参数
HIGH_ENTER_SCORE          = 10
HIGH_AI_WIN_PROB_INITIAL  = 0.35
HIGH_AI_WIN_INC           = 0.05
HIGH_AI_WIN_DEC           = 0.03
HIGH_AI_WIN_MIN           = 0.30
HIGH_AI_WIN_MAX           = 0.45

# 网络
HOST_LISTEN = '0.0.0.0'
PORT = 65432
SOCKET_TIMEOUT = 10.0
READY_TIMEOUT  = 120.0  # 双方准备最大等待秒数
RESULT_DISPLAY_SECONDS = 3  # 结果界面停留秒数后弹窗

WINDOW_NAME = "RPS Completion Adaptive"

# =============================================================================
# 2. 模板管理
# =============================================================================
def load_templates() -> Dict[str, Dict[str, List[float]]]:
    if not os.path.exists(TEMPLATE_FILE):
        data = {"open": {}, "fist": {}}
        with open(TEMPLATE_FILE, 'w') as f: json.dump(data, f, indent=2)
        return data
    with open(TEMPLATE_FILE, 'r') as f:
        data = json.load(f)
    changed = False
    for k in ("open", "fist"):
        if k not in data:
            data[k] = {}; changed = True
    if changed:
        with open(TEMPLATE_FILE, 'w') as f: json.dump(data, f, indent=2)
    return data

def save_templates(tpl: Dict[str, Dict[str, List[float]]]) -> None:
    with open(TEMPLATE_FILE, 'w') as f: json.dump(tpl, f, indent=2)

gesture_templates = load_templates()

def has_template(name: str) -> bool:
    return name in gesture_templates and len(gesture_templates[name]) > 0

# =============================================================================
# 3. 几何 & 归一化
# =============================================================================
def palm_center_and_width(lm_list) -> Tuple[float, float, float]:
    cx = sum(lm_list[i].x for i in PALM_POINTS) / len(PALM_POINTS)
    cy = sum(lm_list[i].y for i in PALM_POINTS) / len(PALM_POINTS)
    w  = math.dist((lm_list[5].x, lm_list[5].y), (lm_list[17].x, lm_list[17].y)) or 1e-6
    return cx, cy, w

def normalize_landmarks(lm_list) -> Dict[int, Tuple[float,float]]:
    cx, cy, w = palm_center_and_width(lm_list)
    return {i: ((lm.x - cx)/w, (lm.y - cy)/w) for i, lm in enumerate(lm_list)}

def openness_ratio(lm_list) -> float:
    cx, cy, w = palm_center_and_width(lm_list)
    ds = [math.dist((lm_list[t].x, lm_list[t].y), (cx, cy))/w for t in FINGERTIPS]
    avg = sum(ds)/len(ds)
    return min(1.0, avg/0.9)

# =============================================================================
# 4. 完成度计算
# =============================================================================
def compute_devs(norm: Dict[int,Tuple[float,float]], template: Dict[str,List[float]]) -> Dict[int,float]:
    devs = {}
    for k, (tx, ty) in template.items():
        idx = int(k)
        if idx in norm:
            x, y = norm[idx]
            devs[idx] = math.hypot(x - tx, y - ty)
    return devs

def finger_weight(idx: int, base: str) -> float:
    if base == "open": return 1.2 if idx == 4 else 1.0
    if base == "fist": return 1.1 if idx in (8,12) else 1.0
    return 1.0

def shape_score(devs: Dict[int,float], base: str) -> float:
    if not devs: return 0.0
    sims = [math.exp(-8*d) * finger_weight(i, base) for i,d in devs.items()]
    return (sum(sims)/len(sims))*100.0

def total_open(shape_part: float, open_part: float) -> float:
    return SHAPE_WEIGHT_OPEN*shape_part + OPEN_WEIGHT_OPEN*open_part

def total_fist(shape_part: float, open_part: float) -> float:
    closure = 100 - open_part
    return SHAPE_WEIGHT_FIST*shape_part + CLOSURE_WEIGHT_FIST*closure

# =============================================================================
# 5. 手势分类
# =============================================================================
def classify_rps(lm_list) -> str:
    tips = [8,12,16,20]
    fingers = [1 if lm_list[t].y < lm_list[t-2].y else 0 for t in tips]
    c = sum(fingers)
    if c == 0: return "rock"
    if c == 2: return "scissors"
    if c >= 4: return "paper"
    return "unknown"

# =============================================================================
# 6. CSV 记录
# =============================================================================
def init_csv() -> None:
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE,'w',newline='') as f:
            csv.writer(f).writerow([
                "Time","Round","PlayerGesture","OpponentGesture","Result",
                "Total","Shape","Open","Score","Wins","Losses","Draws",
                "HighModeActive","HighWinProb","AssistActive","AssistWinProb"
            ])

def append_csv(ts: str, rnd: int, pg: str, og: str, res: str,
               total: Optional[float], shape: Optional[float], open_pct: Optional[float],
               score: int, w: int, l: int, d: int,
               high_active: bool, high_win_prob: float,
               assist_active: bool, assist_win_prob: float) -> None:
    with open(CSV_FILE,'a',newline='') as f:
        csv.writer(f).writerow([
            ts, rnd, pg, og, res,
            ("" if total is None else f"{total:.2f}"),
            ("" if shape is None else f"{shape:.2f}"),
            ("" if open_pct is None else f"{open_pct:.2f}"),
            score, w, l, d,
            int(high_active), f"{high_win_prob:.2f}" if high_active else "",
            int(assist_active), f"{assist_win_prob:.2f}" if assist_active else ""
        ])

# =============================================================================
# 7. 胜负判定
# =============================================================================
def judge(player: str, opp: str) -> str:
    if player == opp: return "Draw"
    if (player == "rock" and opp == "scissors") or \
       (player == "scissors" and opp == "paper") or \
       (player == "paper" and opp == "rock"):
        return "You Win!"
    return "You Lose!"

# =============================================================================
# 8. 网络工具 & 握手 / 交换
# =============================================================================
def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8",80))
        ip = s.getsockname()[0]
        s.close(); return ip
    except Exception:
        return "127.0.0.1"

def setup_host() -> Optional[socket.socket]:
    try:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind((HOST_LISTEN, PORT))
        srv.listen(1)
        ip = get_local_ip()
        print(f"[Host] 等待客户端连接... (IP: {ip}:{PORT})")
        conn, addr = srv.accept(); print(f"[Host] 客户端已连接: {addr}")
        return conn
    except Exception as e:
        print("[Host] 启动失败:", e)
        return None

def setup_client() -> Optional[socket.socket]:
    ip = input("输入主机IP: ").strip()
    try:
        cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print(f"[Client] 连接 {ip}:{PORT} ...")
        cli.connect((ip, PORT))
        print("[Client] 连接成功!")
        return cli
    except Exception as e:
        print("[Client] 连接失败:", e)
        return None

def handshake(sock: socket.socket, is_host: bool, timeout=5.0) -> bool:
    try:
        sock.settimeout(timeout)
        if is_host:
            data = sock.recv(16)
            if data.decode().strip() != "HELLO_V1":
                print("[Handshake] Unexpected data from client")
                return False
            sock.sendall(b"WELCOME_V1")
        else:
            sock.sendall(b"HELLO_V1")
            data = sock.recv(16)
            if data.decode().strip() != "WELCOME_V1":
                print("[Handshake] Unexpected data from host")
                return False
        print("[Handshake] OK")
        return True
    except Exception as e:
        print("[Handshake] Failed:", e)
        return False

def exchange_gesture_host(sock, my_gesture: str, timeout=5.0) -> Optional[str]:
    try:
        sock.settimeout(timeout)
        data = sock.recv(16)
        if not data:
            print("[Host] Peer closed")
            return None
        other = data.decode().strip()
        sock.sendall(my_gesture.encode())
        return other if other in ("rock","paper","scissors","None") else None
    except Exception as e:
        print("[Host] 交换失败:", e)
        return None

def exchange_gesture_client(sock, my_gesture: str, timeout=5.0) -> Optional[str]:
    try:
        sock.settimeout(timeout)
        sock.sendall(my_gesture.encode())
        data = sock.recv(16)
        if not data:
            print("[Client] Peer closed")
            return None
        other = data.decode().strip()
        return other if other in ("rock","paper","scissors","None") else None
    except Exception as e:
        print("[Client] 交换失败:", e)
        return None

# =============================================================================
# 9. Assist / High 模式 AI 决策
# =============================================================================
def assisted_ai_choice(player_gesture: str, ai_win_prob: float) -> str:
    p_ai = ai_win_prob
    p_draw = ASSIST_DRAW_FIXED
    p_player = max(0.0, 1.0 - (p_ai + p_draw))
    r = random.random()
    if player_gesture == 'rock':
        if r < p_ai: return 'paper'
        if r < p_ai + p_draw: return 'rock'
        return 'scissors'
    if player_gesture == 'paper':
        if r < p_ai: return 'scissors'
        if r < p_ai + p_draw: return 'paper'
        return 'rock'
    if player_gesture == 'scissors':
        if r < p_ai: return 'rock'
        if r < p_ai + p_draw: return 'scissors'
        return 'paper'
    return random.choice(["rock","paper","scissors"])  # fallback

def highmode_ai_choice(player_gesture: str, ai_win_prob: float) -> str:
    p_ai = ai_win_prob
    remain = 1.0 - p_ai
    p_draw = remain / 2.0
    r = random.random()
    if player_gesture == 'rock':
        if r < p_ai: return 'paper'
        if r < p_ai + p_draw: return 'rock'
        return 'scissors'
    if player_gesture == 'paper':
        if r < p_ai: return 'scissors'
        if r < p_ai + p_draw: return 'paper'
        return 'rock'
    if player_gesture == 'scissors':
        if r < p_ai: return 'rock'
        if r < p_ai + p_draw: return 'scissors'
        return 'paper'
    return random.choice(["rock","paper","scissors"])  # fallback

# =============================================================================
# 10. 双方准备同步弹窗 / Ready Synchronization
# =============================================================================
READY_MSG_LOCAL = "Press ENTER to start"
READY_MSG_WAIT  = "Waiting peer..."
READY_MSG_BOTH  = "Both ready!"

def wait_both_ready(window_name, sock, is_host, remote_mode,
                    label="ROUND", timeout=READY_TIMEOUT, extra_lines: Optional[List[str]]=None) -> bool:
    start_wait = time.time()
    local_ready = False
    remote_ready = False
    if not remote_mode:
        # 本地模式直接展示一次即可
        while True:
            frame = np.zeros((360,720,3), dtype='uint8')
            lines = extra_lines if extra_lines else [label, READY_MSG_LOCAL]
            y = 80
            for i,t in enumerate(lines):
                color = (255,255,255) if i < len(lines)-1 else (0,255,255)
                cv2.putText(frame, t, (60,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                y += 70
            cv2.imshow(window_name, frame)
            k = cv2.waitKey(50) & 0xFF
            if k == 27: return False
            if k == 13: return True
        return True
    # 联机模式
    sock.settimeout(0.05)
    SENT_READY = False
    while True:
        now = time.time()
        if now - start_wait > timeout:
            print("[ReadySync] Timeout")
            return False
        # 键盘
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            return False
        if k == 13:
            local_ready = True
        # 发送本地 ready 信号 (一次)
        if local_ready and (not SENT_READY):
            try:
                sock.sendall(b"RDY")
                SENT_READY = True
            except Exception as e:
                print("[ReadySync] send failed:", e)
                return False
        # 接收对方 ready
        try:
            data = sock.recv(8)
            if data:
                if data.startswith(b"RDY"):
                    remote_ready = True
        except socket.timeout:
            pass
        except Exception as e:
            print("[ReadySync] recv failed:", e)
            return False
        # 绘制
        frame = np.zeros((360,720,3), dtype='uint8')
        lines = extra_lines if extra_lines else [label]
        y = 60
        for t in lines:
            cv2.putText(frame, t, (60,y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255),2)
            y += 65
        status = READY_MSG_BOTH if (local_ready and remote_ready) else \
                 (READY_MSG_WAIT if local_ready else READY_MSG_LOCAL)
        cv2.putText(frame, status, (60,y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255),2)
        cv2.imshow(window_name, frame)
        if local_ready and remote_ready:
            # 小延时让双方看到 Both ready 文本
            time.sleep(0.4)
            return True

# 弹出下一局准备弹窗（本地或联机）
def show_next_round_popup(window_name, upcoming_round,
                          last_result, last_best_total,
                          is_network, sock, is_host,
                          timeout=120.0, extra_lines=None):
    """
    弹出准备下一局界面。
    - upcoming_round: 即将开始的回合号
    - last_result: 上一局结果 ("You Win!"/"You Lose!"/"Draw") 或 None
    - last_best_total: 上一局完成度百分比 或 None
    - is_network: 联机模式
    - sock, is_host: 网络同步
    - timeout: 等待超时时间
    - extra_lines: 可选额外文本行，优先使用
    返回: True=继续, False=退出/取消
    """
    import numpy as _np
    READY_PROMPT = "Press ENTER to start"
    START_LABEL = f"Round {upcoming_round}"
    # 构造文本行
    lines = [START_LABEL]
    if upcoming_round > 1:
        if last_result:
            if "Win" in last_result:
                res = "Win"
            elif "Lose" in last_result:
                res = "Lose"
            elif "Draw" in last_result:
                res = "Draw"
            else:
                res = last_result
            lines.append(f"Last Result: {res}")
        if last_best_total is not None:
            lines.append(f"Last Best Total: {last_best_total:.1f}%")
    # 本地模式下添加按键提示，联机模式使用 wait_both_ready 自身提示
    if not is_network:
        lines.append(READY_PROMPT)
    # 联机模式等待双方
    if is_network and sock:
        return wait_both_ready(window_name, sock, is_host, True,
                               label=START_LABEL,
                               extra_lines=lines,
                               timeout=timeout)
    # 本地模式
    while True:
        frame = _np.zeros((360, 720, 3), dtype='uint8')
        y = 80
        for i, text in enumerate(lines):
            color = (255,255,255)
            if text == READY_PROMPT:
                color = (0,255,255)
            cv2.putText(frame, text, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            y += 70
        cv2.imshow(window_name, frame)
        k = cv2.waitKey(50) & 0xFF
        if k == 27:
            return False
        if k == 13:
            return True
# =============================================================================
# 11. 主循环
# =============================================================================
def main() -> None:
    print("==== 模式选择 / Mode Select ====")
    print("1) 本地 vs AI")
    print("2) 联机 (Host)")
    print("3) 联机 (Client)")
    mode = input(">>> ").strip()

    remote_mode = False
    net_sock: Optional[socket.socket] = None
    is_host = False
    if mode == '2':
        net_sock = setup_host(); remote_mode = net_sock is not None; is_host = True
    elif mode == '3':
        net_sock = setup_client(); remote_mode = net_sock is not None; is_host = False
    else:
        print("进入单人 AI 模式 ...")

    if mode in ('2','3') and not remote_mode:
        print("联机失败, 回退单人模式。")

    if remote_mode:
        if not handshake(net_sock, is_host):
            print("握手失败, 回退本地模式")
            remote_mode = False

    init_csv()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    wins = losses = draws = 0
    score = 0
    round_id = 0  # 已完成回合数

    assist_active = False
    assist_ai_win_prob = ASSIST_AI_WIN_PROB_INITIAL
    high_active = False
    high_ai_win_prob = HIGH_AI_WIN_PROB_INITIAL

    last_result: Optional[str] = None
    last_best_total: Optional[float] = None

    # 首次弹窗 (Round 1)
    ok = show_next_round_popup(1, None, None, remote_mode, net_sock, is_host)
    if not ok:
        cap.release();
        if net_sock: net_sock.close()
        cv2.destroyAllWindows(); return

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5) as hands:
        while True:
            # 回合编号+1 (开始新一局)
            round_id += 1

            # ---- 模式切换优先级 ----
            if not remote_mode and score >= HIGH_ENTER_SCORE:
                if not high_active:
                    high_active = True
                    high_ai_win_prob = HIGH_AI_WIN_PROB_INITIAL
                    if assist_active:
                        assist_active = False
                    print(f"[High] 激活 score={score} win_prob={high_ai_win_prob:.2f}")
            else:
                if high_active and score < HIGH_ENTER_SCORE:
                    high_active = False
                    print(f"[High] 关闭 score={score}")
            if (not remote_mode) and (not high_active) and (score < ASSIST_ENTER_SCORE):
                if not assist_active:
                    assist_active = True
                    assist_ai_win_prob = ASSIST_AI_WIN_PROB_INITIAL
                    print(f"[Assist] 激活 score={score} win_prob={assist_ai_win_prob:.2f}")
            if assist_active and score >= ASSIST_EXIT_SCORE:
                assist_active = False
                print(f"[Assist] 关闭 score={score}")

            # ---- 准备阶段 (短倒计时保留视觉动画) ----
            prep_start = time.time()
            while time.time() - prep_start < PREP_SECONDS:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                remain = PREP_SECONDS - int(time.time() - prep_start)
                cv2.putText(frame, f"Round {round_id} Ready:{remain}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
                status = f"Score:{score} W:{wins} L:{losses} D:{draws}"
                if high_active:
                    status += f" | High winP={high_ai_win_prob:.2f}"
                elif assist_active:
                    status += f" | Assist winP={assist_ai_win_prob:.2f}"
                if remote_mode:
                    status += " | NET"
                cv2.putText(frame, status, (10,470), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255,255,255), 2)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release();
                    if net_sock: net_sock.close()
                    cv2.destroyAllWindows(); return

            # ---- 采集阶段 ----
            capture_start = time.time()
            player_gesture = "None"
            gesture_captured = False
            best_total: Optional[float] = None
            best_shape: Optional[float] = None
            best_open: Optional[float] = None
            total_window: deque = deque(maxlen=SMOOTH_WINDOW)

            while time.time() - capture_start < CAPTURE_SECONDS:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    lm_list = results.multi_hand_landmarks[0].landmark
                    cur = classify_rps(lm_list)
                    if cur in ("rock","paper","scissors"):
                        player_gesture = cur; gesture_captured = True

                    # 模板采集
                    ktemp = cv2.waitKey(1) & 0xFF
                    if ktemp == ord('t') and player_gesture in ("rock","paper"):
                        base = 'fist' if player_gesture == 'rock' else 'open'
                        norm_cap = normalize_landmarks(lm_list)
                        gesture_templates[base] = {str(i): norm_cap[i] for i in FINGERTIPS}
                        save_templates(gesture_templates)
                        print(f"[Template Updated] {base}")

                    # 完成度
                    if player_gesture in ("rock","paper"):
                        base = 'fist' if player_gesture == 'rock' else 'open'
                        norm = normalize_landmarks(lm_list)
                        if has_template(base):
                            devs = compute_devs(norm, gesture_templates[base])
                            shape = shape_score(devs, base)
                        else:
                            devs = {}; shape = 0.0
                        open_pct = openness_ratio(lm_list)*100
                        total_raw = total_fist(shape, open_pct) if base=='fist' else total_open(shape, open_pct)
                        total_window.append(total_raw)
                        total_disp = sum(total_window)/len(total_window)
                        if (best_total is None) or (total_disp > best_total):
                            best_total = total_disp; best_shape = shape; best_open = open_pct
                        y0 = 180
                        if devs:
                            for tip, dval in devs.items():
                                color = (0,255,0) if dval < DEVIATION_GREEN_THRESHOLD else (0,0,255)
                                cv2.putText(frame, f"{tip}:{dval:.3f}", (20,y0), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                                y0 += 18
                        closure = 100 - open_pct
                        cv2.putText(frame, f"Shape:{shape:.1f}% Open:{open_pct:.1f}% Clo:{closure:.1f}%", (20,140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,255), 2)
                        cv2.putText(frame, f"Total:{total_disp:.1f}% (Best:{best_total:.1f}%)", (20,115), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
                    else:
                        cv2.putText(frame, "Completion: N/A (scissors)", (20,140), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 2)

                    mp.solutions.drawing_utils.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

                remain = CAPTURE_SECONDS - int(time.time() - capture_start)
                cv2.putText(frame, f"Show ({remain})", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.putText(frame, f"Gesture:{player_gesture}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                footer = "[t]template rock/paper  ESC quit"
                if high_active:
                    footer += f"  High winP={high_ai_win_prob:.2f}"
                elif assist_active:
                    footer += f"  Assist winP={assist_ai_win_prob:.2f}"
                if remote_mode:
                    footer += "  NET"
                cv2.putText(frame, footer, (200,470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release();
                    if net_sock: net_sock.close()
                    cv2.destroyAllWindows(); return

            # ---- AI / 对手手势 ----
            if remote_mode and net_sock:
                if is_host:
                    opp_gesture = exchange_gesture_host(net_sock, player_gesture if gesture_captured else "None")
                else:
                    opp_gesture = exchange_gesture_client(net_sock, player_gesture if gesture_captured else "None")
                if opp_gesture is None:
                    print("[Network] 通信异常 -> 本地随机 AI")
                    remote_mode = False
                    opp_gesture = random.choice(["rock","paper","scissors"])
            else:
                if not gesture_captured or player_gesture not in ("rock","paper","scissors"):
                    player_gesture = "None"
                    opp_gesture = random.choice(["rock","paper","scissors"])
                else:
                    if high_active:
                        opp_gesture = highmode_ai_choice(player_gesture, high_ai_win_prob)
                    elif assist_active:
                        opp_gesture = assisted_ai_choice(player_gesture, assist_ai_win_prob)
                    else:
                        opp_gesture = random.choice(["rock","paper","scissors"])

            # ---- 判定与计分 ----
            if player_gesture in ("rock","paper","scissors"):
                result = judge(player_gesture, opp_gesture)
                if result == "You Win!":
                    score += 3; wins += 1
                    if high_active:
                        high_ai_win_prob = min(HIGH_AI_WIN_MAX, high_ai_win_prob + HIGH_AI_WIN_INC)
                elif result == "You Lose!":
                    score -= 1; losses += 1
                    if assist_active:
                        assist_ai_win_prob = max(ASSIST_AI_WIN_PROB_MIN, assist_ai_win_prob - ASSIST_AI_WIN_DEC)
                    if high_active:
                        high_ai_win_prob = max(HIGH_AI_WIN_MIN, high_ai_win_prob - HIGH_AI_WIN_DEC)
                else:
                    draws += 1
            else:
                result = "No gesture!"; player_gesture = "None"

            # ---- 回合结束边界模式检查 ----
            if high_active and score < HIGH_ENTER_SCORE:
                high_active = False
                print(f"[High] 关闭 score={score}")
            if assist_active and score >= ASSIST_EXIT_SCORE:
                assist_active = False
                print(f"[Assist] 关闭 score={score}")
            if (not high_active) and (not assist_active):
                if score >= HIGH_ENTER_SCORE:
                    high_active = True
                    high_ai_win_prob = HIGH_AI_WIN_PROB_INITIAL
                    print(f"[High] 激活 score={score} win_prob={high_ai_win_prob:.2f}")
                elif score < ASSIST_ENTER_SCORE:
                    assist_active = True
                    assist_ai_win_prob = ASSIST_AI_WIN_PROB_INITIAL
                    print(f"[Assist] 激活 score={score} win_prob={assist_ai_win_prob:.2f}")

            # ---- 结果显示 ----
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            show_total = best_total if player_gesture in ("rock","paper") else None
            show_shape = best_shape if player_gesture in ("rock","paper") else None
            show_open  = best_open  if player_gesture in ("rock","paper") else None
            cv2.putText(frame, f"You: {player_gesture}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            who = "Opponent" if remote_mode else "AI"
            cv2.putText(frame, f"{who}: {opp_gesture}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, result, (10,125), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
            if show_total is not None:
                cv2.putText(frame, f"Best Total:{show_total:.1f}% Shape:{show_shape:.1f}% Open:{show_open:.1f}%", (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            else:
                cv2.putText(frame, "Best Total: N/A", (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
            status = f"Score:{score} W:{wins} L:{losses} D:{draws}"
            if high_active:
                status += f" | High winP={high_ai_win_prob:.2f}"
            elif assist_active:
                status += f" | Assist winP={assist_ai_win_prob:.2f}"
            if remote_mode:
                status += " | NET"
            cv2.putText(frame, status, (10,470), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            cv2.imshow(WINDOW_NAME, frame)

            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_csv(ts, round_id, player_gesture, opp_gesture, result,
                       show_total, show_shape, show_open, score, wins, losses, draws,
                       high_active and (not remote_mode), high_ai_win_prob if high_active else 0.0,
                       assist_active and (not remote_mode), assist_ai_win_prob if assist_active else 0.0)

            # ---- 结果停留 + 进入下一局弹窗 ----
            time.sleep(RESULT_DISPLAY_SECONDS)
            last_result = result if result not in ("No gesture!","No gesture") else None
            last_best_total = show_total
            # 弹出下一局准备 (下一局编号 = round_id + 1)
            cont = show_next_round_popup(round_id+1, last_result, last_best_total,
                                         remote_mode, net_sock, is_host)
            if not cont:
                break

    cap.release()
    if net_sock: net_sock.close()
    cv2.destroyAllWindows()

# =============================================================================
# 入口
# =============================================================================
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
