#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPS (Rock / Paper / Scissors) + Completion (Open/Fist) + Network & Cheat AI
关闭型作弊模式 + 动态低分扶持/高分加压概率调节
================================================================================
本版本根据你的最新要求新增：
  1. **低分区(< -3) 动态降低 AI 胜率**：玩家分数 <= -3 时进入“扶持模式”。
       * 维护一个 `assist_ai_win_prob` （AI 赢概率），初始为 `ASSIST_AI_WIN_PROB_INITIAL` (默认 0.30)。
       * 每当玩家再输一局 (AI 获胜) 时，`assist_ai_win_prob -= ASSIST_AI_WIN_DEC` (默认 0.05)。
       * 设下限 `ASSIST_AI_WIN_PROB_MIN` (默认 0.05)。
       * 平局概率固定 `ASSIST_AI_DRAW_PROB` (默认 0.35)。
       * 剩余概率自动分配给玩家获胜（玩家在扶持区越输越容易赢，形成扶持梯度）。
       * 当玩家分数回到  > -3 (即 >= -2) 时退出扶持模式，并在下次再次跌回 <= -3 时重新按初始值进入。
  2. **高分区(>=10) 动态增加 AI 胜率**：作弊模式激活条件仍为 `score >= 10`。
       * 在作弊模式下，每当玩家 *获胜* 一局（玩家胜）就提高 AI 的“强制克制概率” `cheat_prob += CHEAT_PLAYER_WIN_INC` (默认 0.05)；
         上限仍为 `CHEAT_PROB_MAX`。
       * 原先“只有触发到一次强制克制才增长”的机制也保留（即：1) 触发作弊成功 + 2) 玩家赢  都会推动难度上升）。

优先级：
   Cheat 模式 (score >= 10) > Assist 扶持模式 (score <= -3) > 正常随机。

主要新增 / 修改变量：
   ASSIST_AI_WIN_PROB_INITIAL, ASSIST_AI_WIN_PROB_MIN, ASSIST_AI_WIN_DEC,
   assist_ai_win_prob (运行期实时值), CHEAT_PLAYER_WIN_INC.

CSV 新增列保持不变:CheatActive, CheatProb, AssistActive。

"""
from __future__ import annotations
import cv2, mediapipe as mp, random, time, os, csv, json, math, datetime, socket, sys
from collections import deque
from typing import Dict, List, Tuple, Optional

# =============================================================================
# 1. 配置 / Configuration
# =============================================================================
TEMPLATE_FILE = "gesture_templates.json"
CSV_FILE      = "rps_open_fist_completion.csv"
SMOOTH_WINDOW = 5
CAPTURE_SECONDS = 5
PREP_SECONDS    = 2

# 完成度权重
SHAPE_WEIGHT_OPEN   = 0.7
OPEN_WEIGHT_OPEN    = 0.3
SHAPE_WEIGHT_FIST   = 0.6
CLOSURE_WEIGHT_FIST = 0.4

DEVIATION_GREEN_THRESHOLD = 0.05

FINGERTIPS  = [4, 8, 12, 16, 20]
PALM_POINTS = [0, 5, 9, 13, 17]

# Cheat 机制参数 (高分加压)
CHEAT_START_SCORE    = 10
CHEAT_PROB_INITIAL   = 0.30
CHEAT_PROB_STEP       = 0.05   # 作弊成功（AI 触发克制）后递增
CHEAT_PLAYER_WIN_INC  = 0.05   # 玩家在作弊模式中赢一局也递增
CHEAT_PROB_MAX        = 0.70

# 低分扶持机制参数 (动态降低 AI 胜率)
LOW_SCORE_ASSIST_THRESHOLD = -3          # 分数 <= 此值进入扶持模式
ASSIST_AI_WIN_PROB_INITIAL = 0.30        # 扶持模式初始 AI 赢概率
ASSIST_AI_WIN_PROB_MIN     = 0.05        # AI 赢概率下限
ASSIST_AI_WIN_DEC          = 0.05        # 每次玩家再输一局 (AI 赢) 的递减
ASSIST_AI_DRAW_PROB        = 0.35        # 平局概率固定
# 玩家赢概率 = 1 - (AI赢 + 平局)，自动浮动

# 网络参数
HOST_LISTEN = '0.0.0.0'
PORT = 65432
SOCKET_TIMEOUT = 10.0

# =============================================================================
# 2. 模板管理 / Templates
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
# 3. 几何与归一化 / Geometry & Normalization
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
# 4. 评分 / Scoring
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
# 5. 手势分类 / Gesture Classification
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
# 6. CSV 记录 / CSV Logging
# =============================================================================
def init_csv() -> None:
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE,'w',newline='') as f:
            csv.writer(f).writerow([
                "Time","Round","PlayerGesture","OpponentGesture","Result",
                "Total","Shape","Open","Score","Wins","Losses","Draws","CheatActive","CheatProb","AssistActive","AssistWinProb"
            ])

def append_csv(ts: str, rnd: int, pg: str, og: str, res: str,
               total: Optional[float], shape: Optional[float], open_pct: Optional[float],
               score: int, w: int, l: int, d: int, cheat_active: bool, cheat_prob: float,
               assist_active: bool, assist_win_prob: float) -> None:
    with open(CSV_FILE,'a',newline='') as f:
        csv.writer(f).writerow([
            ts, rnd, pg, og, res,
            ("" if total is None else f"{total:.2f}"),
            ("" if shape is None else f"{shape:.2f}"),
            ("" if open_pct is None else f"{open_pct:.2f}"),
            score, w, l, d, int(cheat_active), f"{cheat_prob:.2f}", int(assist_active), f"{assist_win_prob:.2f}" if assist_active else ""
        ])

# =============================================================================
# 7. 胜负判定 / Winner Determination
# =============================================================================
def judge(player: str, opp: str) -> str:
    if player == opp: return "Draw"
    if (player == "rock" and opp == "scissors") or \
       (player == "scissors" and opp == "paper") or \
       (player == "paper" and opp == "rock"):
        return "You Win!"
    return "You Lose!"

# =============================================================================
# 8. 网络工具 / Networking
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

def exchange_gesture(sock: socket.socket, my_gesture: str) -> Optional[str]:
    try:
        sock.settimeout(SOCKET_TIMEOUT)
        sock.sendall(my_gesture.encode('utf-8'))
        data = sock.recv(1024)
        if not data: return None
        other = data.decode('utf-8').strip()
        if other not in ("rock","paper","scissors"): return None
        return other
    except Exception as e:
        print("[Network] 交换失败:", e)
        return None

# =============================================================================
# 9. 扶持模式 AI 选择 / Assist Mode Choice
# =============================================================================
def assisted_ai_choice(player_gesture: str, win_prob: float) -> str:
    """根据当前扶持 AI 赢概率 win_prob, 固定平局概率 ASSIST_AI_DRAW_PROB, 剩余给玩家赢。"""
    p_win  = max(0.0, min(1.0, win_prob))
    p_draw = ASSIST_AI_DRAW_PROB
    p_player_win = max(0.0, 1.0 - (p_win + p_draw))
    r = random.random()
    # 映射三种结果 -> 具体手势
    # 玩家 rock: AI 赢=paper, 平=rock, 输=scissors
    if player_gesture == 'rock':
        if r < p_win: return 'paper'
        elif r < p_win + p_draw: return 'rock'
        else: return 'scissors'
    if player_gesture == 'paper':
        if r < p_win: return 'scissors'
        elif r < p_win + p_draw: return 'paper'
        else: return 'rock'
    if player_gesture == 'scissors':
        if r < p_win: return 'rock'
        elif r < p_win + p_draw: return 'scissors'
        else: return 'paper'
    # 未知玩家手势时退回随机
    return random.choice(["rock","paper","scissors"])

# =============================================================================
# 10. 主循环 / Main Loop
# =============================================================================
def main() -> None:
    print("==== 模式选择 / Mode Select ====")
    print("1) 本地 vs AI")
    print("2) 联机 (Host)")
    print("3) 联机 (Client)")
    mode = input(">>> ").strip()

    remote_mode = False
    net_sock: Optional[socket.socket] = None
    if mode == '2':
        net_sock = setup_host(); remote_mode = net_sock is not None
    elif mode == '3':
        net_sock = setup_client(); remote_mode = net_sock is not None
    else:
        print("进入单人 AI 模式 ...")

    if mode in ('2','3') and not remote_mode:
        print("联机失败, 回退单人模式。")

    init_csv()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return

    wins = losses = draws = 0
    score = 0
    round_id = 0

    # Cheat 状态
    cheat_mode_active = False
    cheat_prob = CHEAT_PROB_INITIAL

    # Assist 状态变量
    assist_active = False
    assist_ai_win_prob = ASSIST_AI_WIN_PROB_INITIAL

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5) as hands:
        while True:
            round_id += 1

            # ---------- 模式激活/关闭判定 ----------
            # 作弊激活
            if not remote_mode and (score >= CHEAT_START_SCORE) and not cheat_mode_active:
                cheat_mode_active = True
                cheat_prob = CHEAT_PROB_INITIAL
                print(f"[Cheat] 激活 score={score} p={cheat_prob:.2f}")
            # 作弊关闭
            if cheat_mode_active and score < CHEAT_START_SCORE:
                cheat_mode_active = False
                cheat_prob = CHEAT_PROB_INITIAL
                print(f"[Cheat] 关闭 score={score}")
            # 扶持激活
            if (not remote_mode) and (score <= LOW_SCORE_ASSIST_THRESHOLD) and (not cheat_mode_active) and (not assist_active):
                assist_active = True
                assist_ai_win_prob = ASSIST_AI_WIN_PROB_INITIAL
                print(f"[Assist] 激活 score={score} AIwin={assist_ai_win_prob:.2f}")
            # 扶持关闭 (分数回升)
            if assist_active and score > LOW_SCORE_ASSIST_THRESHOLD:
                assist_active = False
                print(f"[Assist] 关闭 score={score}")

            # ---------- 准备阶段 ----------
            prep_start = time.time()
            while time.time() - prep_start < PREP_SECONDS:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                remain = PREP_SECONDS - int(time.time() - prep_start)
                cv2.putText(frame, f"Round {round_id} Ready:{remain}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 3)
                status_line = f"Score:{score} W:{wins} L:{losses} D:{draws}"
                if (not remote_mode) and cheat_mode_active:
                    status_line += f" | Cheat p={cheat_prob:.2f}"
                elif (not remote_mode) and assist_active:
                    status_line += f" | Assist AIwin={assist_ai_win_prob:.2f}"
                cv2.putText(frame, status_line, (10,470), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255,255,255), 2)
                cv2.imshow("RPS Completion Net+Cheat+Assist", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release();
                    if net_sock: net_sock.close()
                    cv2.destroyAllWindows(); return

            # ---------- 采集阶段 ----------
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

                    ktemp = cv2.waitKey(1) & 0xFF
                    if ktemp == ord('t') and player_gesture in ("rock","paper"):
                        base = 'fist' if player_gesture == 'rock' else 'open'
                        norm_cap = normalize_landmarks(lm_list)
                        gesture_templates[base] = {str(i): norm_cap[i] for i in FINGERTIPS}
                        save_templates(gesture_templates)
                        print(f"[模板更新] {base}")

                    if player_gesture in ("rock","paper"):
                        base = 'fist' if player_gesture == 'rock' else 'open'
                        norm = normalize_landmarks(lm_list)
                        if has_template(base):
                            devs = compute_devs(norm, gesture_templates[base])
                            shape = shape_score(devs, base)
                        else:
                            devs = {}; shape = 0.0
                        open_pct = openness_ratio(lm_list)*100
                        total_raw = total_fist(shape, open_pct) if base == 'fist' else total_open(shape, open_pct)
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
                footer = "[t]模板 rock/paper  ESC退出"
                if (not remote_mode) and cheat_mode_active:
                    footer += f"  Cheat p={cheat_prob:.2f}"
                elif (not remote_mode) and assist_active:
                    footer += f"  Assist AIwin={assist_ai_win_prob:.2f}"
                cv2.putText(frame, footer, (230,470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
                cv2.imshow("RPS Completion Net+Cheat+Assist", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release();
                    if net_sock: net_sock.close()
                    cv2.destroyAllWindows(); return

            # ---------- 对手手势生成 ----------
            if remote_mode and net_sock:
                opp_gesture = exchange_gesture(net_sock, player_gesture if gesture_captured else "None")
                if opp_gesture is None:
                    print("[Network] 通信异常 -> 随机 AI")
                    opp_gesture = random.choice(["rock","paper","scissors"])
            else:
                if not gesture_captured or player_gesture not in ("rock","paper","scissors"):
                    player_gesture = "None"
                    opp_gesture = random.choice(["rock","paper","scissors"])
                else:
                    if cheat_mode_active:
                        # 先随机，再按概率强制克制
                        opp_gesture = random.choice(["rock","paper","scissors"])
                        if random.random() < cheat_prob:
                            if player_gesture == "rock": opp_gesture = "paper"
                            elif player_gesture == "paper": opp_gesture = "scissors"
                            elif player_gesture == "scissors": opp_gesture = "rock"
                            cheat_prob = min(CHEAT_PROB_MAX, cheat_prob + CHEAT_PROB_STEP)
                    elif assist_active:
                        opp_gesture = assisted_ai_choice(player_gesture, assist_ai_win_prob)
                    else:
                        opp_gesture = random.choice(["rock","paper","scissors"])

            # ---------- 判定 & 记分 ----------
            if player_gesture in ("rock","paper","scissors"):
                result = judge(player_gesture, opp_gesture)
                if result == "You Win!":
                    score += 3; wins += 1
                    # 高分作弊模式中，玩家胜利 -> AI 获胜概率（cheat_prob）递增
                    if cheat_mode_active:
                        cheat_prob = min(CHEAT_PROB_MAX, cheat_prob + CHEAT_PLAYER_WIN_INC)
                elif result == "You Lose!":
                    score -= 1; losses += 1
                    # 扶持模式中，AI 赢则进一步降低 AI 赢概率
                    if assist_active:
                        assist_ai_win_prob = max(ASSIST_AI_WIN_PROB_MIN, assist_ai_win_prob - ASSIST_AI_WIN_DEC)
                else:
                    draws += 1
            else:
                result = "No gesture!"; player_gesture = "None"

            # 模式关闭再检测 (避免显示不一致)
            if cheat_mode_active and score < CHEAT_START_SCORE:
                cheat_mode_active = False
                cheat_prob = CHEAT_PROB_INITIAL
                print(f"[Cheat] 关闭 score={score}")
            if assist_active and score > LOW_SCORE_ASSIST_THRESHOLD:
                assist_active = False
                print(f"[Assist] 关闭 score={score}")

            # ---------- 结果显示 ----------
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
                cv2.putText(frame, "Best Total: N/A (scissors / none)", (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
            status_line = f"Score:{score} W:{wins} L:{losses} D:{draws}"
            if (not remote_mode) and cheat_mode_active:
                status_line += f" | Cheat p={cheat_prob:.2f}"
            elif (not remote_mode) and assist_active:
                status_line += f" | Assist AIwin={assist_ai_win_prob:.2f}"
            cv2.putText(frame, status_line, (10,470), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            cv2.putText(frame, "ENTER下一回合  ESC退出", (10,440), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180,180,180), 2)
            cv2.imshow("RPS Completion Net+Cheat+Assist", frame)

            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_csv(ts, round_id, player_gesture, opp_gesture, result,
                       show_total, show_shape, show_open, score, wins, losses, draws,
                       cheat_mode_active and (not remote_mode),
                       cheat_prob if cheat_mode_active else 0.0,
                       assist_active and (not remote_mode), assist_ai_win_prob)

            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == 13:  # Enter
                    break
                if k == 27:  # ESC
                    cap.release();
                    if net_sock: net_sock.close()
                    cv2.destroyAllWindows(); return

    cap.release()
    if net_sock: net_sock.close()
    cv2.destroyAllWindows()

# =============================================================================
# 入口 / Entry Point
# =============================================================================
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("用户中断 / Interrupted by user")
        sys.exit(0)
