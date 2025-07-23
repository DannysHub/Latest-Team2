#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPS (Rock / Paper / Scissors) + Completion (Open/Fist) + Network
自适应难度 (低分扶持 / 正常 / 高分加压) —— 按最新概率需求实现
================================================================================
**最新需求精确实现：**
1. **低分区 (< -3)**：进入 *扶持模式 (Assist)*
   - 固定平局概率 = 30%。
   - 设定 AI 胜率初始值 = 30% (可调 `ASSIST_AI_WIN_PROB_INITIAL`)。
   - 玩家 *再输一局* (AI 赢) -> AI 胜率下降 5% (`ASSIST_AI_WIN_DEC`)。
   - AI 胜率在扶持模式下 **≥ 20%** (`ASSIST_AI_WIN_PROB_MIN = 0.20`)；没有上限要求（自然 ≤ 70% 由逻辑保证）。
   - 剩余概率全部给玩家赢：`P(player_win) = 1 - (P(ai_win)+0.30)`。
   - 扶持模式只在分数 < -3 激活（进入时重置 AI 胜率）；当分数 ≥ -3 即可保持或退出? —— 需求里说明“当玩家分数大于等于3时恢复正常”，因此：
       * 分数 < -3 → 激活 / 保持
       * 分数 ≥ 3  → 退出扶持
       * -3 ≤ 分数 < 3 区间：不再新增惩罚或奖励，若之前已激活继续保持当前 win 概率（可根据需要改，这里采用**保持**）。

2. **正常区 (-3 ≤ 分数 < 10)**：
   - 不启用特殊加成/扶持；AI 采用 **均匀随机** (1/3,1/3,1/3)。
   - 如果刚从扶持或高分模式退出，恢复均匀随机。

3. **高分区 (≥ 10)**：进入 *高分加压模式 (High Mode)*
   - 初始化 AI 胜率 = 35% (`HIGH_AI_WIN_PROB_INITIAL=0.35`)。
   - 玩家每 **赢** 一局：AI 胜率 +5% (`HIGH_AI_WIN_INC=0.05`)。
   - 玩家每 **输** 一局：AI 胜率 -3% (`HIGH_AI_WIN_DEC=0.03`)。
   - AI 胜率在高分模式下必须保持 **30% ≤ win_prob < 45%** （实现上 clamp 到 [0.30, 0.45]；45% 设为上界不再上涨）。
   - 未指明平局概率固定，因此令：`P(draw) = (1 - P(ai_win)) / 2`，`P(player_win) = 同上`，保持对称。
   - 分数跌回 <10 退出高分模式；若仍 ≥10 再进入时重置 AI 胜率到 35%。

4. **互斥关系**：高分模式与扶持模式互斥，优先级：高分 > 扶持。

5. **与旧“作弊(强制克制)”机制区别**：本版本 **移除强制克制 cheat_prob**，改为完全按概率抽样出 AI 手势，使整体统计赢/平/输概率符合策略。

6. **完成度**：只计算 rock(拳=fist) 与 paper(掌=open)，scissors 不计算 (显示 N/A)。

7. **CSV 新列**：`HighModeActive, HighWinProb, AssistActive, AssistWinProb`；去除旧 Cheat 列。

如需：
  * 在 -3 ≤ score < 3 强制退出扶持改为恢复正常 → 告知我即可修改。
  * 添加剪刀手完成度或更细概率策略
  * 将概率写回屏幕更多细节
  * 使用 EMA 平滑完成度
随时提出。
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

# ---- Assist 低分扶持参数 ----
ASSIST_ENTER_SCORE        = -3   # score < -3 激活 (严格小于)
ASSIST_EXIT_SCORE         = 3    # score >= 3 退出
ASSIST_AI_WIN_PROB_INITIAL= 0.30 # 进入时 AI win 初始
ASSIST_AI_WIN_DEC         = 0.05 # 玩家继续输 -> AI win 减 5%
ASSIST_AI_WIN_PROB_MIN    = 0.20 # 下限 20%
ASSIST_DRAW_FIXED         = 0.30 # Draw 固定 30%

# ---- High Mode 高分加压参数 ----
HIGH_ENTER_SCORE          = 10   # score >= 10 激活
HIGH_AI_WIN_PROB_INITIAL  = 0.35 # 初始 35%
HIGH_AI_WIN_INC           = 0.05 # 玩家赢 -> AI win +5%
HIGH_AI_WIN_DEC           = 0.03 # 玩家输 -> AI win -3%
HIGH_AI_WIN_MIN           = 0.30 # 下限 30%
HIGH_AI_WIN_MAX           = 0.45 # 上限 45% (不超过, 严格 <45 表述用 clamp)

# 网络
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
# 6. CSV 记录 / CSV Logging (更新列)
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
def assisted_ai_choice(player_gesture: str, ai_win_prob: float) -> str:
    """扶持：固定平局 30%，AI 赢 = ai_win_prob (≥20%)，剩余给玩家赢。
    根据玩家手势映射具体手势，使概率分配满足目标。"""
    p_ai = ai_win_prob
    p_draw = ASSIST_DRAW_FIXED
    p_player = max(0.0, 1.0 - (p_ai + p_draw))
    r = random.random()
    # player: rock -> AI赢=paper / 平=rock / AI输=scissors
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
# 10. 高分模式 AI 选择 / High Mode Choice
# =============================================================================
def highmode_ai_choice(player_gesture: str, ai_win_prob: float) -> str:
    """高分模式：AI 赢概率 = ai_win_prob (30~45%)；平局、玩家赢对称分配。"""
    p_ai = ai_win_prob
    remain = 1.0 - p_ai
    p_draw = remain / 2.0
    p_player = remain / 2.0
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
# 11. 主循环 / Main Loop
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

    # 模式状态变量
    assist_active = False
    assist_ai_win_prob = ASSIST_AI_WIN_PROB_INITIAL
    high_active = False
    high_ai_win_prob = HIGH_AI_WIN_PROB_INITIAL

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5) as hands:
        while True:
            round_id += 1

            # ---- 状态切换 (回合开始) ----
            # 高分模式优先
            if not remote_mode and score >= HIGH_ENTER_SCORE:
                if not high_active:
                    high_active = True
                    high_ai_win_prob = HIGH_AI_WIN_PROB_INITIAL
                    # 高分激活时若扶持开着则关闭
                    if assist_active:
                        assist_active = False
                    print(f"[High] 激活 score={score} win_prob={high_ai_win_prob:.2f}")
            else:
                if high_active and score < HIGH_ENTER_SCORE:
                    high_active = False
                    print(f"[High] 关闭 score={score}")

            # 扶持模式（仅在非高分 & 分数 < -3）
            if (not remote_mode) and (not high_active) and (score < ASSIST_ENTER_SCORE):
                if not assist_active:
                    assist_active = True
                    assist_ai_win_prob = ASSIST_AI_WIN_PROB_INITIAL
                    print(f"[Assist] 激活 score={score} win_prob={assist_ai_win_prob:.2f}")
            # 退出扶持：分数 ≥ 3 (保持到阈值避免频繁抖动)
            if assist_active and score >= ASSIST_EXIT_SCORE:
                assist_active = False
                print(f"[Assist] 关闭 score={score}")

            # ---- 准备阶段 ----
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
                cv2.putText(frame, status, (10,470), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255,255,255), 2)
                cv2.imshow("RPS Completion Adaptive", frame)
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
                        print(f"[模板更新] {base}")

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
                footer = "[t]模板 rock/paper  ESC退出"
                if high_active:
                    footer += f"  High winP={high_ai_win_prob:.2f}"
                elif assist_active:
                    footer += f"  Assist winP={assist_ai_win_prob:.2f}"
                cv2.putText(frame, footer, (240,470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
                cv2.imshow("RPS Completion Adaptive", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release();
                    if net_sock: net_sock.close()
                    cv2.destroyAllWindows(); return

            # ---- AI / 对手手势 ----
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

            # ---- 回合结束重新检查模式边界 ----
            # 高分退出
            if high_active and score < HIGH_ENTER_SCORE:
                high_active = False
                print(f"[High] 关闭 score={score}")
            # 扶持退出
            if assist_active and score >= ASSIST_EXIT_SCORE:
                assist_active = False
                print(f"[Assist] 关闭 score={score}")
            # 若无模式，检查是否应进入某模式 (优先高分)
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
                cv2.putText(frame, f"Best Total:{show_total:.1f}% Shape:{show_shape:.1f}% Open:{show_open:.1f}%", (10,170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            else:
                cv2.putText(frame, "Best Total: N/A (scissors / none)", (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
            status = f"Score:{score} W:{wins} L:{losses} D:{draws}"
            if high_active:
                status += f" | High winP={high_ai_win_prob:.2f}"
            elif assist_active:
                status += f" | Assist winP={assist_ai_win_prob:.2f}"
            cv2.putText(frame, status, (10,470), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            cv2.putText(frame, "ENTER下一回合  ESC退出", (10,440), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180,180,180), 2)
            cv2.imshow("RPS Completion Adaptive", frame)

            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_csv(ts, round_id, player_gesture, opp_gesture, result,
                       show_total, show_shape, show_open, score, wins, losses, draws,
                       high_active and (not remote_mode), high_ai_win_prob if high_active else 0.0,
                       assist_active and (not remote_mode), assist_ai_win_prob if assist_active else 0.0)

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
        print("\n用户中断 / Interrupted by user")
        sys.exit(0)
