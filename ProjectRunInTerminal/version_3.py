#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPS (Rock / Paper / Scissors) + Completion (Open / Fist) + Network & Cheat AI
================================================================================
超详细注释版 / FULLY ANNOTATED VERSION

功能概览 (High‑Level Overview)
------------------------------------------------------------
本脚本提供一个强化的“猜拳+动作质量”训练工具：
  1. **识别手势** (rock / paper / scissors) 基于手指伸展简单规则。
  2. **计算完成度 (Completion)** 仅对 rock(拳=fist) 与 paper(掌=open)：
       - Shape%: 与模板(5 指尖归一化坐标)的匹配程度 (指数衰减 + 指尖权重)。
       - Open% : 手张开度 (五指尖到掌心平均归一化距离 → 百分比)。
       - Closure% = 100 - Open%。
       - Total%  : paper -> 0.7*Shape + 0.3*Open; rock -> 0.6*Shape + 0.4*Closure。
       - scissors 不评分 (显示 N/A)。
  3. **模板采集**：采集阶段按 't'，若当前识别为 rock / paper，记录其模板（fist/open）。
  4. **AI 作弊模式**：得分达到阈值后，AI 有概率强制选克制手势；概率可逐回合递增，模拟“自适应难度”。
  5. **联机对战 (TCP)**：可选 Host / Client 互连，只交换手势字符串，不传输图像与完成度，简单、带宽占用小。
  6. **数据记录 (CSV)**：每回合写入（时间、手势、胜负、完成度、分数、作弊状态等），便于后期分析。
  7. **实时可视反馈**：突出当前手势、剩余时间、分数、最佳总分、指尖偏差等，增强训练感知。

适用场景：
  * 上肢 / 手部康复中“张开/握拳”动作质量监控（同时 gamification 增加参与度）。
  * 简单远程对局 / 康复对比练习（完成度仍为本地私有，不上传）。
  * 数据收集：回合级别的定量指标 (Total, Shape, Open)。

依赖 (Dependencies)
------------------------------------------------------------
    pip install --upgrade mediapipe opencv-python numpy

运行步骤 (Run)
------------------------------------------------------------
    python rps_open_fist_completion_net_cheat.py
  → 选择模式：
      1) 单人 (vs AI，可触发作弊)
      2) Host (等待客户端)
      3) Client (连接 Host IP)
  → 采集阶段做手势，按 't' 采集模板 (建议先 paper 后 rock)。
  → 回合结束查看结果，按 Enter 下一回合，ESC 退出。

键位 (Key Bindings)
------------------------------------------------------------
  t       : 在采集阶段记录当前 (rock/paper) 模板
  ENTER   : 结果界面进入下一回合
  ESC     : 随时退出程序

输出文件 (Outputs)
------------------------------------------------------------
  gesture_templates.json        仅保存 open / fist 两类模板 (指尖归一化坐标)
  rps_open_fist_completion.csv  回合日志 (含作弊状态, 完成度, 胜负)

设计权衡 (Design Rationale)
------------------------------------------------------------
  * **模板只存 5 指尖**：压缩 & 稳健；掌骨节点等微抖动对“姿态宏观形状”影响较小。
  * **归一化**：以掌心中心 & 拇指根->小指根距离 去除平移/缩放影响；无需旋转对齐，简单但对手腕旋转轻微敏感。
  * **Shape 指数衰减** `exp(-8*d)`：小偏差 ≈ 线性(一阶)近似，大偏差快速衰减 → 奖励精准对齐。
  * **指尖权重**：强化 open 时拇指外展；fist 时食/中指屈曲对称性。
  * **简单伸指数分类**：无模型依赖，鲁棒性一般；若需提升可接入分类器或阈值调优。
  * **Cheat 逐步提升概率**：模拟自适应难度；防止玩家轻松滚雪球高分。
  * **网络协议极简**：每回合只发送一个 3~8 字节的字符串，无握手帧字段；失败即降级本地 AI。

扩展建议 (Extensions)
------------------------------------------------------------
  * 给 scissors 定义一个“中等张开”目标：Total_scissors = 0.5*Shape + 0.5*(100 - |Open-50|)。
  * 引入旋转不变性：对归一化点做 PCA 主轴旋转或以腕部向量对齐。
  * 多线程网络：避免阻塞 (当前使用 settimeout 并在回合边界交换即可，阻塞成本低)。
  * WebSocket/UDP：降低延迟 (需要心跳与丢包处理)。
  * 更细粒度记录：帧级别数据 (可能需要压缩 & 性能调优)。

--------------------------------------------------------------------------------
"""
from __future__ import annotations
import cv2, mediapipe as mp, random, time, os, csv, json, math, datetime, socket, sys
from collections import deque
from typing import Dict, List, Tuple, Optional

# =============================================================================
# 1. 全局配置 / Global Configuration
# =============================================================================
TEMPLATE_FILE = "gesture_templates.json"           # 模板文件：open / fist
CSV_FILE      = "rps_open_fist_completion.csv"     # 回合日志
SMOOTH_WINDOW = 5                                   # Total 平滑窗口大小 (帧数)
CAPTURE_SECONDS = 5                                 # 采集阶段时长 (秒)
PREP_SECONDS    = 2                                 # 准备倒计时 (秒)

# 完成度权重 (paper=open / rock=fist)
SHAPE_WEIGHT_OPEN   = 0.7
OPEN_WEIGHT_OPEN    = 0.3
SHAPE_WEIGHT_FIST   = 0.6
CLOSURE_WEIGHT_FIST = 0.4

DEVIATION_GREEN_THRESHOLD = 0.05                   # 指尖偏差上色阈值 (匹配质量)

FINGERTIPS  = [4, 8, 12, 16, 20]                   # 5 指尖索引
PALM_POINTS = [0, 5, 9, 13, 17]                    # 掌心近似采样点 (横跨掌面)

# Cheat (仅本地 vs AI)
CHEAT_START_SCORE = 10      # 达到该分触发 cheat 模式
CHEAT_PROB_INITIAL = 0.30   # 初始作弊概率
CHEAT_PROB_STEP    = 0.05   # 每次成功作弊后增量
CHEAT_PROB_MAX     = 0.70   # 作弊概率上限

# 网络 (TCP)
HOST_LISTEN = '0.0.0.0'
PORT = 65432
SOCKET_TIMEOUT = 10.0       # 交换手势时的阻塞超时

# =============================================================================
# 2. 模板管理 / Template Handling
# =============================================================================
def load_templates() -> Dict[str, Dict[str, List[float]]]:
    """加载/初始化模板。
    结构: {"open": {"4": [x,y], ...}, "fist": {...}} 仅保存指尖, 减少对非关键区域敏感度。"""
    if not os.path.exists(TEMPLATE_FILE):
        data = {"open": {}, "fist": {}}
        with open(TEMPLATE_FILE, 'w') as f: json.dump(data, f, indent=2)
        return data
    with open(TEMPLATE_FILE, 'r') as f:
        data = json.load(f)
    changed = False
    for k in ("open", "fist"):
        if k not in data:
            data[k] = {}
            changed = True
    if changed:
        with open(TEMPLATE_FILE, 'w') as f: json.dump(data, f, indent=2)
    return data

def save_templates(tpl: Dict[str, Dict[str, List[float]]]) -> None:
    """保存模板 JSON (覆盖写入)。"""
    with open(TEMPLATE_FILE, 'w') as f:
        json.dump(tpl, f, indent=2)

gesture_templates = load_templates()

def has_template(name: str) -> bool:
    return name in gesture_templates and len(gesture_templates[name]) > 0

# =============================================================================
# 3. 几何与归一化 / Geometry & Normalization
# =============================================================================
def palm_center_and_width(lm_list) -> Tuple[float, float, float]:
    """估算掌心中心 & 尺度 (拇指根→小指根距离)。
    返回:
        cx, cy: 掌心归一化中心 (0~1 图像坐标系)
        w     : 尺度 (避免除零)"""
    cx = sum(lm_list[i].x for i in PALM_POINTS) / len(PALM_POINTS)
    cy = sum(lm_list[i].y for i in PALM_POINTS) / len(PALM_POINTS)
    w  = math.dist((lm_list[5].x, lm_list[5].y), (lm_list[17].x, lm_list[17].y)) or 1e-6
    return cx, cy, w

def normalize_landmarks(lm_list) -> Dict[int, Tuple[float,float]]:
    """以掌心中心平移，按掌宽缩放，得到非尺度/位移依赖的局部形状表示。"""
    cx, cy, w = palm_center_and_width(lm_list)
    return {i: ((lm.x - cx)/w, (lm.y - cy)/w) for i, lm in enumerate(lm_list)}

def openness_ratio(lm_list) -> float:
    """计算张开度 (0~1)。指标: 指尖到掌心的平均归一化距离 / 0.9 (截断 ≤1)。"""
    cx, cy, w = palm_center_and_width(lm_list)
    ds = [math.dist((lm_list[t].x, lm_list[t].y), (cx, cy))/w for t in FINGERTIPS]
    avg = sum(ds)/len(ds)
    return min(1.0, avg/0.9)

# =============================================================================
# 4. 评分 / Scoring Functions
# =============================================================================
def compute_devs(norm: Dict[int,Tuple[float,float]], template: Dict[str,List[float]]) -> Dict[int,float]:
    """与模板指尖坐标的欧氏距离 (归一化空间)。"""
    devs = {}
    for k, (tx, ty) in template.items():
        idx = int(k)
        if idx in norm:
            x, y = norm[idx]
            devs[idx] = math.hypot(x - tx, y - ty)
    return devs

def finger_weight(idx: int, base: str) -> float:
    """指尖权重策略: open 强化拇指 (外展难点); fist 强化食/中指屈曲对齐。"""
    if base == "open": return 1.2 if idx == 4 else 1.0
    if base == "fist": return 1.1 if idx in (8,12) else 1.0
    return 1.0

def shape_score(devs: Dict[int,float], base: str) -> float:
    """用 exp(-8*d) 将距离转换为相似度，平均后 ×100 得 Shape%。"""
    if not devs: return 0.0
    sims = [math.exp(-8*d) * finger_weight(i, base) for i,d in devs.items()]
    return (sum(sims)/len(sims))*100.0

def total_open(shape_part: float, open_part: float) -> float:
    return SHAPE_WEIGHT_OPEN*shape_part + OPEN_WEIGHT_OPEN*open_part

def total_fist(shape_part: float, open_part: float) -> float:
    closure = 100 - open_part
    return SHAPE_WEIGHT_FIST*shape_part + CLOSURE_WEIGHT_FIST*closure

# =============================================================================
# 5. 手势分类 / Gesture Classification (Rule-Based)
# =============================================================================
def classify_rps(lm_list) -> str:
    """通过 4 个指尖是否伸展 (相对 PIP 关节) 简单判定 R/P/S。
    局限: 角度极端或部分遮挡时易误判; 可用 ML 模型替换。"""
    tips = [8,12,16,20]
    fingers = [1 if lm_list[t].y < lm_list[t-2].y else 0 for t in tips]
    c = sum(fingers)
    if c == 0: return "rock"
    if c == 2: return "scissors"
    if c >= 4: return "paper"
    return "unknown"

# =============================================================================
# 6. CSV 记录 / Logging
# =============================================================================
def init_csv() -> None:
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE,'w',newline='') as f:
            csv.writer(f).writerow([
                "Time","Round","PlayerGesture","OpponentGesture","Result",
                "Total","Shape","Open","Score","Wins","Losses","Draws","CheatActive","CheatProb"
            ])

def append_csv(ts: str, rnd: int, pg: str, og: str, res: str,
               total: Optional[float], shape: Optional[float], open_pct: Optional[float],
               score: int, w: int, l: int, d: int, cheat_active: bool, cheat_prob: float) -> None:
    with open(CSV_FILE,'a',newline='') as f:
        csv.writer(f).writerow([
            ts, rnd, pg, og, res,
            ("" if total is None else f"{total:.2f}"),
            ("" if shape is None else f"{shape:.2f}"),
            ("" if open_pct is None else f"{open_pct:.2f}"),
            score, w, l, d, int(cheat_active), f"{cheat_prob:.2f}"
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
# 8. 网络工具 / Networking Helpers (Minimal Protocol)
# =============================================================================
def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8",80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def setup_host() -> Optional[socket.socket]:
    """Host 监听并等待客户端连接 (单连接)。失败返回 None。"""
    try:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind((HOST_LISTEN, PORT))
        srv.listen(1)
        ip = get_local_ip()
        print(f"[Host] 等待客户端... (IP: {ip}:{PORT})")
        conn, addr = srv.accept()
        print(f"[Host] 客户端已连接: {addr}")
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
    """交换双方手势：发送自身手势 (字符串) -> 接收对方手势。
    若失败返回 None (上层降级为随机 AI)。"""
    try:
        sock.settimeout(SOCKET_TIMEOUT)
        sock.sendall(my_gesture.encode('utf-8'))
        data = sock.recv(1024)
        if not data:
            return None
        other = data.decode('utf-8').strip()
        if other not in ("rock","paper","scissors"):
            return None
        return other
    except Exception as e:
        print("[Network] 交换失败:", e)
        return None

# =============================================================================
# 9. 主循环 / Main Game Loop
# =============================================================================
def main() -> None:
    # ---------- 模式选择 ----------
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
        print("进入单人 AI 模式 (Enter single-player mode).")

    if mode in ('2','3') and not remote_mode:
        print("联机失败, 回退单人。")

    init_csv()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头 / Cannot open camera")
        return

    # 统计变量
    wins = losses = draws = 0
    score = 0
    round_id = 0

    # Cheat 状态
    cheat_mode_active = False
    cheat_prob = CHEAT_PROB_INITIAL

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5) as hands:
        while True:
            round_id += 1

            # ------------------------------------------------------------
            # (A) 准备阶段
            # ------------------------------------------------------------
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
                cv2.putText(frame, status_line, (10,470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                cv2.imshow("RPS Completion Net+Cheat", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    cap.release();
                    if net_sock: net_sock.close()
                    cv2.destroyAllWindows(); return

            # 达到阈值启动 cheat (仅单机)
            if not remote_mode and (score >= CHEAT_START_SCORE) and not cheat_mode_active:
                cheat_mode_active = True
                print("[Cheat] 激活: 玩家得分较高, AI 增加难度")

            # ------------------------------------------------------------
            # (B) 采集阶段：实时识别 & 计算完成度
            # ------------------------------------------------------------
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

                    # 1) 手势分类
                    cur = classify_rps(lm_list)
                    if cur in ("rock","paper","scissors"):
                        player_gesture = cur
                        gesture_captured = True

                    # 2) 模板采集 (仅 rock/paper)
                    ktemp = cv2.waitKey(1) & 0xFF
                    if ktemp == ord('t') and player_gesture in ("rock","paper"):
                        base = 'fist' if player_gesture == 'rock' else 'open'
                        norm_cap = normalize_landmarks(lm_list)
                        gesture_templates[base] = {str(i): norm_cap[i] for i in FINGERTIPS}
                        save_templates(gesture_templates)
                        print(f"[模板更新] {base}")

                    # 3) 完成度计算 (仅 rock/paper)
                    if player_gesture in ("rock","paper"):
                        base = 'fist' if player_gesture == 'rock' else 'open'
                        norm = normalize_landmarks(lm_list)
                        if has_template(base):
                            devs = compute_devs(norm, gesture_templates[base])
                            shape = shape_score(devs, base)
                        else:
                            devs = {}
                            shape = 0.0
                        open_pct = openness_ratio(lm_list)*100
                        total_raw = total_fist(shape, open_pct) if base == 'fist' else total_open(shape, open_pct)

                        # 4) 平滑 Total
                        total_window.append(total_raw)
                        total_disp = sum(total_window)/len(total_window)

                        # 5) 刷新最佳
                        if (best_total is None) or (total_disp > best_total):
                            best_total = total_disp; best_shape = shape; best_open = open_pct

                        # 6) 指尖偏差显示 (有模板才有意义)
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

                    # 7) 绘制骨架 (视觉反馈)
                    mp.solutions.drawing_utils.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

                # UI: 回合剩余秒数 & 当前手势 & 底部状态
                remain = CAPTURE_SECONDS - int(time.time() - capture_start)
                cv2.putText(frame, f"Show ({remain})", (20,60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                cv2.putText(frame, f"Gesture:{player_gesture}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                footer = "[t]模板 rock/paper  ESC退出"
                if (not remote_mode) and cheat_mode_active:
                    footer += f"  Cheat p={cheat_prob:.2f}"
                cv2.putText(frame, footer, (280,470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

                cv2.imshow("RPS Completion Net+Cheat", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    cap.release();
                    if net_sock: net_sock.close()
                    cv2.destroyAllWindows(); return

            # ------------------------------------------------------------
            # (C) 对手手势获取 / Acquire Opponent Gesture
            # ------------------------------------------------------------
            if remote_mode and net_sock:
                opp_gesture = exchange_gesture(net_sock, player_gesture if gesture_captured else "None")
                if opp_gesture is None:
                    print("[Network] 通信异常, 使用随机 AI")
                    opp_gesture = random.choice(["rock","paper","scissors"])
            else:
                if not gesture_captured or player_gesture not in ("rock","paper","scissors"):
                    player_gesture = "None"  # 标记无效回合
                    opp_gesture = random.choice(["rock","paper","scissors"])
                else:
                    opp_gesture = random.choice(["rock","paper","scissors"])
                    if cheat_mode_active and player_gesture in ("rock","paper","scissors"):
                        if random.random() < cheat_prob:
                            # 强制克制手势 + 提升概率 (难度递增)
                            if player_gesture == "rock": opp_gesture = "paper"
                            elif player_gesture == "paper": opp_gesture = "scissors"
                            elif player_gesture == "scissors": opp_gesture = "rock"
                            cheat_prob = min(CHEAT_PROB_MAX, cheat_prob + CHEAT_PROB_STEP)

            # ------------------------------------------------------------
            # (D) 胜负 & 计分 / Decision & Scoring
            # ------------------------------------------------------------
            if player_gesture in ("rock","paper","scissors"):
                result = judge(player_gesture, opp_gesture)
                if result == "You Win!":
                    score += 3; wins += 1
                elif result == "You Lose!":
                    score -= 1; losses += 1
                else:
                    draws += 1
            else:
                result = "No gesture!"
                player_gesture = "None"

            # ------------------------------------------------------------
            # (E) 结果显示 / Display Summary
            # ------------------------------------------------------------
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            show_total = best_total if player_gesture in ("rock","paper") else None
            show_shape = best_shape if player_gesture in ("rock","paper") else None
            show_open  = best_open  if player_gesture in ("rock","paper") else None

            cv2.putText(frame, f"You: {player_gesture}", (10,40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            who = "Opponent" if remote_mode else "AI"
            cv2.putText(frame, f"{who}: {opp_gesture}", (10,80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, result, (10,125), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
            if show_total is not None:
                cv2.putText(frame, f"Best Total:{show_total:.1f}% Shape:{show_shape:.1f}% Open:{show_open:.1f}%", (10,170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            else:
                cv2.putText(frame, "Best Total: N/A (scissors / none)", (10,170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)
            status_line = f"Score:{score} W:{wins} L:{losses} D:{draws}"
            if (not remote_mode) and cheat_mode_active:
                status_line += f" | Cheat p={cheat_prob:.2f}"
            cv2.putText(frame, status_line, (10,470), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
            cv2.putText(frame, "ENTER下一回合  ESC退出", (10,440), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180,180,180), 2)
            cv2.imshow("RPS Completion Net+Cheat", frame)

            # (F) 记录 CSV
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_csv(ts, round_id, player_gesture, opp_gesture, result,
                       show_total, show_shape, show_open, score, wins, losses, draws,
                       cheat_mode_active and (not remote_mode),
                       cheat_prob if cheat_mode_active else 0.0)

            # ------------------------------------------------------------
            # (G) 等待下一回合 / Wait for Next Round
            # ------------------------------------------------------------
            while True:
                k = cv2.waitKey(0) & 0xFF
                if k == 13:  # Enter -> 下一回合
                    break
                if k == 27:  # ESC -> 退出
                    cap.release();
                    if net_sock: net_sock.close()
                    cv2.destroyAllWindows(); return

    cap.release()
    if net_sock: net_sock.close()
    cv2.destroyAllWindows()

# =============================================================================
# 10. 入口 / Entry Point
# =============================================================================
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断 / Interrupted by user")
        sys.exit(0)
