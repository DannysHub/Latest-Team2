#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rps_adaptive_full.py (with ACK handshake)
====================
整合：
  - Local/Host/Client 模式选择与网络交互
  - Host/Client 特殊弹窗（B 返回 / ESC 退出），加入 ACK 确认机制
  - MediaPipe Hands 手势关键点检测与绘制
  - Shape/Open/Total 实时完成度展示
  - 按 't' 采集模板
  - 自适应难度（Assist/Normal/High）
  - 下一回合弹窗（显示回合、Score、上一回合 Best Total、按键提示）
  - 回合结果展示与日志记录（CSV）
"""
import cv2
import mediapipe as mp
import numpy as np
import random
import time
import os
import csv
import json
import math
import datetime
import socket
import sys
from collections import deque



# ----------------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------------
TEMPLATE_FILE    = "gesture_templates.json"
CSV_FILE         = "rps_open_fist_completion.csv"
SMOOTH_WINDOW    = 5
CAPTURE_SECONDS  = 5
PREP_SECONDS     = 2
DEVIATION_THRESH = 0.05

# 自适应难度参数
ASSIST_ENTER       = -3
ASSIST_EXIT        = 3
ASSIST_WIN_PROB    = 0.30
ASSIST_DEC         = 0.05
ASSIST_MIN         = 0.20
ASSIST_DRAW_FIXED  = 0.30

HIGH_ENTER         = 10
HIGH_WIN_PROB      = 0.35
HIGH_INC           = 0.05
HIGH_DEC           = 0.03
HIGH_MIN           = 0.30
HIGH_MAX           = 0.45

HOST_LISTEN = "0.0.0.0"
PORT        = 65432
WINDOW_NAME = "RPS Completion Adaptive"

# 手势关键点
FINGERTIPS  = [4,8,12,16,20]
PALM_POINTS = [0,5,9,13,17]




# ----------------------------------------------------------------------------
# 模板管理 & CSV 日志
# ----------------------------------------------------------------------------
def load_templates():
    if not os.path.exists(TEMPLATE_FILE):
        tpl = {"open":{}, "fist":{}, "scissors":{}}
        with open(TEMPLATE_FILE,"w") as f: json.dump(tpl,f,indent=2)
        return tpl
    with open(TEMPLATE_FILE,"r") as f:
        tpl = json.load(f)
    for k in ("open","fist","scissors"):
        tpl.setdefault(k,{})
    return tpl

def save_templates(tpl):
    with open(TEMPLATE_FILE,"w") as f:
        json.dump(tpl,f,indent=2)

gesture_templates = load_templates()

def has_template(name):
    return name in gesture_templates and bool(gesture_templates[name])

def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE,"w",newline="") as f:
            csv.writer(f).writerow([
                "Time","Round","Mode","PlayerGesture","OpponentGesture","Result",
                "Total","Shape","Open","Score","Wins","Losses","Draws",
                "HighActive","HighProb","AssistActive","AssistProb"
            ])

def append_csv(round_id, mode, pg, og, res, total, shape, open_pct,
               score, wins, losses, draws,
               high_active, high_prob, assist_active, assist_prob):
    with open(CSV_FILE,"a",newline="") as f:
        csv.writer(f).writerow([
            datetime.datetime.now().isoformat(),
            round_id, mode, pg, og, res,
            f"{total:.2f}", f"{shape:.2f}", f"{open_pct:.2f}",
            score, wins, losses, draws,
            int(high_active), f"{high_prob:.2f}" if high_active else "",
            int(assist_active), f"{assist_prob:.2f}" if assist_active else ""
        ])

# ----------------------------------------------------------------------------
# 完成度计算辅助
# ----------------------------------------------------------------------------
def palm_center_and_width(lm):
    cx = sum(lm[i].x for i in PALM_POINTS)/len(PALM_POINTS)
    cy = sum(lm[i].y for i in PALM_POINTS)/len(PALM_POINTS)
    w  = math.dist((lm[5].x,lm[5].y),(lm[17].x,lm[17].y)) or 1e-6
    return cx, cy, w

def normalize_landmarks(lm):
    cx, cy, w = palm_center_and_width(lm)
    return {i:((pt.x-cx)/w,(pt.y-cy)/w) for i,pt in enumerate(lm)}

def openness_ratio(lm):
    cx, cy, w = palm_center_and_width(lm)
    ds = [math.dist((lm[t].x,lm[t].y),(cx,cy))/w for t in FINGERTIPS]
    return min(1.0, sum(ds)/len(ds)/0.9)

def compute_devs(norm, tpl):
    devs = {}
    for k,(tx,ty) in tpl.items():
        idx = int(k)
        if idx in norm:
            x,y = norm[idx]
            devs[idx] = math.hypot(x-tx,y-ty)
    return devs

def shape_score(devs, base):
    if not devs: return 0.0
    sims = []
    for i,d in devs.items():
        bonus = 1.0
        if base=="open"   and i==4:     bonus=1.2
        if base=="fist"   and i in (8,12): bonus=1.1
        sims.append(math.exp(-8*d)*bonus)
    return sum(sims)/len(sims)*100

def total_score(base, shape, open_pct):
    if base=="rock":     return 0.6*shape + 0.4*(100-open_pct)
    if base=="paper":    return 0.7*shape + 0.3*open_pct
    # scissors
    return 0.5*shape + 0.5*(100-abs(open_pct-50))

# ----------------------------------------------------------------------------
# 手势分类 & 判定
# ----------------------------------------------------------------------------
def classify_rps(lm):
    tips = [8,12,16,20]
    fingers = [1 if lm[t].y<lm[t-2].y else 0 for t in tips]
    c = sum(fingers)
    if c==0:    return "rock"
    if c==2:    return "scissors"
    if c>=4:    return "paper"
    return "unknown"

def judge(player, opp):
    if player==opp: return "Draw"
    win = (player=="rock" and opp=="scissors") or \
          (player=="scissors" and opp=="paper") or \
          (player=="paper" and opp=="rock")
    return "You Win!" if win else "You Lose!"

# ----------------------------------------------------------------------------
# 弹窗：模式选择
# ----------------------------------------------------------------------------
def popup_mode_selection(title="Mode Selection"):
    opts = ["Local", "Host", "Client", "Template"]  # ✅ 增加 Template
    idx = 0
    cv2.namedWindow(title)
    while True:
        frm = np.zeros((360, 600, 3), dtype="uint8")  # 高度调大一点
        cv2.putText(frm, "Select Mode:", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        for i, opt in enumerate(opts):
            col = (0, 255, 255) if i == idx else (255, 255, 255)
            prefix = "> " if i == idx else "  "
            cv2.putText(frm, prefix + opt, (50, 100 + i * 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        cv2.putText(frm, "W/S: move   Enter: select   ESC: quit", (30, 340),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.imshow(title, frm)
        k = cv2.waitKey(100) & 0xFF
        if k in (ord('w'), ord('W')):
            idx = (idx - 1) % 4
        if k in (ord('s'), ord('S')):
            idx = (idx + 1) % 4
        if k in (13, 10):  # ENTER
            cv2.destroyWindow(title)
            return idx + 1  # 返回 1=Local, 2=Host, 3=Client, 4=Template
        if k == 27:
            cv2.destroyAllWindows()
            sys.exit(0)


# ----------------------------------------------------------------------------
# Host 等待连接弹窗 (B 返回 / ESC 退出)
# ----------------------------------------------------------------------------
def setup_host():
    srv = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST_LISTEN,PORT)); srv.listen(1)
    ip = get_local_ip()
    title="Waiting for Client"
    cv2.namedWindow(title)
    while True:
        frm = np.zeros((240,500,3),dtype="uint8")
        cv2.putText(frm,"Waiting for Connection...",(20,60),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
        cv2.putText(frm,f"IP: {ip}:{PORT}",(20,140),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        cv2.putText(frm,"Press B to go back   ESC to quit",(20,200),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(180,180,180),1)
        cv2.imshow(title,frm)
        k = cv2.waitKey(100)&0xFF
        if k in (ord('b'),ord('B')):
            srv.close(); cv2.destroyWindow(title); return None
        if k==27:
            srv.close(); cv2.destroyAllWindows(); sys.exit(0)
        try:
            srv.settimeout(0.1)
            conn, addr = srv.accept()
            print(">> [Host] accepted connection from", addr)
            srv.settimeout(None)
            cv2.destroyWindow(title)
            # —— 切换为非阻塞，才能让 wait_both_ready 中的 recv() 及时抛出异常
            conn.setblocking(False)
            return conn
        except socket.timeout:
            continue

# ----------------------------------------------------------------------------
# Client 输入 IP & 错误重试弹窗
# ----------------------------------------------------------------------------
def client_error_popup():
    opts = ["Retry","Back"]; idx=0; title="Connection Failed"
    cv2.namedWindow(title)
    while True:
        frm = np.zeros((180,400,3),dtype="uint8")
        cv2.putText(frm,"❌ Connection Failed",(20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        for i,o in enumerate(opts):
            col = (0,255,255) if i==idx else (255,255,255)
            cv2.putText(frm,"> "+o,(50,100+i*30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)
        cv2.imshow(title,frm)
        k = cv2.waitKey(100)&0xFF
        if k in (ord('w'),ord('W')): idx=(idx-1)%2
        if k in (ord('s'),ord('S')): idx=(idx+1)%2
        if k in (13,10):
            cv2.destroyWindow(title); return opts[idx]
        if k==27:
            cv2.destroyWindow(title); return "Back"

def setup_client():
    while True:
        ip_chars=""; cursor=True; last=time.time()
        win="Enter Host IP"; cv2.namedWindow(win)
        while True:
            frm=np.zeros((200,600,3),dtype="uint8")
            cv2.putText(frm,"Enter Host IP:",(20,50),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            disp = ip_chars + ("_" if cursor else "")
            cv2.putText(frm,disp,(20,120),
                        cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,255),2)
            cv2.putText(frm,"Enter:OK   Esc:Back   Del:BS",(20,180),
                        cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)
            cv2.imshow(win,frm)
            if time.time()-last>0.5:
                cursor = not cursor; last=time.time()
            k = cv2.waitKey(100)
            if k==27:
                cv2.destroyWindow(win); return None
            if k in (13,10): break
            if k in (8,127): ip_chars=ip_chars[:-1]
            elif 32<=k<127: ip_chars+=chr(k)
        cv2.destroyWindow(win)
        try:
            cli = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
            cli.connect((ip_chars.strip(),PORT))
            print(">> [Client] connected to host at", ip_chars.strip(), ":", PORT)
            # 同理，客户端也切到非阻塞
            cli.setblocking(False)
            return cli
        except:
            choice = client_error_popup()
            if choice=="Back": return None

# ----------------------------------------------------------------------------
# 同步弹窗 & “下一回合” 弹窗 with ACK
# ----------------------------------------------------------------------------
def wait_both_ready(name, sock, is_host, network, lines, timeout=30):
    start = time.time()
    local_ready = False
    remote_ready = False
    ack_received = False
    last_ready_sent = 0  # 定时重发 READY
    cv2.namedWindow(name)

    while True:
        # ---------- UI 显示 ----------
        frame = np.zeros((300, 640, 3), dtype="uint8")
        for i, t in enumerate(lines):
            cv2.putText(frame, t, (20, 40 + i * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        status = f"Local={'✔' if local_ready else ' '}, Remote={'✔' if remote_ready else ' '}, ACK={'✔' if ack_received else ' '}"
        cv2.putText(frame, status, (20, 40 + len(lines) * 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(name, frame)

        # ---------- 按键响应 ----------
        k = cv2.waitKey(100) & 0xFF
        if k in (ord('b'), ord('B')):
            cv2.destroyWindow(name)
            return 'back'
        if k == 27:
            cv2.destroyAllWindows()
            return False
        if k in (13, 10) and not local_ready:
            local_ready = True
            print(">>> ENTER pressed, local ready")

        # ---------- 网络握手 ----------
        if local_ready and network and sock:
            if time.time() - last_ready_sent > 1.5:
                try:
                    sock.sendall(b"READY")
                    print(">>> [SEND] READY")
                    last_ready_sent = time.time()
                except Exception as e:
                    print(">>> [ERROR] sending READY:", e)

        if network and sock:
            try:
                data = sock.recv(16)
                if data == b"READY":
                    if not remote_ready:
                        print(">>> [RECV] READY -> sending ACK")
                        sock.sendall(b"ACK")
                    remote_ready = True
                elif data == b"ACK":
                    print(">>> [RECV] ACK")
                    ack_received = True
            except BlockingIOError:
                pass
            except Exception as e:
                print(">>> [ERROR] receiving:", e)

        # ---------- 判断握手是否完成 ----------
        if local_ready and remote_ready and (not network or ack_received):
            cv2.destroyWindow(name)
            print(">>> Both ready, proceeding")
            return True

        # ---------- 超时退出 ----------
        if time.time() - start > timeout:
            cv2.destroyWindow(name)
            print(">>> Timeout in wait_both_ready()")
            return False


def show_next_round_popup(title, rnd, score, last_best):
    global sock, is_host, network
    lines = [f"Round {rnd}", f"Score: {score}"]
    if last_best is not None:
        lines.append(f"Last Best Total: {last_best:.1f}%")
    lines += ["Press ENTER to ready", "Press ESC to quit", "Press B to go back"]
    # ====== 方案 A 分支：Local 模式走简化弹窗，网络模式走三步握手弹窗 ======
    if not network:
        # 本地模式：只等待 ENTER / ESC / B，不再收发任何网络数据
        cv2.namedWindow(title)
        while True:
            frame = np.zeros((300,640,3),dtype="uint8")
            for i, t in enumerate(lines):
               cv2.putText(frame, t, (20,40+i*40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow(title, frame)
            k = cv2.waitKey(100) & 0xFF
            if k in (ord('b'), ord('B')):
                cv2.destroyWindow(title)
                return 'back'
            if k in (13, 10):  # ENTER
                cv2.destroyWindow(title)
                return True
            if k == 27:        # ESC
                cv2.destroyAllWindows()
                return False
    else:
        # 网络模式：原有的 READY/ACK 三步握手等待
        return wait_both_ready(title, sock, is_host, network, lines)

# ... 其余代码保持不变 (handshake, main loop, gesture detection, etc.)
# ----------------------------------------------------------------------------
# AI 策略 & 工具
# ----------------------------------------------------------------------------
def assisted_ai_choice(player, p_ai):
    p_draw = ASSIST_DRAW_FIXED; r=random.random()
    mapping={'rock':('paper','rock','scissors'),
             'paper':('scissors','paper','rock'),
             'scissors':('rock','scissors','paper')}
    a,p,s = mapping.get(player,('rock','paper','scissors'))
    if r<p_ai: return a
    if r<p_ai+p_draw: return p
    return s

def highmode_ai_choice(player, p_ai):
    rem = 1-p_ai; p_draw=rem/2; r=random.random()
    mapping={'rock':('paper','rock','scissors'),
             'paper':('scissors','paper','rock'),
             'scissors':('rock','scissors','paper')}
    a,p,s = mapping.get(player,('rock','paper','scissors'))
    if r<p_ai: return a
    if r<p_ai+p_draw: return p
    return s

def get_local_ip():
    try:
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(("8.8.8.8",80))
        ip=s.getsockname()[0]; s.close()
        return ip
    except:
        return "127.0.0.1"

def run_template_collection_window():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera"); return

    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cv2.namedWindow("Template Collector")
    info = "Press R/P/S to save rock/paper/scissors   ESC: quit"

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0].landmark
            mp.solutions.drawing_utils.draw_landmarks(
                frame, res.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS
            )
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('r'), ord('R')):
                gesture_templates["fist"] = {str(i): normalize_landmarks(lm)[i] for i in FINGERTIPS}
                save_templates(gesture_templates)
                print("✅ Rock (fist) template saved")
            elif k in (ord('p'), ord('P')):
                gesture_templates["open"] = {str(i): normalize_landmarks(lm)[i] for i in FINGERTIPS}
                save_templates(gesture_templates)
                print("✅ Paper (open) template saved")
            elif k in (ord('s'), ord('S')):
                gesture_templates["scissors"] = {str(i): normalize_landmarks(lm)[i] for i in FINGERTIPS}
                save_templates(gesture_templates)
                print("✅ Scissors template saved")
            elif k == 27:
                break
        else:
            cv2.putText(frame, "Show one hand", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Template Collector", frame)

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    init_csv()
    while True:
        mode = popup_mode_selection()  # 1=Local,2=Host,3=Client
        sock = None; is_host=False; network=False
        if mode==2:
            sock = setup_host()
            if sock is None: continue
            is_host, network = True, True
        elif mode==3:
            sock = setup_client()
            if sock is None: continue
            is_host, network = False, True
        elif mode == 4:
            run_template_collection_window()
            continue  # 回到模式选择

        # 初始化摄像头、Mediapipe、分数等，随后进入回合循环
        # ... (其余流程保持不变) ...
        # 摄像头 & Mediapipe Hands
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera"); sys.exit(1)
        hands = mp.solutions.hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        # 计分 & 难度
        score = wins = losses = draws = 0
        assist_active, assist_prob = False, ASSIST_WIN_PROB
        high_active,   high_prob   = False, HIGH_WIN_PROB

        round_id    = 1
        last_best   = None
        last_result = None

        # 回合循环
        while True:
            cont = show_next_round_popup(WINDOW_NAME, round_id, score, last_best)
            if cont is False:  # ESC 退出
                sys.exit(0)
            if cont == "back":
                # 回到模式选择
                break

            # 自适应难度开关
            if score >= HIGH_ENTER and not high_active:
                high_active,assist_active = True,False; high_prob=HIGH_WIN_PROB
            if high_active and score < HIGH_ENTER:
                high_active=False
            if score < ASSIST_ENTER and not assist_active and not high_active:
                assist_active,assist_prob = True,ASSIST_WIN_PROB
            if assist_active and score >= ASSIST_EXIT:
                assist_active=False

            # 准备阶段
            t0=time.time()
            while time.time()-t0<PREP_SECONDS:
                ret,frm = cap.read(); frm=cv2.flip(frm,1)
                rem = int(PREP_SECONDS-(time.time()-t0))
                cv2.putText(frm,f"Ready: {rem}s",(20,80),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),3)
                cv2.imshow(WINDOW_NAME,frm)
                if cv2.waitKey(1)&0xFF==27:
                    sys.exit(0)

            # 采集阶段
            buf = deque(maxlen=SMOOTH_WINDOW)
            best_total = best_shape = best_open = 0.0
            player_gesture = "None"
            t1=time.time()
            while time.time()-t1 < CAPTURE_SECONDS:
                ret,frm = cap.read(); frm=cv2.flip(frm,1)
                rgb = cv2.cvtColor(frm,cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    lm = res.multi_hand_landmarks[0].landmark
                    g  = classify_rps(lm)
                    if g in ("rock","paper","scissors"):
                        player_gesture = g
                    # 模板采集
                    if cv2.waitKey(1)&0xFF==ord('t') and g in ("rock","paper","scissors"):
                        key = {"rock":"fist","paper":"open","scissors":"scissors"}[g]
                        norm=normalize_landmarks(lm)
                        gesture_templates[key] = {str(i):norm[i] for i in FINGERTIPS}
                        save_templates(gesture_templates)
                    # 计算完成度
                    norm = normalize_landmarks(lm)
                    base = {"rock":"fist","paper":"open","scissors":"scissors"}.get(g,None)
                    s_val=o_val=0.0
                    if base and has_template(base):
                        devs = compute_devs(norm,gesture_templates[base])
                        s_val = shape_score(devs,base)
                        # 可视化偏差
                        y0=180
                        for tip,dv in devs.items():
                            col = (0,255,0) if dv<DEVIATION_THRESH else (0,0,255)
                            cv2.putText(frm,f"{tip}:{dv:.3f}",(20,y0),
                                        cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)
                            y0+=20
                    o_val = openness_ratio(lm)*100
                    t_val=total_score(g,s_val,o_val)
                    buf.append(t_val)
                    avg = sum(buf)/len(buf)
                    if avg>best_total:
                        best_total,best_shape,best_open = avg,s_val,o_val
                    # 显示 Shape/Open/Total
                    cv2.putText(frm,
                        f"Shape:{s_val:.1f}%  Open:{o_val:.1f}%  Total:{avg:.1f}%",
                        (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2
                    )
                    mp.solutions.drawing_utils.draw_landmarks(
                        frm,res.multi_hand_landmarks[0],mp.solutions.hands.HAND_CONNECTIONS
                    )
                cv2.imshow(WINDOW_NAME,frm)
                if cv2.waitKey(1)&0xFF==27:
                    sys.exit(0)

            # 对手出拳
            if player_gesture in ("rock","paper","scissors"):
                if network:
                    try:
                        sock.sendall(player_gesture.encode())
                        opp = sock.recv(16).decode().strip()
                    except:
                        opp = random.choice(["rock","paper","scissors"])
                else:
                    if high_active:
                        opp = highmode_ai_choice(player_gesture,high_prob)
                    elif assist_active:
                        opp = assisted_ai_choice(player_gesture,assist_prob)
                    else:
                        opp = random.choice(["rock","paper","scissors"])
                result = judge(player_gesture,opp)
                if result=="You Win!":
                    score+=3; wins+=1; high_prob=min(HIGH_MAX,high_prob+HIGH_INC)
                elif result=="You Lose!":
                    score-=1; losses+=1; assist_prob=max(ASSIST_MIN,assist_prob-ASSIST_DEC)
                    high_prob=max(HIGH_MIN,high_prob-HIGH_DEC)
                else:
                    draws+=1
            else:
                opp="None"; result="No gesture!"

            # 结果展示
            ret,frm = cap.read(); frm=cv2.flip(frm,1)
            cv2.putText(frm,f"You: {player_gesture}",(10,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            cv2.putText(frm,f"Opp: {opp}",(10,80),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(frm,result,(10,130),
                        cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),3)
            cv2.imshow(WINDOW_NAME,frm)
            cv2.waitKey(3000)

            # 记录 CSV
            append_csv(
                round_id,
                "Local"  if mode==1 else "Host" if mode==2 else "Client",
                player_gesture, opp, result,
                best_total, best_shape, best_open,
                score, wins, losses, draws,
                high_active, high_prob, assist_active, assist_prob
            )
            last_best   = best_total
            last_result = result
            round_id   += 1

        # 清理并回到模式选择
        cap.release()
        hands.close()
        cv2.destroyAllWindows()
    # end while True (模式循环)
