#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
version_4.3 同步改进版 (双人对战增加"双方准备"机制)
=================================================
改动目标:
  1. **首回合不立即开始**: 联机 (Host/Client) 握手成功后, 先显示提示, 双方都按下 Enter 才开始 Round 1。
  2. **每一回合结束后进入下一回合前也需要双方都按 Enter**: 防止一方过快进入下一轮导致另一方措手不及。
  3. **处理一方提前按 Enter 的情况**: 一方按下后发送 READY 标记, 等待对方; 界面显示"已准备, 等待对方..."。
  4. **支持超时/断线降级**: 如果等待过程中网络断开或超时, 自动降级为本地 AI 模式, 不再阻塞。

核心新增:
  * 函数 `wait_both_ready(window, sock, is_host, remote_mode, label)` 用于同步。
  * 使用简单文本协议: 发送字节串 `READY\n` 表示已准备。
  * Host / Client 都在循环中:
        - 本地检测是否按下 Enter。
        - 若按下且尚未发送则发送 READY。
        - 检查是否收到对方 READY。
        - 两者均 True 则返回继续游戏。
  * 支持 ESC 退出; 超时 (默认 120 秒) -> 自动降级或退出。

保持不变:
  * 自适应难度 (Assist / High) 概念与概率逻辑。
  * 完成度计算 / 模板管理 / CSV 记录 / 握手 / 定向手势交换。

可调参数:
  * READY_TIMEOUT = 120   # 最长等待时间 (秒)
  * READY_POLL_DELAY = 0.05  # 轮询间隔

如需: 改成定时器倒计时 / 添加聊天消息 / 多人扩展, 后续可再加。
"""
from __future__ import annotations
import cv2, mediapipe as mp, random, time, os, csv, json, math, datetime, socket, sys, select
from collections import deque
from typing import Dict, List, Tuple, Optional
import numpy as np

# =============================================================================
# 1. 配置 / Configuration (与前版本保持一致 + 新增同步参数)
# =============================================================================
TEMPLATE_FILE = "gesture_templates.json"
CSV_FILE      = "rps_open_fist_completion.csv"
SMOOTH_WINDOW = 5
CAPTURE_SECONDS = 5
PREP_SECONDS    = 2

SHAPE_WEIGHT_OPEN   = 0.7
OPEN_WEIGHT_OPEN    = 0.3
SHAPE_WEIGHT_FIST   = 0.6
CLOSURE_WEIGHT_FIST = 0.4

DEVIATION_GREEN_THRESHOLD = 0.05

FINGERTIPS  = [4, 8, 12, 16, 20]
PALM_POINTS = [0, 5, 9, 13, 17]

# Assist Mode
ASSIST_ENTER_SCORE        = -3
ASSIST_EXIT_SCORE         = 3
ASSIST_AI_WIN_PROB_INITIAL= 0.30
ASSIST_AI_WIN_DEC         = 0.05
ASSIST_AI_WIN_PROB_MIN    = 0.20
ASSIST_DRAW_FIXED         = 0.30
# High Mode
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

# 准备同步 (新增)
READY_TIMEOUT = 120        # 等待双方准备超时(秒)
READY_POLL_DELAY = 0.05
READY_TOKEN = b"READY\n"

# =============================================================================
# 2. 模板加载 / 保存
# =============================================================================
def load_templates() -> Dict[str, Dict[str, List[float]]]:
    if not os.path.exists(TEMPLATE_FILE):
        data = {"open": {}, "fist": {}}
        with open(TEMPLATE_FILE, 'w') as f: json.dump(data, f, indent=2)
        return data
    with open(TEMPLATE_FILE, 'r') as f:
        data = json.load(f)
    changed = False
    for k in ("open","fist"):
        if k not in data:
            data[k] = {}; changed = True
    if changed:
        with open(TEMPLATE_FILE,'w') as f: json.dump(data,f,indent=2)
    return data

def save_templates(tpl: Dict[str, Dict[str, List[float]]]):
    with open(TEMPLATE_FILE,'w') as f: json.dump(tpl,f,indent=2)

gesture_templates = load_templates()

def has_template(name: str) -> bool:
    return name in gesture_templates and len(gesture_templates[name])>0

# =============================================================================
# 3. 几何 & 归一化
# =============================================================================
def palm_center_and_width(lm_list) -> Tuple[float,float,float]:
    cx = sum(lm_list[i].x for i in PALM_POINTS)/len(PALM_POINTS)
    cy = sum(lm_list[i].y for i in PALM_POINTS)/len(PALM_POINTS)
    w  = math.dist((lm_list[5].x,lm_list[5].y),(lm_list[17].x,lm_list[17].y)) or 1e-6
    return cx,cy,w

def normalize_landmarks(lm_list) -> Dict[int,Tuple[float,float]]:
    cx,cy,w = palm_center_and_width(lm_list)
    return {i:((lm.x-cx)/w,(lm.y-cy)/w) for i,lm in enumerate(lm_list)}

def openness_ratio(lm_list) -> float:
    cx,cy,w = palm_center_and_width(lm_list)
    ds=[math.dist((lm_list[t].x,lm_list[t].y),(cx,cy))/w for t in FINGERTIPS]
    avg=sum(ds)/len(ds)
    return min(1.0, avg/0.9)

# =============================================================================
# 4. 评分函数
# =============================================================================
def compute_devs(norm: Dict[int,Tuple[float,float]], template: Dict[str,List[float]])->Dict[int,float]:
    devs={}
    for k,(tx,ty) in template.items():
        idx=int(k)
        if idx in norm:
            x,y=norm[idx]
            devs[idx]=math.hypot(x-tx,y-ty)
    return devs

def finger_weight(idx:int, base:str)->float:
    if base=="open": return 1.2 if idx==4 else 1.0
    if base=="fist": return 1.1 if idx in (8,12) else 1.0
    return 1.0

def shape_score(devs:Dict[int,float], base:str)->float:
    if not devs: return 0.0
    sims=[math.exp(-8*d)*finger_weight(i,base) for i,d in devs.items()]
    return (sum(sims)/len(sims))*100

def total_open(shape_part:float, open_part:float)->float:
    return SHAPE_WEIGHT_OPEN*shape_part + OPEN_WEIGHT_OPEN*open_part

def total_fist(shape_part:float, open_part:float)->float:
    closure=100-open_part
    return SHAPE_WEIGHT_FIST*shape_part + CLOSURE_WEIGHT_FIST*closure

# =============================================================================
# 5. 手势分类
# =============================================================================
def classify_rps(lm_list)->str:
    tips=[8,12,16,20]
    fingers=[1 if lm_list[t].y < lm_list[t-2].y else 0 for t in tips]
    c=sum(fingers)
    if c==0: return "rock"
    if c==2: return "scissors"
    if c>=4: return "paper"
    return "unknown"

# =============================================================================
# 6. CSV
# =============================================================================
def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE,'w',newline='') as f:
            csv.writer(f).writerow([
                "Time","Round","PlayerGesture","OpponentGesture","Result",
                "Total","Shape","Open","Score","Wins","Losses","Draws",
                "HighModeActive","HighWinProb","AssistActive","AssistWinProb"
            ])

def append_csv(ts:str,rnd:int,pg:str,og:str,res:str,
               total:Optional[float], shape:Optional[float], open_pct:Optional[float],
               score:int, w:int,l:int,d:int,
               high_active:bool, high_win_prob:float,
               assist_active:bool, assist_win_prob:float):
    with open(CSV_FILE,'a',newline='') as f:
        csv.writer(f).writerow([
            ts,rnd,pg,og,res,
            ("" if total is None else f"{total:.2f}"),
            ("" if shape is None else f"{shape:.2f}"),
            ("" if open_pct is None else f"{open_pct:.2f}"),
            score,w,l,d,
            int(high_active), f"{high_win_prob:.2f}" if high_active else "",
            int(assist_active), f"{assist_win_prob:.2f}" if assist_active else ""
        ])

# =============================================================================
# 7. 胜负判定
# =============================================================================
def judge(player:str,opp:str)->str:
    if player==opp: return "Draw"
    if (player=="rock" and opp=="scissors") or \
       (player=="scissors" and opp=="paper") or \
       (player=="paper" and opp=="rock"):
        return "You Win!"
    return "You Lose!"

# =============================================================================
# 8. 网络 & 握手 & 同步
# =============================================================================
def get_local_ip()->str:
    try:
        s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        s.connect(("8.8.8.8",80))
        ip=s.getsockname()[0]
        s.close(); return ip
    except Exception:
        return "127.0.0.1"

def setup_host()->Optional[socket.socket]:
    try:
        srv=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        srv.bind((HOST_LISTEN,PORT))
        srv.listen(1)
        ip=get_local_ip()
        print(f"[Host] 等待客户端连接... (IP: {ip}:{PORT})")
        conn,addr=srv.accept(); print(f"[Host] 客户端已连接: {addr}")
        return conn
    except Exception as e:
        print("[Host] 启动失败:",e); return None

def setup_client()->Optional[socket.socket]:
    ip=input("输入主机IP: ").strip()
    print("IP debug:", [ord(c) for c in ip], "->", ip)
    try:
        cli=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        print(f"[Client] 连接 {ip}:{PORT} ...")
        cli.connect((ip,PORT))
        print("[Client] 连接成功!")
        return cli
    except Exception as e:
        print("[Client] 连接失败:",e); return None

def handshake(sock:socket.socket, is_host:bool, timeout=5.0)->bool:
    try:
        sock.settimeout(timeout)
        if is_host:
            data=sock.recv(16)
            if data.decode().strip()!="HELLO_V1":
                print("[Handshake] 非预期客户端数据",data); return False
            sock.sendall(b"WELCOME_V1")
        else:
            sock.sendall(b"HELLO_V1")
            data=sock.recv(16)
            if data.decode().strip()!="WELCOME_V1":
                print("[Handshake] 非预期主机数据",data); return False
        print("[Handshake] OK")
        return True
    except Exception as e:
        print("[Handshake] 失败:",e); return False

def exchange_gesture_host(sock:socket.socket, my_gesture:str, timeout=5.0)->Optional[str]:
    try:
        sock.settimeout(timeout)
        data=sock.recv(16)
        if not data:
            print("[Host] Peer closed"); return None
        other=data.decode().strip()
        sock.sendall(my_gesture.encode())
        if other not in ("rock","paper","scissors","None"):
            print("[Host] Invalid gesture:",other); return None
        return other
    except Exception as e:
        print("[Host] 交换失败:",e); return None

def exchange_gesture_client(sock:socket.socket, my_gesture:str, timeout=5.0)->Optional[str]:
    try:
        sock.settimeout(timeout)
        sock.sendall(my_gesture.encode())
        data=sock.recv(16)
        if not data:
            print("[Client] Peer closed"); return None
        other=data.decode().strip()
        if other not in ("rock","paper","scissors","None"):
            print("[Client] Invalid gesture:",other); return None
        return other
    except Exception as e:
        print("[Client] 交换失败:",e); return None

# ---- 新增: 双方准备同步 ----
def wait_both_ready(window_name:str, sock:Optional[socket.socket], is_host:bool, remote_mode:bool, label:str)->bool:
    """在联机模式下等待双方按下 Enter; 本地模式直接返回 True.
    返回 False 表示用户 ESC 退出或网络失败降级。
    label: 用于提示 ("开始游戏" / "下一回合")"""
    if not remote_mode or not sock:
        return True
    start=time.time()
    my_ready=False
    other_ready=False
    # 为了非阻塞收数据, 将 socket 设为非阻塞
    sock.setblocking(False)

    # 创建简易提示窗口循环
    while True:
        # 检查超时
        if time.time()-start > READY_TIMEOUT:
            print("[ReadySync] 超时, 降级为本地 AI")
            return False
        # 读取键盘
        k=cv2.waitKey(1) & 0xFF
        if k==27: # ESC
            return False
        if k==13: # Enter
            if not my_ready:
                try:
                    sock.sendall(READY_TOKEN)
                    my_ready=True
                except Exception as e:
                    print("[ReadySync] 发送失败:",e); return False
        # 非阻塞接收 READY
        try:
            data=sock.recv(64)
            if data:
                # 可能一次收到多个, 简单 split
                parts=data.split(b'\n')
                for p in parts:
                    if p==b'READY' and not other_ready:
                        other_ready=True
        except BlockingIOError:
            pass
        except Exception as e:
            print("[ReadySync] 接收失败:",e); return False

        # 绘制等待界面
        canvas = 255* (1 - 0) * 0 + 0  # dummy
        # 用黑底图更直观
        frame = (0*np.ones((300,640,3),dtype='uint8'))
        txt1=f"联机同步: {label}"
        txt2=f"你(Enter): {'OK' if my_ready else '...'} | 对方: {'OK' if other_ready else '...'}"
        remain=int(READY_TIMEOUT-(time.time()-start))
        cv2.putText(frame, txt1, (30,80), cv2.FONT_HERSHEY_SIMPLEX, 1.0,(0,255,255),2)
        cv2.putText(frame, txt2, (30,140),cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2)
        cv2.putText(frame, f"剩余等待: {remain}s  (ESC退出)", (30,200), cv2.FONT_HERSHEY_SIMPLEX,0.7,(180,180,180),2)
        cv2.imshow(window_name, frame)

        if my_ready and other_ready:
            # 稍作延迟给双方一致视觉
            time.sleep(0.3)
            sock.setblocking(True)
            return True
        time.sleep(READY_POLL_DELAY)

# =============================================================================
# 9. 自适应 AI 决策
# =============================================================================
def assisted_ai_choice(player_gesture:str, ai_win_prob:float)->str:
    p_ai=ai_win_prob
    p_draw=ASSIST_DRAW_FIXED
    p_player=max(0.0,1.0-(p_ai+p_draw))
    r=random.random()
    if player_gesture=='rock':
        if r<p_ai: return 'paper'
        if r<p_ai+p_draw: return 'rock'
        return 'scissors'
    if player_gesture=='paper':
        if r<p_ai: return 'scissors'
        if r<p_ai+p_draw: return 'paper'
        return 'rock'
    if player_gesture=='scissors':
        if r<p_ai: return 'rock'
        if r<p_ai+p_draw: return 'scissors'
        return 'paper'
    return random.choice(["rock","paper","scissors"])  # fallback

def highmode_ai_choice(player_gesture:str, ai_win_prob:float)->str:
    p_ai=ai_win_prob
    remain=1.0-p_ai
    p_draw=remain/2
    r=random.random()
    if player_gesture=='rock':
        if r<p_ai: return 'paper'
        if r<p_ai+p_draw: return 'rock'
        return 'scissors'
    if player_gesture=='paper':
        if r<p_ai: return 'scissors'
        if r<p_ai+p_draw: return 'paper'
        return 'rock'
    if player_gesture=='scissors':
        if r<p_ai: return 'rock'
        if r<p_ai+p_draw: return 'scissors'
        return 'paper'
    return random.choice(["rock","paper","scissors"])  # fallback

# =============================================================================
# 10. 主流程
# =============================================================================
def main():
    print("==== 模式选择 / Mode Select ====")
    print("1) 本地 vs AI")
    print("2) 联机 (Host)")
    print("3) 联机 (Client)")
    mode=input(">>> ").strip()

    remote_mode=False
    net_sock:Optional[socket.socket]=None
    is_host=False
    if mode=='2':
        net_sock=setup_host(); remote_mode=net_sock is not None; is_host=True
    elif mode=='3':
        net_sock=setup_client(); remote_mode=net_sock is not None; is_host=False
    else:
        print("进入单人 AI 模式 ...")

    if mode in ('2','3') and not remote_mode:
        print("联机失败, 回退单人模式。")

    if remote_mode:
        if not handshake(net_sock, is_host):
            print("握手失败, 回退本地模式")
            remote_mode=False

    init_csv()

    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头"); return

    wins=losses=draws=0
    score=0
    round_id=0

    assist_active=False
    assist_ai_win_prob=ASSIST_AI_WIN_PROB_INITIAL
    high_active=False
    high_ai_win_prob=HIGH_AI_WIN_PROB_INITIAL

    window_name="RPS Completion Adaptive"

    # --- 首回合开始前同步 ---
    if remote_mode:
        ok=wait_both_ready(window_name, net_sock, is_host, remote_mode, label="开始 Round 1")
        if not ok:
            if remote_mode:
                print("准备阶段失败或超时, 降级本地 AI")
            remote_mode=False

    mp_hands=mp.solutions.hands
    with mp_hands.Hands(max_num_hands=1,min_detection_confidence=0.7,min_tracking_confidence=0.5) as hands:
        while True:
            round_id+=1
            # 模式切换 (开局)
            if not remote_mode and score>=HIGH_ENTER_SCORE:
                if not high_active:
                    high_active=True
                    high_ai_win_prob=HIGH_AI_WIN_PROB_INITIAL
                    if assist_active: assist_active=False
                    print(f"[High] 激活 score={score} win_prob={high_ai_win_prob:.2f}")
            else:
                if high_active and score<HIGH_ENTER_SCORE:
                    high_active=False
                    print(f"[High] 关闭 score={score}")
            if (not remote_mode) and (not high_active) and (score<ASSIST_ENTER_SCORE):
                if not assist_active:
                    assist_active=True
                    assist_ai_win_prob=ASSIST_AI_WIN_PROB_INITIAL
                    print(f"[Assist] 激活 score={score} win_prob={assist_ai_win_prob:.2f}")
            if assist_active and score>=ASSIST_EXIT_SCORE:
                assist_active=False
                print(f"[Assist] 关闭 score={score}")

            # 准备阶段倒计时
            prep_start=time.time()
            while time.time()-prep_start<PREP_SECONDS:
                ret,frame=cap.read();
                if not ret: break
                frame=cv2.flip(frame,1)
                remain=PREP_SECONDS-int(time.time()-prep_start)
                cv2.putText(frame,f"Round {round_id} Ready:{remain}", (20,80), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),3)
                status=f"Score:{score} W:{wins} L:{losses} D:{draws}"
                if high_active: status+=f" | High winP={high_ai_win_prob:.2f}"
                elif assist_active: status+=f" | Assist winP={assist_ai_win_prob:.2f}"
                if remote_mode: status+= " | NET"
                cv2.putText(frame,status,(10,470),cv2.FONT_HERSHEY_SIMPLEX,0.63,(255,255,255),2)
                cv2.imshow(window_name,frame)
                if cv2.waitKey(1) & 0xFF==27:
                    cap.release();
                    if net_sock: net_sock.close(); cv2.destroyAllWindows(); return

            # 采集阶段
            capture_start=time.time()
            player_gesture="None"; gesture_captured=False
            best_total=best_shape=best_open=None
            total_window=deque(maxlen=SMOOTH_WINDOW)

            while time.time()-capture_start<CAPTURE_SECONDS:
                ret,frame=cap.read();
                if not ret: break
                frame=cv2.flip(frame,1)
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                results=hands.process(rgb)
                if results.multi_hand_landmarks:
                    lm_list=results.multi_hand_landmarks[0].landmark
                    cur=classify_rps(lm_list)
                    if cur in ("rock","paper","scissors"):
                        player_gesture=cur; gesture_captured=True
                    ktemp=cv2.waitKey(1) & 0xFF
                    if ktemp==ord('t') and player_gesture in ("rock","paper"):
                        base='fist' if player_gesture=='rock' else 'open'
                        norm_cap=normalize_landmarks(lm_list)
                        gesture_templates[base]={str(i):norm_cap[i] for i in FINGERTIPS}
                        save_templates(gesture_templates)
                        print(f"[模板更新] {base}")
                    if player_gesture in ("rock","paper"):
                        base='fist' if player_gesture=='rock' else 'open'
                        norm=normalize_landmarks(lm_list)
                        if has_template(base):
                            devs=compute_devs(norm, gesture_templates[base])
                            shape=shape_score(devs, base)
                        else:
                            devs={}; shape=0.0
                        open_pct=openness_ratio(lm_list)*100
                        total_raw= total_fist(shape, open_pct) if base=='fist' else total_open(shape, open_pct)
                        total_window.append(total_raw)
                        total_disp=sum(total_window)/len(total_window)
                        if (best_total is None) or (total_disp>best_total):
                            best_total=total_disp; best_shape=shape; best_open=open_pct
                        y0=180
                        if devs:
                            for tip,dval in devs.items():
                                color=(0,255,0) if dval<DEVIATION_GREEN_THRESHOLD else (0,0,255)
                                cv2.putText(frame,f"{tip}:{dval:.3f}",(20,y0),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
                                y0+=18
                        closure=100-open_pct
                        cv2.putText(frame,f"Shape:{shape:.1f}% Open:{open_pct:.1f}% Clo:{closure:.1f}%",(20,140),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,255),2)
                        cv2.putText(frame,f"Total:{total_disp:.1f}% (Best:{best_total:.1f}%)",(20,115),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
                    else:
                        cv2.putText(frame,"Completion: N/A (scissors)",(20,140),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),2)
                    mp.solutions.drawing_utils.draw_landmarks(frame, results.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
                remain=CAPTURE_SECONDS-int(time.time()-capture_start)
                cv2.putText(frame,f"Show ({remain})",(20,60),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
                cv2.putText(frame,f"Gesture:{player_gesture}",(20,35),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
                footer="[t]模板 rock/paper  ESC退出"
                if high_active: footer+=f"  High winP={high_ai_win_prob:.2f}"
                elif assist_active: footer+=f"  Assist winP={assist_ai_win_prob:.2f}"
                if remote_mode: footer+="  NET"
                cv2.putText(frame, footer,(240,470),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)
                cv2.imshow(window_name,frame)
                if cv2.waitKey(1) & 0xFF==27:
                    cap.release();
                    if net_sock: net_sock.close(); cv2.destroyAllWindows(); return

            # 对手手势
            if remote_mode and net_sock:
                if is_host:
                    opp_gesture=exchange_gesture_host(net_sock, player_gesture if gesture_captured else "None")
                else:
                    opp_gesture=exchange_gesture_client(net_sock, player_gesture if gesture_captured else "None")
                if opp_gesture is None:
                    print("[Network] 通信异常 -> 降级本地 AI")
                    remote_mode=False
                    opp_gesture=random.choice(["rock","paper","scissors"])
            else:
                if not gesture_captured or player_gesture not in ("rock","paper","scissors"):
                    player_gesture="None"
                    opp_gesture=random.choice(["rock","paper","scissors"])
                else:
                    if high_active:
                        opp_gesture=highmode_ai_choice(player_gesture, high_ai_win_prob)
                    elif assist_active:
                        opp_gesture=assisted_ai_choice(player_gesture, assist_ai_win_prob)
                    else:
                        opp_gesture=random.choice(["rock","paper","scissors"])

            # 判定 & 更新分 / 模式内部概率
            if player_gesture in ("rock","paper","scissors"):
                result=judge(player_gesture, opp_gesture)
                if result=="You Win!":
                    score+=3; wins+=1
                    if high_active:
                        high_ai_win_prob=min(HIGH_AI_WIN_MAX, high_ai_win_prob + HIGH_AI_WIN_INC)
                elif result=="You Lose!":
                    score-=1; losses+=1
                    if assist_active:
                        assist_ai_win_prob=max(ASSIST_AI_WIN_PROB_MIN, assist_ai_win_prob - ASSIST_AI_WIN_DEC)
                    if high_active:
                        high_ai_win_prob=max(HIGH_AI_WIN_MIN, high_ai_win_prob - HIGH_AI_WIN_DEC)
                else:
                    draws+=1
            else:
                result="No gesture!"; player_gesture="None"

            # 回合末模式边界再次检查
            if high_active and score<HIGH_ENTER_SCORE:
                high_active=False
                print(f"[High] 关闭 score={score}")
            if assist_active and score>=ASSIST_EXIT_SCORE:
                assist_active=False
                print(f"[Assist] 关闭 score={score}")
            if (not high_active) and (not assist_active):
                if score>=HIGH_ENTER_SCORE:
                    high_active=True; high_ai_win_prob=HIGH_AI_WIN_PROB_INITIAL
                    print(f"[High] 激活 score={score} win_prob={high_ai_win_prob:.2f}")
                elif score<ASSIST_ENTER_SCORE:
                    assist_active=True; assist_ai_win_prob=ASSIST_AI_WIN_PROB_INITIAL
                    print(f"[Assist] 激活 score={score} win_prob={assist_ai_win_prob:.2f}")

            # 结果显示
            ret,frame=cap.read();
            if not ret: break
            frame=cv2.flip(frame,1)
            show_total=best_total if player_gesture in ("rock","paper") else None
            show_shape=best_shape if player_gesture in ("rock","paper") else None
            show_open=best_open if player_gesture in ("rock","paper") else None
            cv2.putText(frame,f"You: {player_gesture}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            who="Opponent" if remote_mode else "AI"
            cv2.putText(frame,f"{who}: {opp_gesture}",(10,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(frame,result,(10,125),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),3)
            if show_total is not None:
                cv2.putText(frame,f"Best Total:{show_total:.1f}% Shape:{show_shape:.1f}% Open:{show_open:.1f}%",(10,170),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            else:
                cv2.putText(frame,"Best Total: N/A (scissors / none)",(10,170),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),2)
            status=f"Score:{score} W:{wins} L:{losses} D:{draws}"
            if high_active: status+=f" | High winP={high_ai_win_prob:.2f}"
            elif assist_active: status+=f" | Assist winP={assist_ai_win_prob:.2f}"
            if remote_mode: status+=" | NET"
            cv2.putText(frame,status,(10,470),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
            msg_next = "ENTER下一回合 (同步)  ESC退出" if remote_mode else "ENTER下一回合  ESC退出"
            cv2.putText(frame,msg_next,(10,440),cv2.FONT_HERSHEY_SIMPLEX,0.65,(180,180,180),2)
            cv2.imshow(window_name,frame)

            ts=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_csv(ts, round_id, player_gesture, opp_gesture, result,
                       show_total, show_shape, show_open, score, wins, losses, draws,
                       high_active and (not remote_mode), high_ai_win_prob if high_active else 0.0,
                       assist_active and (not remote_mode), assist_ai_win_prob if assist_active else 0.0)

            if remote_mode and net_sock:
                ok=wait_both_ready(window_name, net_sock, is_host, remote_mode, label="下一回合")
                if not ok:
                    print("[ReadySync] 失败/降级, 后续改为本地 AI")
                    remote_mode=False
            else:
                # 本地等待 Enter 或 ESC
                while True:
                    k=cv2.waitKey(0) & 0xFF
                    if k==13: break
                    if k==27:
                        cap.release();
                        if net_sock: net_sock.close(); cv2.destroyAllWindows(); return

    cap.release()
    if net_sock: net_sock.close()
    cv2.destroyAllWindows()

# =============================================================================
# 入口
# =============================================================================
if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断 / Interrupted")
        sys.exit(0)
