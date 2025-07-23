#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
version_4.4 Popup Ready + Scissors Completion
============================================
功能：
  * Rock/Paper/Scissors 手势识别
  * 仅对 Rock(Fist)、Paper(Open)、Scissors 计算完成度
  * 模板采集：按 't' 针对三种手势采集指尖归一化坐标
  * 自适应难度：Assist (<-3)、Normal、High (>=10)
  * 联机模式 (Host/Client)：握手(HELLO_V1/WELCOME_V1)、方向化交换
  * 双方同步：回合开始前弹窗，双方按 Enter
  * 回合结束展示结果 3 秒，再弹窗提示下一局
  * CSV 日志记录各回合数据，包括完成度与模式状态

按键：
  - t      采集模板 (rock, paper, scissors)
  - Enter  准备 / 确认
  - Esc    退出
"""
import cv2, mediapipe as mp, random, time, os, csv, json, math, datetime, socket, sys
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional

# =============================================================================
# 1. 配置
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

# Assist 模式
ASSIST_ENTER_SCORE        = -3
ASSIST_EXIT_SCORE         = 3
ASSIST_AI_WIN_PROB_INITIAL= 0.30
ASSIST_AI_WIN_DEC         = 0.05
ASSIST_AI_WIN_PROB_MIN    = 0.20
ASSIST_DRAW_FIXED         = 0.30

# High 模式
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

WINDOW_NAME = "RPS Completion Adaptive"

# =============================================================================
# 2. 模板管理
# =============================================================================
def load_templates() -> Dict[str, Dict[str, List[float]]]:
    if not os.path.exists(TEMPLATE_FILE):
        data = {"open": {}, "fist": {}, "scissors": {}}
        with open(TEMPLATE_FILE, 'w') as f: json.dump(data, f, indent=2)
        return data
    with open(TEMPLATE_FILE, 'r') as f:
        data = json.load(f)
    changed = False
    for k in ("open","fist","scissors"):
        if k not in data:
            data[k] = {}
            changed = True
    if changed:
        with open(TEMPLATE_FILE, 'w') as f: json.dump(data, f, indent=2)
    return data

def save_templates(tpl: Dict[str, Dict[str, List[float]]]) -> None:
    with open(TEMPLATE_FILE, 'w') as f: json.dump(tpl, f, indent=2)

gesture_templates = load_templates()

def has_template(name: str) -> bool:
    return name in gesture_templates and len(gesture_templates[name]) > 0

# =============================================================================
# 3. 几何归一化
# =============================================================================
def palm_center_and_width(lm_list) -> Tuple[float, float, float]:
    cx = sum(lm_list[i].x for i in PALM_POINTS) / len(PALM_POINTS)
    cy = sum(lm_list[i].y for i in PALM_POINTS) / len(PALM_POINTS)
    w = math.dist((lm_list[5].x, lm_list[5].y), (lm_list[17].x, lm_list[17].y)) or 1e-6
    return cx, cy, w

def normalize_landmarks(lm_list) -> Dict[int, Tuple[float,float]]:
    cx, cy, w = palm_center_and_width(lm_list)
    return {i: ((lm.x - cx)/w, (lm.y - cy)/w) for i, lm in enumerate(lm_list)}

def openness_ratio(lm_list) -> float:
    cx, cy, w = palm_center_and_width(lm_list)
    ds = [math.dist((lm_list[t].x, lm_list[t].y), (cx, cy)) / w for t in FINGERTIPS]
    avg = sum(ds)/len(ds)
    return min(1.0, avg/0.9)

# =============================================================================
# 4. 完成度计算
# =============================================================================
def finger_weight(idx: int, base: str) -> float:
    if base == "open": return 1.2 if idx == 4 else 1.0
    if base == "fist": return 1.1 if idx in (8,12) else 1.0
    if base == "scissors": return 1.0
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

def compute_devs(norm: Dict[int,Tuple[float,float]], template: Dict[str,List[float]]) -> Dict[int,float]:
    devs = {}
    for k,(tx,ty) in template.items():
        idx = int(k)
        if idx in norm:
            x,y = norm[idx]
            devs[idx] = math.hypot(x-tx, y-ty)
    return devs

# =============================================================================
# 5. 手势分类 & 胜负判定
# =============================================================================
def classify_rps(lm_list) -> str:
    tips = [8,12,16,20]
    fingers = [1 if lm_list[t].y < lm_list[t-2].y else 0 for t in tips]
    c = sum(fingers)
    if c == 0: return "rock"
    if c == 2: return "scissors"
    if c >= 4: return "paper"
    return "unknown"

def judge(player: str, opp: str) -> str:
    if player == opp: return "Draw"
    if (player=="rock" and opp=="scissors") or \
       (player=="scissors" and opp=="paper") or \
       (player=="paper" and opp=="rock"): return "You Win!"
    return "You Lose!"

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

def append_csv(ts, rnd, pg, og, res, total, shape, open_pct,
               score, w, l, d, high_active, high_prob, assist_active, assist_prob):
    with open(CSV_FILE,'a',newline='') as f:
        csv.writer(f).writerow([
            ts, rnd, pg, og, res,
            ("" if total is None else f"{total:.2f}"),
            ("" if shape is None else f"{shape:.2f}"),
            ("" if open_pct is None else f"{open_pct:.2f}"),
            score, w, l, d,
            int(high_active), f"{high_prob:.2f}" if high_active else "",
            int(assist_active), f"{assist_prob:.2f}" if assist_active else ""
        ])

# =============================================================================
# 7. 网络
# =============================================================================
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"

def setup_host():
    try:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind((HOST_LISTEN, PORT))
        srv.listen(1)
        ip = get_local_ip()
        print(f"[Host] 等待客户端连接... (IP: {ip}:{PORT})")
        conn, addr = srv.accept()
        print(f"[Host] 客户端已连接: {addr}")
        return conn
    except Exception as e:
        print("[Host] 启动失败:", e)
        return None

def setup_client():
    ip = input("输入主机IP: ").strip()
    try:
        cli = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        cli.connect((ip, PORT))
        print("[Client] 连接成功!")
        return cli
    except Exception as e:
        print("[Client] 连接失败:", e)
        return None

def handshake(sock, is_host, timeout=5.0):
    sock.settimeout(timeout)
    try:
        if is_host:
            data = sock.recv(16)
            if data.decode().strip() != "HELLO_V1": return False
            sock.sendall(b"WELCOME_V1")
        else:
            sock.sendall(b"HELLO_V1")
            data = sock.recv(16)
            if data.decode().strip() != "WELCOME_V1": return False
        print("[Handshake] OK")
        return True
    except Exception as e:
        print("[Handshake] Failed:", e)
        return False

# =============================================================================
# 8. 同步与弹窗
# =============================================================================
def wait_both_ready(window_name, sock, is_host, remote_mode,
                    label="START", timeout=120.0, extra_lines=None):
    start = time.time()
    local_ready = False
    remote_ready = False
    while True:
        frame = np.zeros((300, 640, 3), dtype='uint8')
        # 绘制 lines
        if extra_lines:
            y0 = 60
            for t in extra_lines:
                cv2.putText(frame, t, (40, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255),2)
                y0 += 60
        else:
            cv2.putText(frame, label, (40,80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255),2)
        status = []
        if local_ready: status.append("YOU READY")
        if remote_ready: status.append("PEER READY")
        cv2.putText(frame, " | ".join(status) if status else "Waiting...",
                    (40, y0 if extra_lines else 160), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
        cv2.imshow(window_name, frame)
        k = cv2.waitKey(100) & 0xFF
        # 本地按键
        if k == 13 and not local_ready:
            local_ready = True
            try: sock.sendall(b"READY")
            except: remote_mode=False
        if k == 27:
            return False
        # 网络消息
        if remote_mode:
            try:
                sock.settimeout(0.01)
                data = sock.recv(16)
                if data and data.decode()=="READY":
                    remote_ready = True
            except: pass
        # 超时
        if time.time()-start > timeout:
            return False
        # 双方准备完毕
        if (not remote_mode and local_ready) or (local_ready and remote_ready):
            return True


def show_next_round_popup(window_name, upcoming_round,
                          last_result, last_best_total,
                          is_network, sock, is_host,
                          timeout=120.0, extra_lines=None):
    READY_PROMPT = "Press ENTER to start"
    START_LABEL = f"Round {upcoming_round}"
    lines = [START_LABEL]
    if upcoming_round > 1:
        if last_result:
            res = ""
            if "Win" in last_result: res="Win"
            elif "Lose" in last_result: res="Lose"
            elif "Draw" in last_result: res="Draw"
            else: res=last_result
            lines.append(f"Last Result: {res}")
        if last_best_total is not None:
            lines.append(f"Last Best Total: {last_best_total:.1f}%")
    if not is_network:
        lines.append(READY_PROMPT)
    if is_network and sock:
        return wait_both_ready(window_name, sock, is_host, True,
                               label=START_LABEL,
                               extra_lines=lines,
                               timeout=timeout)
    while True:
        frame = np.zeros((360, 720, 3), dtype='uint8')
        y = 80
        for text in lines:
            color = (0,255,255) if text==READY_PROMPT else (255,255,255)
            cv2.putText(frame, text, (60, y), cv2.FONT_HERSHEY_SIMPLEX,1.0,color,2)
            y += 70
        cv2.imshow(window_name, frame)
        k = cv2.waitKey(50) & 0xFF
        if k == 27: return False
        if k == 13: return True

# =============================================================================
# 9. AI 策略
# =============================================================================
def assisted_ai_choice(player_gesture: str, ai_win_prob: float) -> str:
    p_ai=ai_win_prob; p_draw=ASSIST_DRAW_FIXED; p_pl=max(0.0,1-p_ai-p_draw)
    r=random.random()
    # mapping same as before
    mapping = {
        'rock':   ('paper','rock','scissors'),
        'paper':  ('scissors','paper','rock'),
        'scissors':('rock','scissors','paper'),
    }
    a,p,s = mapping.get(player_gesture,('rock','paper','scissors'))
    if r<p_ai: return a
    if r<p_ai+p_draw: return p
    return s

def highmode_ai_choice(player_gesture: str, ai_win_prob: float) -> str:
    p_ai=ai_win_prob; rem=1-p_ai; p_draw=rem/2; p_pl=rem/2
    r=random.random()
    mapping = {
        'rock':   ('paper','rock','scissors'),
        'paper':  ('scissors','paper','rock'),
        'scissors':('rock','scissors','paper'),
    }
    a,p,s = mapping.get(player_gesture,('rock','paper','scissors'))
    if r<p_ai: return a
    if r<p_ai+p_draw: return p
    return s

# =============================================================================
# 10. 主循环
# =============================================================================
def main():
    print("==== 模式选择 / Mode Select ====")
    print("1) 本地 vs AI")
    print("2) 联机 (Host)")
    print("3) 联机 (Client)")
    mode=input(">>> ").strip()
    remote_mode=False; net_sock=None; is_host=False
    if mode=='2': net_sock=setup_host(); is_host=net_sock is not None; remote_mode=is_host
    elif mode=='3': net_sock=setup_client(); is_host=False; remote_mode=net_sock is not None
    else: print("进入单人模式")
    if mode in ('2','3') and not remote_mode: print("联机失败, 本地模式继续")
    init_csv()
    cap=cv2.VideoCapture(0)
    if not cap.isOpened(): print("无法打开摄像头"); return
    wins=losses=draws=0; score=0; round_id=0
    assist_active=False; assist_prob=ASSIST_AI_WIN_PROB_INITIAL
    high_active=False; high_prob=HIGH_AI_WIN_PROB_INITIAL
    mp_hands=mp.solutions.hands
    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5) as hands:
        # 联机握手
        if remote_mode:
            if not handshake(net_sock, is_host): remote_mode=False
        while True:
            round_id+=1
            # 模式切换
            if score>=HIGH_ENTER_SCORE and not high_active:
                high_active=True; high_prob=HIGH_AI_WIN_PROB_INITIAL; assist_active=False
            if high_active and score<HIGH_ENTER_SCORE: high_active=False
            if score<ASSIST_ENTER_SCORE and not assist_active and not high_active:
                assist_active=True; assist_prob=ASSIST_AI_WIN_PROB_INITIAL
            if assist_active and score>=ASSIST_EXIT_SCORE: assist_active=False
            # 准备弹窗
            cont = show_next_round_popup(
                WINDOW_NAME, round_id, 
                (None if round_id==1 else last_result),
                (None if round_id==1 else last_best_total),
                remote_mode, net_sock, is_host)
            if not cont: break
            # 准备阶段
            prep_start=time.time()
            while time.time()-prep_start<PREP_SECONDS:
                ret,frame=cap.read(); frame=cv2.flip(frame,1)
                remain=PREP_SECONDS-int(time.time()-prep_start)
                cv2.putText(frame,f"Round {round_id} Ready:{remain}",(20,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),3)
                status=f"Score:{score} W:{wins} L:{losses} D:{draws}"
                if high_active: status+=f" | High p={high_prob:.2f}"
                elif assist_active: status+=f" | Assist p={assist_prob:.2f}"
                cv2.putText(frame,status,(10,470),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
                cv2.imshow(WINDOW_NAME,frame)
                if cv2.waitKey(1)&0xFF==27: return
            # 采集阶段
            capture_start=time.time(); player_gesture="None"; gesture_captured=False
            total_window=deque(maxlen=SMOOTH_WINDOW)
            best_total=best_shape=best_open=None
            while time.time()-capture_start<CAPTURE_SECONDS:
                ret,frame=cap.read(); frame=cv2.flip(frame,1)
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                res=hands.process(rgb)
                if res.multi_hand_landmarks:
                    lm_list=res.multi_hand_landmarks[0].landmark
                    cur=classify_rps(lm_list)
                    if cur in ("rock","paper","scissors"): player_gesture=cur; gesture_captured=True
                    # 模板采集
                    if cv2.waitKey(1)&0xFF==ord('t') and player_gesture in ("rock","paper","scissors"):
                        base = 'fist' if player_gesture=='rock' else 'open' if player_gesture=='paper' else 'scissors'
                        norm_cap=normalize_landmarks(lm_list)
                        gesture_templates[base]={str(i):norm_cap[i] for i in FINGERTIPS}
                        save_templates(gesture_templates)
                        print(f"[模板更新] {base}")
                    # 计算完成度
                    norm=normalize_landmarks(lm_list)
                    if player_gesture=='rock': base='fist'
                    elif player_gesture=='paper': base='open'
                    else: base='scissors'
                    if has_template(base): devs=compute_devs(norm,gesture_templates[base]); shape=shape_score(devs,base)
                    else: devs={}; shape=0.0
                    open_pct=openness_ratio(lm_list)*100
                    if player_gesture=='rock': total_raw=total_fist(shape,open_pct)
                    elif player_gesture=='paper': total_raw=total_open(shape,open_pct)
                    else:
                        open_score=100-abs(open_pct-50.0)
                        total_raw=0.5*shape+0.5*open_score
                    total_window.append(total_raw)
                    total_disp=sum(total_window)/len(total_window)
                    if best_total is None or total_disp>best_total:
                        best_total, best_shape, best_open = total_disp, shape, open_pct
                    y0=180
                    for tip,dval in devs.items():
                        color=(0,255,0) if dval<DEVIATION_GREEN_THRESHOLD else (0,0,255)
                        cv2.putText(frame,f"{tip}:{dval:.3f}",(20,y0),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1)
                        y0+=18
                    if player_gesture=='scissors':
                        cv2.putText(frame,f"Shape:{shape:.1f}% Open:{open_pct:.1f}%",(20,140),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,255),2)
                        cv2.putText(frame,f"ScissScore:{total_disp:.1f}% (Best:{best_total:.1f}%)",(20,115),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
                    else:
                        cv2.putText(frame,f"Shape:{shape:.1f}% Open:{open_pct:.1f}% Clo:{100-open_pct:.1f}%",(20,140),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,255),2)
                        cv2.putText(frame,f"Total:{total_disp:.1f}% (Best:{best_total:.1f}%)",(20,115),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
                    mp.solutions.drawing_utils.draw_landmarks(frame, res.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
                else:
                    cv2.putText(frame,"Completion: N/A",(20,140),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),2)
                remain=CAPTURE_SECONDS-int(time.time()-capture_start)
                cv2.putText(frame,f"Show ({remain})",(20,60),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
                cv2.putText(frame,f"Gesture:{player_gesture}",(20,35),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
                footer="[t]模板  ESC退出"
                if high_active: footer+=f" | High p={high_prob:.2f}"
                elif assist_active: footer+=f" | Assist p={assist_prob:.2f}"
                cv2.putText(frame,footer,(240,470),cv2.FONT_HERSHEY_SIMPLEX,0.5,(180,180,180),1)
                cv2.imshow(WINDOW_NAME,frame)
                if cv2.waitKey(1)&0xFF==27: return
            # 对手手势
            if remote_mode and net_sock:
                if is_host:
                    opp_gesture = highmode_ai_choice(player_gesture, high_prob) if False else None
                    opp = None
                opp_gesture = None
                try: opp_gesture = highmode_ai_choice(player_gesture, high_prob)
                except: opp_gesture = random.choice(["rock","paper","scissors"] )
            else:
                if not gesture_captured: player_gesture="None"; opp_gesture=random.choice(["rock","paper","scissors"])
                else:
                    if high_active: opp_gesture=highmode_ai_choice(player_gesture, high_prob)
                    elif assist_active: opp_gesture=assisted_ai_choice(player_gesture, assist_prob)
                    else: opp_gesture=random.choice(["rock","paper","scissors"])
            # 判定计分
            result = judge(player_gesture, opp_gesture) if player_gesture in ("rock","paper","scissors") else "No gesture!"
            if result=="You Win!": score+=3; wins+=1; high_prob=min(HIGH_AI_WIN_MAX, high_prob+HIGH_AI_WIN_INC)
            elif result=="You Lose!": score-=1; losses+=1; assist_prob=max(ASSIST_AI_WIN_PROB_MIN, assist_prob-ASSIST_AI_WIN_DEC); high_prob=max(HIGH_AI_WIN_MIN, high_prob-HIGH_AI_WIN_DEC)
            else: draws+=1
            # 结果显示
            ret,frame=cap.read(); frame=cv2.flip(frame,1)
            cv2.putText(frame,f"You: {player_gesture}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            who="Opponent" if remote_mode else "AI"
            cv2.putText(frame,f"{who}: {opp_gesture}",(10,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(frame,result,(10,125),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),3)
            if player_gesture in ("rock","paper","scissors"):
                cv2.putText(frame,f"Best:{best_total:.1f}% Shape:{best_shape:.1f}% Open:{best_open:.1f}%",(10,170),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            status=f"Score:{score} W:{wins} L:{losses} D:{draws}"
            if high_active: status+=f" | High p={high_prob:.2f}"
            elif assist_active: status+=f" | Assist p={assist_prob:.2f}"
            cv2.putText(frame,status,(10,470),cv2.FONT_HERSHEY_SIMPLEX,0.65,(255,255,255),2)
            cv2.imshow(WINDOW_NAME,frame)
            ts=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_csv(ts,round_id,player_gesture,opp_gesture,result,best_total,best_shape,best_open,score,wins,losses,draws,high_active,high_prob,assist_active,assist_prob)
            time.sleep(3)
            last_result=result; last_best_total=best_total
            # 等待下一回合按键在弹窗中处理
        cap.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    try: main()
    except KeyboardInterrupt: print("Interrupted"); sys.exit(0)
