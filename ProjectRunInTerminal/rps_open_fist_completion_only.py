#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RPS (Rock / Paper / Scissors) + Completion (Only Open(Paper) & Fist(Rock))
=======================================================================
需求实现:
  * 游戏开始/出拳阶段: 识别玩家手势 只考虑: rock(拳) / paper(掌) / scissors(剪刀)
  * 仅对 rock 与 paper 计算完成度(Completion)。
      - rock 视为 fist:  Total = 0.6*Shape + 0.4*Closure(=100-Open%)
      - paper 视为 open: Total = 0.7*Shape + 0.3*Open%
  * scissors 只分类识别, 不计算完成度 (显示 N/A)。
  * Completion 组成: Shape% (与模板指尖偏差指数衰减) + Open 或 Closure; 未采集模板则 Shape=0
  * 模板采集按 't' (对当前识别手势，如 rock 采集为 fist 模板, paper 采集为 open 模板; scissors 忽略)
  * 每回合写入 CSV: 时间, 回合号, PlayerGesture, OpponentGesture, Result, Total(if any), Shape, Open, Score, Wins, Losses, Draws
    - 若手势=scissors 则 Total/Shape/Open 为空字符串 (""), 方便后续分析区分
  * 平滑窗口(默认5) 仅用于显示 Total (rock/paper)
  * 其余 RPS 逻辑(胜负判定, 分数)保持简化; 去除远程与作弊代码 (可按需要再加)

依赖:
  pip install mediapipe opencv-python numpy

第一次使用:
  1. 做一个最大张开手势 (paper) 按 't' 采模板
  2. 做一个标准握拳手势 (rock) 按 't' 采模板
  然后进入正常游戏; 若未采集, Shape=0, Total 仅靠 Open/Closure 权重部分.

键位:
  ESC    退出
  ENTER  进入下一回合(结果界面)
  t      采集模板 (仅对 rock / paper 有效)
"""
import cv2, mediapipe as mp, random, time, os, csv, json, math, datetime
from collections import deque

# --------------------------- 配置常量 ---------------------------
TEMPLATE_FILE = "gesture_templates.json"  # 保存 open/fist 模板 (scissors 不计)
CSV_FILE      = "rps_open_fist_completion.csv"
SMOOTH_WINDOW = 5
CAPTURE_SECONDS = 5     # 出拳采集阶段时长
PREP_SECONDS    = 2     # 回合准备倒计时

# 完成度权重 (paper=open / rock=fist)
SHAPE_WEIGHT_OPEN   = 0.7
OPEN_WEIGHT_OPEN    = 0.3
SHAPE_WEIGHT_FIST   = 0.6
CLOSURE_WEIGHT_FIST = 0.4
DEVIATION_GREEN_THRESHOLD = 0.05
FINGERTIPS  = [4,8,12,16,20]
PALM_POINTS = [0,5,9,13,17]

# --------------------------- 模板管理 ---------------------------
def load_templates():
    if not os.path.exists(TEMPLATE_FILE):
        data = {"open":{}, "fist":{}}
        with open(TEMPLATE_FILE,'w') as f: json.dump(data,f,indent=2)
        return data
    with open(TEMPLATE_FILE,'r') as f: data=json.load(f)
    changed=False
    for k in ("open","fist"):
        if k not in data: data[k]={}; changed=True
    if changed:
        with open(TEMPLATE_FILE,'w') as f: json.dump(data,f,indent=2)
    return data

def save_templates(tpl):
    with open(TEMPLATE_FILE,'w') as f: json.dump(tpl,f,indent=2)

gesture_templates = load_templates()

def has_template(name):
    return name in gesture_templates and len(gesture_templates[name])>0

# --------------------------- 几何与评分 ---------------------------
def palm_center_and_width(lm_list):
    cx = sum(lm_list[i].x for i in PALM_POINTS)/len(PALM_POINTS)
    cy = sum(lm_list[i].y for i in PALM_POINTS)/len(PALM_POINTS)
    w = math.dist((lm_list[5].x,lm_list[5].y),(lm_list[17].x,lm_list[17].y)) or 1e-6
    return cx,cy,w

def normalize_landmarks(lm_list):
    cx,cy,w = palm_center_and_width(lm_list)
    return {i:((lm.x-cx)/w,(lm.y-cy)/w) for i,lm in enumerate(lm_list)}

def openness_ratio(lm_list):
    cx,cy,w = palm_center_and_width(lm_list)
    ds=[math.dist((lm_list[t].x,lm_list[t].y),(cx,cy))/w for t in FINGERTIPS]
    avg=sum(ds)/len(ds)
    return min(1.0, avg/0.9)

def compute_devs(norm, template):
    devs={}
    for k,(tx,ty) in template.items():
        idx=int(k)
        if idx in norm:
            x,y=norm[idx]
            devs[idx]=math.hypot(x-tx,y-ty)
    return devs

def finger_weight(idx, base):
    if base=="open":
        return 1.2 if idx==4 else 1.0
    if base=="fist":
        return 1.1 if idx in (8,12) else 1.0
    return 1.0

def shape_score(devs, base):
    if not devs: return 0.0
    sims=[math.exp(-8*d)*finger_weight(i,base) for i,d in devs.items()]
    return sum(sims)/len(sims)*100.0

def total_open(shape_part, open_part):
    return SHAPE_WEIGHT_OPEN*shape_part + OPEN_WEIGHT_OPEN*open_part

def total_fist(shape_part, open_part):
    closure = 100 - open_part
    return SHAPE_WEIGHT_FIST*shape_part + CLOSURE_WEIGHT_FIST*closure

# --------------------------- RPS 分类 (简易) ---------------------------
def classify_rps(lm_list):
    tips=[8,12,16,20]
    fingers=[]
    for tip in tips:
        fingers.append(1 if lm_list[tip].y < lm_list[tip-2].y else 0)
    c=sum(fingers)
    if c==0: return "rock"
    if c==2: return "scissors"
    if c>=4: return "paper"
    return "unknown"

# --------------------------- CSV ---------------------------
def init_csv():
    if not os.path.exists(CSV_FILE):
        with open(CSV_FILE,'w',newline='') as f:
            csv.writer(f).writerow(["Time","Round","PlayerGesture","OpponentGesture","Result","Total","Shape","Open","Score","Wins","Losses","Draws"])

def append_csv(ts,rnd,pg,og,res,total,shape,open_pct,score,w,l,d):
    with open(CSV_FILE,'a',newline='') as f:
        csv.writer(f).writerow([ts,rnd,pg,og,res,
                                ("" if total is None else f"{total:.2f}"),
                                ("" if shape is None else f"{shape:.2f}"),
                                ("" if open_pct is None else f"{open_pct:.2f}"),
                                score,w,l,d])

# --------------------------- 胜负判定 ---------------------------
def judge(player, ai):
    if player==ai: return "Draw"
    if (player=="rock" and ai=="scissors") or (player=="scissors" and ai=="paper") or (player=="paper" and ai=="rock"):
        return "You Win!"
    return "You Lose!"

# --------------------------- 主循环 ---------------------------
def main():
    init_csv()
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 无法打开摄像头"); return

    wins=losses=draws=0
    score=0
    round_id=0

    mp_hands=mp.solutions.hands
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while True:
            round_id+=1
            # ---------- 准备阶段 ----------
            prep_start=time.time()
            while time.time()-prep_start < PREP_SECONDS:
                ret,frame=cap.read();
                if not ret: break
                frame=cv2.flip(frame,1)
                remain=PREP_SECONDS-int(time.time()-prep_start)
                cv2.putText(frame,f"Round {round_id} Ready:{remain}",(20,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),3)
                cv2.putText(frame,f"Score:{score} W:{wins} L:{losses} D:{draws}",(10,470),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                cv2.imshow("RPS Completion (Open/Fist)",frame)
                if cv2.waitKey(1)&0xFF==27:
                    cap.release(); cv2.destroyAllWindows(); return

            # ---------- 出拳采集阶段 ----------
            capture_start=time.time()
            player_gesture="None"; gesture_captured=False
            best_total=None; best_shape=None; best_open=None
            total_window=deque(maxlen=SMOOTH_WINDOW)

            while time.time()-capture_start < CAPTURE_SECONDS:
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

                    # 模板采集 (仅 rock / paper)
                    ktemp=cv2.waitKey(1)&0xFF
                    if ktemp==ord('t') and player_gesture in ("rock","paper"):
                        base = 'fist' if player_gesture=='rock' else 'open'
                        norm_cap = normalize_landmarks(lm_list)
                        gesture_templates[base] = {str(i): norm_cap[i] for i in FINGERTIPS}
                        save_templates(gesture_templates)
                        print(f"[模板更新] {base}")

                    # 仅 rock/paper 计算完成度
                    if player_gesture in ("rock","paper"):
                        base = 'fist' if player_gesture=='rock' else 'open'
                        norm=normalize_landmarks(lm_list)
                        shape=0.0
                        if has_template(base):
                            devs=compute_devs(norm,gesture_templates[base])
                            shape=shape_score(devs,base)
                        else:
                            devs={}
                        open_pct = openness_ratio(lm_list)*100
                        total = total_fist(shape,open_pct) if base=='fist' else total_open(shape,open_pct)
                        total_window.append(total)
                        total_disp=sum(total_window)/len(total_window)
                        if (best_total is None) or (total_disp>best_total):
                            best_total=total_disp; best_shape=shape; best_open=open_pct
                        # 显示偏差
                        y0=180
                        for tip in FINGERTIPS:
                            if has_template(base) and str(tip) in gesture_templates[base]:
                                tv=gesture_templates[base][str(tip)]
                                dv=compute_devs(norm,{str(tip):tv}).get(tip,0)
                                color=(0,255,0) if dv<DEVIATION_GREEN_THRESHOLD else (0,0,255)
                                cv2.putText(frame,f"{tip}:{dv:.3f}",(20,y0),cv2.FONT_HERSHEY_SIMPLEX,0.45,color,1); y0+=18
                        closure=100-open_pct
                        cv2.putText(frame,f"Shape:{shape:.1f}% Open:{open_pct:.1f}% Clo:{closure:.1f}%",(20,140),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,255),2)
                        cv2.putText(frame,f"Total:{total_disp:.1f}% (Best:{best_total:.1f}%)",(20,115),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
                    else:
                        # scissors 或 unknown - 不计算
                        cv2.putText(frame,"Completion: N/A (scissors)",(20,140),cv2.FONT_HERSHEY_SIMPLEX,0.55,(200,200,200),2)

                    mp.solutions.drawing_utils.draw_landmarks(frame,results.multi_hand_landmarks[0],mp_hands.HAND_CONNECTIONS)

                remain=CAPTURE_SECONDS-int(time.time()-capture_start)
                cv2.putText(frame,f"Show ({remain})",(20,60),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)
                cv2.putText(frame,f"Gesture:{player_gesture}",(20,35),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
                cv2.putText(frame,"[t]模板 rock/paper  ESC退出",(300,470),cv2.FONT_HERSHEY_SIMPLEX,0.55,(180,180,180),1)
                cv2.imshow("RPS Completion (Open/Fist)",frame)
                if cv2.waitKey(1)&0xFF==27:
                    cap.release(); cv2.destroyAllWindows(); return

            # ---------- 判定与 AI ----------
            ai_choice=random.choice(["rock","paper","scissors"])
            if gesture_captured and player_gesture in ("rock","paper","scissors"):
                result = judge(player_gesture, ai_choice)
                if result=="You Win!":
                    score+=3; wins+=1
                elif result=="You Lose!":
                    score-=1; losses+=1
                else:
                    draws+=1
            else:
                player_gesture="None"; result="No gesture!"

            # ---------- 结果显示 & CSV ----------
            ret,frame=cap.read();
            if not ret: break
            frame=cv2.flip(frame,1)
            show_total=best_total if player_gesture in ("rock","paper") else None
            show_shape=best_shape if player_gesture in ("rock","paper") else None
            show_open =best_open  if player_gesture in ("rock","paper") else None

            cv2.putText(frame,f"You: {player_gesture}",(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
            cv2.putText(frame,f"AI: {ai_choice}",(10,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            cv2.putText(frame,result,(10,125),cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,255),3)
            if show_total is not None:
                cv2.putText(frame,f"Best Total:{show_total:.1f}% Shape:{show_shape:.1f}% Open:{show_open:.1f}%",(10,170),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
            else:
                cv2.putText(frame,"Best Total: N/A (scissors)",(10,170),cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),2)
            cv2.putText(frame,f"Score:{score} W:{wins} L:{losses} D:{draws}",(10,470),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            cv2.putText(frame,"ENTER下一回合  ESC退出",(10,440),cv2.FONT_HERSHEY_SIMPLEX,0.65,(180,180,180),2)
            cv2.imshow("RPS Completion (Open/Fist)",frame)

            ts=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            append_csv(ts,round_id,player_gesture,ai_choice,result,show_total,show_shape,show_open,score,wins,losses,draws)

            while True:
                k=cv2.waitKey(0)&0xFF
                if k==13: # Enter
                    break
                if k==27:
                    cap.release(); cv2.destroyAllWindows(); return

    cap.release(); cv2.destroyAllWindows()

if __name__=='__main__':
    main()
