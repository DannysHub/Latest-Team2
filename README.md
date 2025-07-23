# team2

## Project Description
**HandRehab-RPS** is a **computer-vision-driven serious game** that turns the classic **Rock–Paper–Scissors (RPS)** hand gesture contest into a structured, data-rich rehabilitation exercise.  Using any consumer camera (laptop, tablet or phone), the system recognizes three base gestures—**Rock (握拳)**, **Paper (张掌)** and **Scissors (分指)**—plus optional “hold” and “speed-round” variants.  A rules engine transforms these gestures into competitive rounds (vs AI or another patient online), while an analytics layer streams **range-of-motion (ROM)**, **repeat count**, **reaction time** and a grip-strength proxy to a clinician dashboard.

HandRehab-RPS 是一款基于计算机视觉的严肃游戏，将经典的石头剪刀布（RPS）手势比赛转化为结构化、数据丰富的康复训练。通过任何消费级摄像头（笔记本电脑、平板电脑或手机），系统能够识别三种基础手势——石头（握拳）、布（张掌）和剪刀（分指），并支持可选的“保持”和“速度回合”变体。规则引擎将这些手势转化为对抗性回合（与 AI 或在线的另一位患者对战），同时分析层将运动范围（ROM）、重复次数、反应时间和握力代理数据流式传输到临床医生仪表板。

## 🧭 Why This Project matters? | 我们为什么要做这个项目？

传统的手部康复训练存在多个关键问题，而这些问题至今未被很好地解决：

1. **训练枯燥，患者坚持难**  
   Repetitive hand exercises like finger flexion/extension are boring and painful, leading to poor adherence.  
   手部康复训练高度重复，动作单一、痛苦，患者缺乏动力和持续性。

2. **训练效果看不到，成就感低**  
   Patients often can't perceive short-term progress, especially in the early to mid recovery phases.  
   在恢复早期，动作幅度或功能提升不明显，患者缺乏反馈和成就感。

3. **医生看不到患者在家练了什么、练得怎么样**  
   There's little visibility into what patients actually do at home—how often, how well, and whether safely.  
   医生和治疗师无法远程跟踪动作质量、练习频率或ROM变化，干预时机难以把握。

4. **数据缺乏结构化，不可用于监控或决策**  
   Rehab data is rarely recorded in a structured, actionable form.  
   缺乏可被量化、可被分析的数据，难以支持疗效判断或计划调整。

## Core Features
1. 手势识别模块 Gesture Recognition
2. AI 对战模块 Game Engine
3. 远程对战模块 Live multiplayer 
4. 数据记录与分析模块 Data Logging & Metrics 
5. 数据同步以及临床记录 Sync & Clinician Dashboard
