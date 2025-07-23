---

---

# Team Member

**Zhang Haoran**: Design and Psychological consideration, Front-end developing
**LIU Muzhou**: Back-end development, Remote combat, gesture recognition
**DONG Qianbin**: Back-end development, Model training, gesture recognition

	Note:Here's only the rough task breakdown

---

# Project description

**HandRehab-RPS** is a **computer-vision-driven serious game** that turns the classic **Rock–Paper–Scissors (RPS)** hand gesture contest into a structured, data-rich rehabilitation exercise. Using any consumer camera (laptop, tablet or phone), the system recognizes three base gestures—**Rock (握拳)**, **Paper (张掌)** and **Scissors (分指)**—plus optional “hold” and “speed-round” variants. A rules engine transforms these gestures into competitive rounds (vs AI or another patient online), while an analytics layer streams **range-of-motion (ROM)**, **repeat count**, **reaction time** and a grip-strength proxy to a clinician dashboard.

HandRehab-RPS 是一款基于计算机视觉的严肃游戏，将经典的石头剪刀布（RPS）手势比赛转化为结构化、数据丰富的康复训练。通过任何消费级摄像头（笔记本电脑、平板电脑或手机），系统能够识别三种基础手势——石头（握拳）、布（张掌）和剪刀（分指），并支持可选的“保持”和“速度回合”变体。规则引擎将这些手势转化为对抗性回合（与 AI 或在线的另一位患者对战），同时分析层将运动范围（ROM）、重复次数、反应时间和握力代理数据流式传输到临床医生仪表板。


# Background and motivation

Traditional hand rehabilitation is repetitive, tedious, and often difficult to monitor remotely. Patients may lack motivation to complete daily exercises, and clinicians have limited tools for real-time tracking. By introducing an interactive, game-based approach using real-time hand gesture recognition, we aim to improve training adherence, accuracy, and measurable recovery outcomes.

# Target Users

**Primary**: 
	Patients recovering from hand injuries or surgery
	Clinicians(therapists doctors) who supervise rehabilitataion

**Secondary**
	Patients' family who want to be company with and help recovering

# Project Scope and Futures
## Core Features

1. Real-time hand gesture recongnition
2. Single player vs Computer
3. Player VS Player / Remote combat
4. Training metrics tracking : Capture hand gesture frames, Accuracy, Pain score, Reaction time, ROM(Range of motion)
5. Summary and feedback in each round and after training
6. Historical data visualization (ROM trends, pain score history).
7. Doctor view: patient profiles, progress, configurable training goals.

## **Game Mechanism**（detailed in [[Game Mechanic]]）

Our rehabilitation game adopts a **turn-based match system**, where each match consists of multiple short rounds. In every round, the player performs a gesture (rock, paper, or scissors) and receives immediate feedback. Scores are gained or lost depending on the result, and a match is won when a score threshold is reached.

To maintain challenge and reduce frustration, we use a **dynamic difficulty algorithm** instead of a traditional AI. When the player is losing, the system lowers difficulty by randomizing opponent moves. As the player approaches victory, the chance of an intentional loss (via probabilistic adjustment) increases slightly, ensuring balance and preventing predictable wins or long losing streaks.

Players can select between two training modes:

- **Time-based**: The session runs for a fixed time period.
    
- **Game-based**: The session continues until a fixed number of wins is achieved. This mode emphasizes gesture accuracy.
    

Additionally, the PVP mode (planned) uses skill-based matchmaking to pair patients of similar ability, enhancing fairness and social engagement.

This mechanic design integrates **psychological motivation principles**—such as flow, feedback loops, autonomy, and reward anticipation—while also aligning with key **HCI design goals** like responsiveness, adaptiveness, and simplicity.

# Technical Implementation & Challenges

1. **Hand Gesture Recognition**(completed)
    
    - Uses **MediaPipe** to extract 21 hand landmarks from a webcam feed in real time.
        
    - Gesture classification is based on **geometric rules**, such as relative distances between fingertips and palm landmarks.
        
    - No AI or learning models used; recognition logic is fixed, explainable, and easy to calibrate per user.
        
2. **Game Logic & Difficulty Adjustment**(Completed)
    
    - The game uses a **rule-based algorithm** to control round outcomes:
        
        - When the player performs poorly → opponent behavior is randomized (easier to win).
            
        - When the player approaches match victory → a small, fixed-probability condition introduces challenge by enforcing specific losses.

3. **Frontend Application**(developing)
    
    - Built with **Streamlit**, providing a clean interface for:
        
        - Real-time camera display
            
        - Scoreboard and progress bars
            
        - Mode selection (time-based or match-based)
            
        - Session summaries and export options
            
    - Relies on `st.session_state` for managing gameplay state across views.
        
4. **Data Logging & Visualization**(developing)
    
    - Performance data per session (e.g. gesture timing, match outcome, estimated hand motion range) is saved in CSV or SQLite format.
        
    - Visual feedback includes progress charts, session history, and basic statistical summaries.
        
    - Option to export session reports as PDFs for clinicians.
        
5. **Remote combat and teleport**(completed)
