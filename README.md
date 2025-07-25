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


## Get Start

# Rock Paper Scissors (Hand Gesture Recognition Game In Terminal)

This is an interactive Rock-Paper-Scissors game using hand gesture recognition powered by OpenCV and MediaPipe. It supports local play, online multiplayer via socket, adaptive AI difficulty, gesture template training, and scoring metrics based on gesture shape and openness.

---

## ✅ Features

- 🤖 Local vs AI mode with adaptive difficulty (Assist / Normal / High)
- 🌐 Multiplayer mode (Host or Client with IP input)
- 🖐 Hand gesture recognition using MediaPipe (OpenCV visualization)
- 🎯 Shape / Open / Total score metrics for gesture quality
- 🧠 't' to train gesture templates during play
- 📈 CSV logs for all game rounds
- 📷 Template training mode for custom rock/paper/scissors

---

## 🚀 Getting Started

### 1. Install Python (Recommended: 3.8 ~ 3.11)
Check with:
```bash
python --version
```

### 2. Create and Activate Virtual Environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies
```bash
pip install opencv-python mediapipe numpy
```

---

## 📂 Required Files

Ensure these files are in the **same directory**:

- `version_8.5.py` (main program)
- `gesture_templates.json` (auto-created for gesture data)
- `rps_open_fist_completion.csv` (auto-created for game logs)

---

## 💻 Running the Program

```bash
python version_8.5.py
```

### You will see a black pop-up window for mode selection:

| Key     | Function                        |
|---------|---------------------------------|
| W / S   | Navigate options                |
| ENTER   | Select                          |
| ESC     | Quit program                    |

---

## 🎮 Modes

| Mode     | Description                                      |
|----------|--------------------------------------------------|
| Local    | Play against AI locally                          |
| Host     | Host an online match (displays your IP address)  |
| Client   | Join a match by entering Host's IP               |
| Template | Enter template collection mode                   |

---

## ✋ Template Mode Instructions

- Show one hand to camera
- Press keys to save gesture templates:
  - `R` → Rock (fist)
  - `P` → Paper (open)
  - `S` → Scissors
- Press `ESC` to exit

Templates are saved to `gesture_templates.json`.

---

## 🌐 Multiplayer Instructions

1. One player selects **Host**, waits for connection (IP shown)
2. Other player selects **Client**, inputs host IP
3. Game syncs using READY / ACK handshake
4. Proceed to play synchronously

Make sure port `65432` is open on both sides and devices are reachable.

---

## 📊 Scoring Metrics

- **Shape %**: similarity to template keypoints
- **Open %**: openness of hand
- **Total %**: combination of both based on gesture type

---

## 🧪 Testing Suggestions

1. Start in `Template` mode and collect gesture templates
2. Try `Local` mode to test recognition
3. Connect two devices via `Host`/`Client` for multiplayer
4. Review `rps_open_fist_completion.csv` for scores and logs

---

## ❓ Troubleshooting

| Issue             | Cause / Fix                                        |
|------------------|----------------------------------------------------|
| Camera not opening | Check permissions / Close other camera apps       |
| Connection failed | Use correct IP / Same network / Port not blocked  |
| No pop-up shown   | Make sure you're in a GUI environment              |

---

## 📜 License

This project is provided strictly for educational and demonstration purposes.
All rights reserved. Modification, redistribution, or commercial use is not permitted without explicit permission from the author.


# ✋ Rock-Paper-Scissors (Streamlit Version)

This is a web-based hand gesture recognition RPS (Rock-Paper-Scissors) game built with **Streamlit + OpenCV + MediaPipe**. It supports:

- Local vs AI gameplay
- Host / Client multiplayer via LAN
- Real-time gesture scoring: Shape, Open, and Total completion
- Gesture template collection
- Progress tracking with charts

---

## ✅ Features

- 🖐 Hand gesture detection using MediaPipe
- 🤖 Adaptive AI difficulty (Assist / High)
- 🌐 Multiplayer mode with socket networking
- 📈 Score chart and daily performance logs
- 📄 CSV logging: gesture, score, mode, and timestamps

---

## 🚀 Getting Started

### 1. Clone the Repository & Navigate
```bash
git clone <your-repo-url>
cd <project-folder>
```

### 2. Create Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate   # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Running the App

```bash
streamlit run Home.py
```

- Navigate to `Game Mode` to play.
- Click `Game Mode` to enter the gesture game.
- Choose **Local / Host / Client**, and click `Connect/Start`.

---

## ✋ Gesture Template

- Templates help score your gestures.
- If `gesture_templates.json` is empty, complete scores may be lower.
- To customize templates, switch to Template Mode (if implemented), or press `t` in OpenCV versions.

---

## 🎮 Modes

| Mode     | Description                                      |
|----------|--------------------------------------------------|
| Local    | Play vs AI on your machine                       |
| Host     | Host a match and wait for Client (show IP)       |
| Client   | Connect to Host's IP to play                     |

---

## 📊 Score Chart

From the homepage:

- View average best-completion score per day.
- Filtered by gesture: rock, paper, scissors.
- Data is read from `rps_open_fist_completion.csv`.

---

## ❗ Important Notes

- Your webcam is required.
- Use in a GUI environment (no headless server).
- CSV and JSON files are auto-created:
  - `gesture_templates.json`
  - `rps_open_fist_completion.csv`

---

## 📜 License

This project is provided for educational and demonstration purposes only.  
All rights reserved. Modification, redistribution, or commercial use is not permitted without explicit permission from the author.