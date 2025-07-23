import time
import cv2
import streamlit as st
from rps_engine import RPSBackend, save_templates

st.set_page_config(layout="wide", page_title="Rock–Paper–Scissors")

# ── 页面逻辑判断：如有重定向标记则跳转 ──
if "goto_gamemode" in st.session_state and st.session_state.goto_gamemode:
    st.session_state.goto_gamemode = False
    st.rerun()

# ── 1. Session State 初始化 ───────────────────────────
if "connected"      not in st.session_state: st.session_state.connected      = False
if "mode"           not in st.session_state: st.session_state.mode           = "Local vs AI"
if "host_ip"        not in st.session_state: st.session_state.host_ip        = ""
if "backend"        not in st.session_state: st.session_state.backend        = None
if "round_started"  not in st.session_state: st.session_state.round_started  = False
if "camera"         not in st.session_state: st.session_state.camera         = None  # 新增摄像头状态

# ── 2. 回调函数 ───────────────────────────────────────
def connect_cb():
    st.session_state.connected = True

def start_round_cb():
    st.session_state.round_started = True

def next_round_cb():
    st.session_state.round_started = False
    st.session_state.backend.start_round()

def return_to_home():
    # ✅ 释放摄像头
    if st.session_state.camera:
        st.session_state.camera.release()
        st.session_state.camera = None
    st.session_state.connected = False
    st.session_state.round_started = False
    st.session_state.backend = None
    st.switch_page("Home.py")

# ── 3. 未连接时：显示开始界面 ───────────────────────────
if not st.session_state.connected:
    st.title("🎮 Game Mode")
    st.session_state.mode = st.radio(
        "Choose the Mode",
        ["Local vs AI", "Host (Net)", "Client (Net)"],
        index=["Local vs AI", "Host (Net)", "Client (Net)"].index(st.session_state.mode)
    )
    if st.session_state.mode.startswith("Client"):
        st.session_state.host_ip = st.text_input("输入 Host IP:", st.session_state.host_ip)
    st.button("Connect/Start", on_click=connect_cb)

    st.markdown("""
    <div style="border:2px solid white; padding: 1rem; border-radius: 8px; background-color: #e6f2ff; margin-top: 1.5rem;">
      <h4>📝 Hint:</h4>
      <ul>
        <li><b>👊Rock</b> BEATS <b>✌️Scissors</b></li>
        <li><b>✌️Scissors</b> BEATS <b>🖐️Paper</b></li>
        <li><b>🖐️Paper</b> BEATS <b>👊Rock</b></li>
        <li><b>If the guestures are same: Draw</b></li>
        <li><b>Win: Score + 3</b></li>
        <li><b>Loose: Score - 1</b></li>
        <li><b>Draw: Score + 0</b></li>
      </ul>
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ── 4. 后端初始化 ───────────────────────────────────────
if st.session_state.backend is None:
    m  = st.session_state.mode
    mi = "host" if m.startswith("Host") else "client" if m.startswith("Client") else "local"
    st.session_state.backend = RPSBackend(mode=mi, host_ip=st.session_state.host_ip)

# ✅ 初始化摄像头
if st.session_state.camera is None:
    st.session_state.camera = cv2.VideoCapture(0)

backend = st.session_state.backend
cap = st.session_state.camera  # ✅ 用摄像头实例

# ── 5. 页面布局 ───────────────────────────────────────
col_video, _, col_info = st.columns([4, 0.5, 2], gap="large")

# —— 右侧信息区
with col_info:
    score_slot = st.empty()
    def render_score():
        score_slot.markdown(f"""
        <div style="background:#f0f0f0;padding:1rem;border-radius:0.5rem;">
          <h4>🏆 Score：{backend.score}</h4>
          <h5>🔄 Round：{backend.round_id}</h5>
          <h5>✅ Win：{backend.wins} 次</h5>
          <h5>❌ Loose：{backend.losses} 次</h5>
          <h5>🤝 Draw：{backend.draws} 次</h5>
        </div>
        """, unsafe_allow_html=True)
    render_score()

    st.markdown("""
    <div style="background:#f0f0f0;padding:1rem;border-radius:0.5rem;">
      <h4>📋 Tasks Remaining</h4>
      <ol>
        <li>Task 1</li><li>Task 2</li><li>Task 3</li><li>Task 4</li>
      </ol>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Back To Home", key="return_to_home_sidebar"):
        return_to_home()

# —— 左侧主区域
with col_video:
    subcol1, subcol2 = st.columns([4, 1])
    subcol1.subheader(f"Round {backend.round_id or 1}")
    gesture_slot = subcol1.empty()
    gesture_slot.markdown("🔍 completion: 0.00")
    timer_slot = subcol2.empty()

    btn_col1, btn_col2 = st.columns([3, 1], gap="medium")
    with btn_col1:
        if st.button("Back To GameMode", key="return_to_gamemode_banner"):
            if st.session_state.camera:
                st.session_state.camera.release()
                st.session_state.camera = None
            st.session_state.connected = False
            st.session_state.round_started = False
            st.session_state.backend = None
            st.rerun()
    with btn_col2:
        st.button("Next Round", on_click=next_round_cb)

    if not st.session_state.round_started:
        st.button("▶️ Start", on_click=start_round_cb)
    else:
        frame_slot = st.empty()
        max_total = 0.0
        best_gesture = "rock"
        num_frames = 50
        interval = 0.1

        for i in range(num_frames):
            data = backend.process_frame()
            img = cv2.cvtColor(data["frame"], cv2.COLOR_BGR2RGB)
            frame_slot.image(img, use_container_width=True)

            current = data.get("total") or 0.0
            gesture_slot.markdown(f"🔍 completion: {current:.2f}")

            remaining = (num_frames - i) * interval
            timer_slot.markdown(f"⏱️ Time remaining: **{remaining:.1f}s**")

            if current > max_total:
                max_total = current
                best_gesture = data.get("gesture") or best_gesture

            time.sleep(interval)

        player_gesture = best_gesture
        result, ai_g = backend.end_round(player_gesture, best_total=max_total)

        st.success(f"Result：{result} （opponent：{ai_g}）")

        comment = "💙 Excellent" if max_total >= 28 else "👍 Good job" if max_total >= 25 else "💡 Come on, you can do better"
        st.info(f"💡 Highest completion：{max_total:.2f} Best gesture **{best_gesture}**\n\n{comment}")

        render_score()
