
import streamlit as st
import pandas as pd

# 页面设置
st.set_page_config(layout="wide")
st.title("Hi, Brandon")
st.caption("How are you today")

# 两列布局：左侧 daily tasks + 模式选择，右侧图表
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📝 Daily tasks")
    st.write("Your list of Tasks")
    st.markdown("""
                - Task 1:
                - Task 2:
                - Task 3:""")

    st.markdown("---")
    st.markdown("### 🎮 Mode Choosing")
    if st.button("Game Mode"):
        st.switch_page("pages/Game.py")
    if st.button("Time Mode"):
        st.info("Time mode 尚未实现")

with col2:
    try:
        df = pd.read_csv("rps_open_fist_completion.csv")

        # 只保留需要的列
        df = df[["Time", "Gesture", "BestCompletion"]]
        df["Time"] = pd.to_datetime(df["Time"], format="%Y-%m-%dT%H:%M:%S")

        st.markdown("### 📉 Your progress (Chart)")
        c1, c2, c3 = st.columns(3)

        for i, (gesture, col) in enumerate(zip(["rock", "paper", "scissors"], [c1, c2, c3])):
            df_g = df[df["Gesture"] == gesture]
            if not df_g.empty:
                # 按日期分组计算每日平均完成度
                df_g["Date"] = df_g["Time"].dt.date
                daily_avg = df_g.groupby("Date")["BestCompletion"].mean()

                with col:
                    st.markdown(f"**{gesture.title()}**", unsafe_allow_html=True)
                    st.line_chart(daily_avg)

    except Exception as e:
        st.warning(f"读取数据失败: {e}")
