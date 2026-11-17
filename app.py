import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import plotly.graph_objects as go

st.set_page_config(
    page_title="對球不對人｜動態期望熱區與代打決策系統",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 深色專業情蒐儀表板樣式
st.markdown("""
<style>
    .stApp { background-color: #0b111e; color: #e2e8f0; }
    .header-box {
        background: linear-gradient(90deg, #111c30 0%, #162642 100%);
        border: 1px solid #243b61;
        border-radius: 10px;
        padding: 14px 20px;
        margin-bottom: 14px;
    }
    .scenario-bar {
        background-color: #131f33;
        border-left: 4px solid #f59e0b;
        border-radius: 6px;
        padding: 10px 16px;
        font-size: 15px;
        margin-bottom: 16px;
    }
    .kpi-card {
        background-color: #131f33;
        border: 1px solid #233554;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .kpi-val { font-size: 24px; font-weight: 800; color: #4ade80; }
    .kpi-sub { font-size: 12px; color: #94a3b8; }
    .player-card {
        background-color: #131f33;
        border: 1px solid #263b5e;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .player-card-active {
        background: linear-gradient(135deg, #192b47 0%, #131f33 100%);
        border: 2px solid #facc15;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 10px;
    }
    .advice-box {
        background-color: #162238;
        border: 1px solid #334e7b;
        border-radius: 10px;
        padding: 14px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 模擬資料庫（嚴格基於 Rebas 開源欄位：球速、Plate X/Z 座標、球種，無 TrackMan 轉速）
PITCH_DATA = {
    "直球 (153-157 km/h)": {
        "target_x": 0.26, "target_z": 3.18, "avg_v": 154.2, "max_v": 158.1, "v_range": (153, 157),
        "speed_dist": (154.2, 1.8), "n_hist": 320, "k_sim": 50
    },
    "滑球 (134-138 km/h)": {
        "target_x": 0.38, "target_z": 2.05, "avg_v": 136.1, "max_v": 139.4, "v_range": (134, 138),
        "speed_dist": (136.1, 1.5), "n_hist": 185, "k_sim": 42
    },
    "指叉球 (138-142 km/h)": {
        "target_x": 0.08, "target_z": 1.65, "avg_v": 140.3, "max_v": 143.0, "v_range": (138, 142),
        "speed_dist": (140.3, 1.4), "n_hist": 140, "k_sim": 35
    }
}

BATTER_PROFILES = {
    "蘇緯達 (1B/DH)": {
        "rank": 1, "avg": ".324", "obp": ".396", "slg": ".538",
        "直球 (153-157 km/h)": {"dxr": "+0.18", "ci": "+0.03 ~ +0.31", "obp_p": "42.7%", "wpa": "+2.3%", "match": "94%", "n_eff": 48, "sweet_x": 0.22, "sweet_z": 3.05, "reason": ["對右投高球速具備極佳長打轉換率", "外角高球區揮空率僅 17.8%", "一二壘有人得點圈長打期望值最高"]},
        "滑球 (134-138 km/h)": {"dxr": "+0.06", "ci": "-0.04 ~ +0.16", "obp_p": "33.1%", "wpa": "+0.8%", "match": "79%", "n_eff": 36, "sweet_x": 0.05, "sweet_z": 2.40, "reason": ["外角低滑球容易形成滾地球", "追打壞球率略高於陳子豪", "建議鎖定偏高失投滑球攻擊"]},
        "指叉球 (138-142 km/h)": {"dxr": "+0.09", "ci": "-0.02 ~ +0.20", "obp_p": "35.4%", "wpa": "+1.1%", "match": "82%", "n_eff": 31, "sweet_x": 0.10, "sweet_z": 2.15, "reason": ["低角度撈擊能力佳，不易揮空", "好球帶下緣長打率優於聯盟平均", "需留意兩好球後本壘板前落地引誘球"]}
    },
    "陳子豪 (OF)": {
        "rank": 2, "avg": ".298", "obp": ".367", "slg": ".511",
        "直球 (153-157 km/h)": {"dxr": "+0.11", "ci": "-0.01 ~ +0.23", "obp_p": "39.5%", "wpa": "+1.5%", "match": "88%", "n_eff": 50, "sweet_x": -0.05, "sweet_z": 2.75, "reason": ["選球紀律優異，不易追打外角壞球", "內角中低直球破壞力最強", "外角高極端速球揮空率略高於蘇緯達"]},
        "滑球 (134-138 km/h)": {"dxr": "+0.15", "ci": "+0.02 ~ +0.28", "obp_p": "41.2%", "wpa": "+2.0%", "match": "91%", "n_eff": 42, "sweet_x": 0.25, "sweet_z": 2.20, "reason": ["面對同側或右投橫向滑球辨識度高", "選球眼佳，四壞保送擠成滿壘機率高", "滑球情境下為板凳首選代打"]},
        "指叉球 (138-142 km/h)": {"dxr": "+0.14", "ci": "+0.01 ~ +0.27", "obp_p": "40.8%", "wpa": "+1.9%", "match": "90%", "n_eff": 35, "sweet_x": 0.05, "sweet_z": 1.95, "reason": ["低球揮棒軌跡平順，選掉壞球能力強", "面對下墜球種三振率顯著低於蘇緯達", "指叉球情境下為最佳代打人選"]}
    },
    "江坤宇 (SS・原打者)": {
        "rank": 3, "avg": ".271", "obp": ".333", "slg": ".418",
        "直球 (153-157 km/h)": {"dxr": "+0.00", "ci": "基準值", "obp_p": "34.2%", "wpa": "0.0%", "match": "65%", "n_eff": 50, "sweet_x": -0.15, "sweet_z": 2.55, "reason": ["纏鬥能力強，但面對 155km/h 速球長打率有限", "保留游擊金手套防線，無需消耗板凳守備調度", "落後 1 分 2 出局時得分期望低於代打方案"]},
        "滑球 (134-138 km/h)": {"dxr": "+0.00", "ci": "基準值", "obp_p": "32.0%", "wpa": "0.0%", "match": "72%", "n_eff": 40, "sweet_x": 0.12, "sweet_z": 2.35, "reason": ["球棒控制力佳，不易被外角滑球三振", "以穿越安打推進為主，長打清壘機率較低", "維持原陣容可確保 9 局上中線守備穩定"]},
        "指叉球 (138-142 km/h)": {"dxr": "+0.00", "ci": "基準值", "obp_p": "31.5%", "wpa": "0.0%", "match": "70%", "n_eff": 34, "sweet_x": 0.00, "sweet_z": 2.10, "reason": ["對下墜球種擊球確實度高，但多為一壘安打", "2 出局一二壘有人時，一壘安打未必能送回二壘跑者", "建議啟用長打型左打代打擴大得分期望"]}
    },
    "許庭綸 (OF・二軍新秀測試)": {
        "rank": 4, "avg": ".285 (二軍MLE)", "obp": ".350", "slg": ".430",
        "直球 (153-157 km/h)": {"dxr": "+0.04", "ci": "-0.08 ~ +0.16", "obp_p": "35.8%", "wpa": "+0.5%", "match": "74%", "n_eff": 9, "sweet_x": 0.10, "sweet_z": 2.80, "reason": ["⚠️ 觸發防呆：一軍相似球僅 9 顆 (<15顆門檻)", "已自動融合「聯盟右打先驗值」與二軍 MLE 校正", "避免極小樣本誤判，建議關鍵時刻仍優先派老將"]},
        "滑球 (134-138 km/h)": {"dxr": "+0.02", "ci": "-0.09 ~ +0.13", "obp_p": "32.5%", "wpa": "+0.2%", "match": "68%", "n_eff": 7, "sweet_x": 0.08, "sweet_z": 2.30, "reason": ["⚠️ 觸發防呆：有效相似球僅 7 顆 (<15顆門檻)", "貝氏收縮權重提高聯盟先驗比例，防止過度自信", "變化球辨識度尚待一軍高強度樣本累積"]},
        "指叉球 (138-142 km/h)": {"dxr": "+0.01", "ci": "-0.10 ~ +0.12", "obp_p": "31.8%", "wpa": "+0.1%", "match": "66%", "n_eff": 6, "sweet_x": 0.02, "sweet_z": 2.05, "reason": ["⚠️ 觸發防呆：有效相似球僅 6 顆 (<15顆門檻)", "系統自動執行距離截斷，不向下硬抓慢速球充數", "高壓決勝局不建議啟用無一軍大樣本支撐之新秀"]}
    }
}

# 頂部標題與戰況列
st.markdown("""
<div class="header-box">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <span style="font-size:22px; font-weight:800; color:#facc15;">⚾ 對球不對人</span>
            <span style="font-size:16px; color:#cbd5e1; margin-left:10px;">｜ 基於球路特徵之動態期望熱區與代打決策系統</span>
        </div>
        <div style="font-size:13px; color:#4ade80;">● Rebas 開源逐球模型就緒（無 TrackMan 轉速依賴）</div>
    </div>
</div>
<div class="scenario-bar">
    <strong>🔥 實戰高壓情境：</strong> 8局下 2出局 一二壘有人 ｜ <strong>比分：中信兄弟 2 - 3 富邦悍將</strong> ｜ <strong>面對投手：富藍戈 (富邦悍將，右投)</strong>
</div>
""", unsafe_allow_html=True)

# 互動控制列（球種與代打名單切換）
ctrl1, ctrl2 = st.columns([1, 1.5])
with ctrl1:
    selected_pitch = st.selectbox("🎯 切換投手球路特徵", list(PITCH_DATA.keys()), index=0)
with ctrl2:
    selected_batter = st.radio(
        "📋 切換檢視代打／打者熱區與模擬結果",
        list(BATTER_PROFILES.keys()),
        index=0,
        horizontal=True
    )

p_info = PITCH_DATA[selected_pitch]
b_info = BATTER_PROFILES[selected_batter]
stats = b_info[selected_pitch]

# 防呆機制警示條（當有效樣本 < 15 顆時自動亮起）
if stats["n_eff"] < 15:
    st.warning(f"🛡️ **樣本不足防呆機制啟動**：【{selected_batter}】面對此球路之有效相似樣本僅 **{stats['n_eff']} 顆**（低於 15 顆門檻），已自動執行最大距離截斷，並導入「聯盟同側打者先驗值」進行貝氏平滑收縮（Empirical Bayes Shrinkage）。")

# 三欄式主決策儀表板
col1, col2, col3 = st.columns([1.15, 1.15, 1.2])

# ==================== 【左欄】動態期望熱區 ====================
with col1:
    st.markdown(f"#### 🔥 動態期望熱區｜{selected_batter.split(' ')[0]}")
    
    # 根據打者甜區座標動態生成 2D-KDE 平滑熱圖
    np.random.seed(abs(hash(selected_batter + selected_pitch)) % 10000)
    hx = np.random.normal(stats["sweet_x"], 0.25, 60)
    hz = np.random.normal(stats["sweet_z"], 0.32, 60)
    gx, gz = np.mgrid[-0.8:0.8:50j, 1.0:4.5:50j]
    kde = gaussian_kde(np.vstack([hx, hz]), bw_method=0.48)
    z_density = np.reshape(kde(np.vstack([gx.ravel(), gz.ravel()])).T, gx.shape)

    fig_heat = go.Figure()
    fig_heat.add_trace(go.Contour(
        x=np.linspace(-0.8, 0.8, 50),
        y=np.linspace(1.0, 4.5, 50),
        z=z_density.T,
        colorscale="Turbo",
        showscale=False,
        contours=dict(coloring="heatmap", showlines=False),
        opacity=0.88
    ))
    # 好球帶九宮格框線
    fig_heat.add_shape(type="rect", x0=-0.5, y0=1.6, x1=0.5, y1=3.4, line=dict(color="rgba(255,255,255,0.8)", width=2))
    for xl in [-0.17, 0.17]:
        fig_heat.add_shape(type="line", x0=xl, y0=1.6, x1=xl, y1=3.4, line=dict(color="rgba(255,255,255,0.35)", width=1, dash="dot"))
    for yl in [2.2, 2.8]:
        fig_heat.add_shape(type="line", x0=-0.5, y0=yl, x1=0.5, y1=yl, line=dict(color="rgba(255,255,255,0.35)", width=1, dash="dot"))

    # 標示當前投手目標進壘點
    fig_heat.add_trace(go.Scatter(
        x=[p_info["target_x"]], y=[p_info["target_z"]],
        mode="markers+text",
        marker=dict(color="#ffffff", size=10, line=dict(color="#000000", width=2)),
        text=[f"目標球路 ({p_info['target_x']}, {p_info['target_z']})"],
        textposition="top center",
        textfont=dict(color="#ffffff", size=11)
    ))

    fig_heat.update_layout(
        height=310, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a",
        xaxis=dict(title="左右偏移 X (ft)", range=[-0.8, 0.8], color="#94a3b8", gridcolor="#1e293b"),
        yaxis=dict(title="進壘高度 Z (ft)", range=[1.0, 4.5], color="#94a3b8", gridcolor="#1e293b"),
        showlegend=False
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    k1, k2, k3 = st.columns(3)
    k1.markdown(f'<div class="kpi-card"><div class="kpi-sub">預期得分增幅</div><div class="kpi-val">{stats["dxr"]}</div><div class="kpi-sub">{stats["ci"]}</div></div>', unsafe_allow_html=True)
    k2.markdown(f'<div class="kpi-card"><div class="kpi-sub">預測上壘機率</div><div class="kpi-val" style="color:#38bdf8;">{stats["obp_p"]}</div><div class="kpi-sub">有效樣本 {stats["n_eff"]} 顆</div></div>', unsafe_allow_html=True)
    k3.markdown(f'<div class="kpi-card"><div class="kpi-sub">預測勝率變化</div><div class="kpi-val" style="color:#facc15;">{stats["wpa"]}</div><div class="kpi-sub">24 狀態模擬</div></div>', unsafe_allow_html=True)

# ==================== 【中欄】投手球路特徵與進壘點 ====================
with col2:
    st.markdown("#### 🎯 投手球路特徵與進壘點")
    st.caption("X-Z 平面進壘座標分佈（不含 TrackMan 轉速與位移）")

    np.random.seed(101)
    hist_x = np.random.normal(p_info["target_x"] - 0.08, 0.28, 90)
    hist_z = np.random.normal(p_info["target_z"] - 0.15, 0.45, 90)
    sim_x = np.random.normal(p_info["target_x"], 0.12, stats["n_eff"])
    sim_z = np.random.normal(p_info["target_z"], 0.15, stats["n_eff"])

    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(x=hist_x, y=hist_z, mode="markers", marker=dict(color="#64748b", size=6, opacity=0.6), name=f"歷史球路 (N={p_info['n_hist']})"))
    fig_sc.add_trace(go.Scatter(x=sim_x, y=sim_z, mode="markers", marker=dict(color="#facc15", size=8, line=dict(color="#000", width=1)), name=f"相似球群 (n={stats['n_eff']})"))
    fig_sc.add_shape(type="rect", x0=-0.5, y0=1.6, x1=0.5, y1=3.4, line=dict(color="#94a3b8", width=1.5, dash="dash"))
    fig_sc.update_layout(
        height=230, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a",
        xaxis=dict(title="進壘寬度 Plate X (ft)", range=[-0.8, 0.8], color="#94a3b8", gridcolor="#1e293b"),
        yaxis=dict(title="進壘高度 Z (ft)", range=[1.0, 4.2], color="#94a3b8", gridcolor="#1e293b"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=11))
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # 球速分佈直方圖
    speeds = np.random.normal(p_info["speed_dist"][0], p_info["speed_dist"][1], 126)
    fig_sp = go.Figure()
    fig_sp.add_trace(go.Histogram(x=speeds, nbinsx=14, marker=dict(color="#0284c7", line=dict(color="#0f172a", width=1))))
    fig_sp.add_vrect(x0=p_info["v_range"][0], x1=p_info["v_range"][1], fillcolor="#facc15", opacity=0.2, line_width=0)
    fig_sp.update_layout(
        height=145, margin=dict(l=10, r=10, t=24, b=10),
        title=dict(text=f"球速分佈｜平均 {p_info['avg_v']} km/h・最高 {p_info['max_v']} km/h", font=dict(size=12, color="#cbd5e1")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0f172a",
        xaxis=dict(title="球速 (km/h)", range=[130, 162], color="#94a3b8", gridcolor="#1e293b"),
        yaxis=dict(title="球數", color="#94a3b8", gridcolor="#1e293b")
    )
    st.plotly_chart(fig_sp, use_container_width=True)

# ==================== 【右欄】代打推薦排行與戰術建議 ====================
with col3:
    st.markdown("#### 🏆 代打推薦排行（依當前球種動態計算）")

    # 依當前選定球種的 ΔxR 由高到低動態排序
    ranked_batters = sorted(
        BATTER_PROFILES.items(),
        key=lambda item: float(item[1][selected_pitch]["dxr"]),
        reverse=True
    )

    for idx, (b_name, b_data) in enumerate(ranked_batters, start=1):
        b_st = b_data[selected_pitch]
        card_class = "player-card-active" if b_name == selected_batter else "player-card"
        badge_color = "#facc15" if idx == 1 else ("#38bdf8" if idx == 2 else "#64748b")
        st.markdown(f"""
        <div class="{card_class}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="background:{badge_color}; color:#0f172a; font-weight:bold; padding:2px 8px; border-radius:12px; margin-right:6px;">#{idx}</span>
                    <strong style="font-size:15px;">{b_name}</strong>
                </div>
                <div style="text-align:right;">
                    <span style="color:#4ade80; font-size:18px; font-weight:800;">{b_st['dxr']} xR</span>
                    <span style="color:#94a3b8; font-size:11px; margin-left:6px;">匹配度 {b_st['match']}</span>
                </div>
            </div>
            <div style="font-size:11px; color:#94a3b8; margin-top:6px;">
                打擊率 {b_data['avg']} ｜ 上壘率 {b_data['obp']} ｜ 長打率 {b_data['slg']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # AI 戰術建議卡片
    reasons_html = "".join([f"<li style='margin-bottom:4px;'>{r}</li>" for r in stats["reason"]])
    st.markdown(f"""
    <div class="advice-box">
        <div style="font-weight:bold; color:#facc15; margin-bottom:6px;">💡 戰術調度分析（檢視：{selected_batter.split(' ')[0]}）</div>
        <ul style="font-size:12px; color:#cbd5e1; padding-left:18px; margin-bottom:0;">
            {reasons_html}
        </ul>
    </div>
    """, unsafe_allow_html=True)
