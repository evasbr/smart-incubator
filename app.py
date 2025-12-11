import streamlit as st
import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sqlite3
import os
import json
import joblib
import threading
import paho.mqtt.client as mqtt
from datetime import datetime

# -----------------------------------------------------------------------------
# 1. KONFIGURASI & SETUP GLOBAL
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard IoT Smart Incubator",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Config Variables
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC_SENSOR = "iot/stage3/sensor"
MQTT_TOPIC_OUTPUT = "iot/stage3/output"
DB_NAME = "iot_database.db"
MODEL_PATH = "model.pkl"

# -----------------------------------------------------------------------------
# 2. CUSTOM CSS (Style Card Putih & Clean)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Poppins', sans-serif !important; }
    header[data-testid="stHeader"] { display: none; }
    .stMainBlockContainer { padding: 0 20px; padding-top: 1rem; }
    
    /* 1. STYLE KARTU HTML BIASA (Untuk Metrics & Notif) */
    .css-card { 
        border-radius: 10px; 
        padding: 20px; 
        background-color: white; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        margin-bottom: 20px; 
        height: 100%; 
    }
    
    /* 2. STYLE KARTU CONTAINER (Untuk Chart & Table) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: none !important;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Metrics */
    .metric-label { font-size: 13px; font-weight: 500; margin-bottom: 5px; color: #555; text-transform: uppercase; letter-spacing: 0.5px;}
    .metric-value { font-size: 28px; font-weight: bold; color: #000; margin: 0; }
    .metric-sub { font-size: 11px; color: #888; margin-top: 5px; }
    
    /* Text Colors */
    .text-red { color: #d9534f; } .text-blue { color: #0275d8; } .text-green { color: #5cb85c; }
    .text-orange { color: #f0ad4e; } .text-purple { color: #6f42c1; }
    
    /* Notifications */
    .notif-container { height: 420px; overflow-y: auto; padding-right: 5px; }
    .notif-item { padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid; }
    .notif-desc { font-size: 14px; margin: 5px 0; color: #333; } .notif-time { font-size: 11px; color: #888; }
    .notif-blue { background-color: #e8f4fd; border-color: #2196F3; } 
    .notif-blue h4 { color: #0d47a1; margin: 0; font-size: 16px; font-weight: bold;}
    .notif-red { background-color: #fdecea; border-color: #f44336; } 
    .notif-red h4 { color: #b71c1c; margin: 0; font-size: 16px; font-weight: bold;}
    
    /* Summary & Alert */
    .summary-card { background: white; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); border: 1px solid #eee; }
    .summary-val { font-size: 24px; font-weight: bold; margin: 0; }
    .summary-label { font-size: 12px; color: #666; }
    .alert-banner { padding: 15px 20px; border-radius: 8px; margin-bottom: 25px; display: flex; align-items: center; box-shadow: 0 2px 5px rgba(0,0,0,0.05); animation: pulse 2s infinite; }
    .alert-banner-red { background-color: #ffebee; border: 1px solid #ffcdd2; color: #c62828; }
    .alert-banner-blue { background-color: #e3f2fd; border: 1px solid #bbdefb; color: #1565c0; }
    .alert-icon { font-size: 24px; margin-right: 15px; }
    
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(0, 0, 0, 0.1); } 70% { box-shadow: 0 0 0 10px rgba(0, 0, 0, 0); } 100% { box-shadow: 0 0 0 0 rgba(0, 0, 0, 0); } }
    
    /* Radio Buttons */
    div[role="radiogroup"] > label { background-color: #f0f2f6; padding: 5px 15px; border-radius: 20px; border: 1px solid #e0e0e0; margin-right: 5px; font-size: 14px; }
    div[role="radiogroup"] > label[data-checked="true"] { background-color: #0275d8; color: white; border-color: #0275d8; }
    .stPlotlyChart { background-color: white; border-radius: 0px; box-shadow: none; padding: 0px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. BACKEND LOGIC (Database & MQTT)
# -----------------------------------------------------------------------------

# --- Database Functions ---
def init_db():
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            temp REAL,
            hum REAL,
            prediction TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(ts, temp, hum, pred):
    try:
        conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (timestamp, temp, hum, prediction) VALUES (?, ?, ?, ?)",
            (ts, temp, hum, str(pred))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"❌ Gagal simpan DB: {e}")

def get_data_from_db(date_obj):
    if date_obj > datetime.now().date(): return pd.DataFrame()
    if not os.path.exists(DB_NAME): return pd.DataFrame()
    
    try:
        conn = sqlite3.connect(DB_NAME, check_same_thread=False)
        query = "SELECT timestamp, temp, hum, prediction FROM logs WHERE date(timestamp) = ? ORDER BY timestamp ASC"
        df = pd.read_sql_query(query, conn, params=(date_obj,))
        conn.close()
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        return pd.DataFrame()

# --- Load Model (Cached) ---
@st.cache_resource
def load_ai_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        return None

# --- MQTT Worker ---
def run_mqtt_client():
    try:
        model = joblib.load(MODEL_PATH)
    except:
        model = None
        print("⚠️ Model tidak ditemukan untuk MQTT Worker")

    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            client.subscribe(MQTT_TOPIC_SENSOR)
            print("🔗 Connected to MQTT Broker")

    def on_message(client, userdata, msg):
        try:
            payload = msg.payload.decode()
            data = json.loads(payload)
            temp = float(data.get("temp", 0))
            hum = float(data.get("hum", 0))
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Prediksi
            pred = "Normal"
            if model:
                X = np.array([[temp, hum]])
                pred = model.predict(X)[0]

            # Simpan ke DB
            save_to_db(ts, temp, hum, pred)

            # Logic Kontrol
            if pred == "TOO_WARM":
                client.publish(MQTT_TOPIC_OUTPUT, "LOWER_TEMP")
            elif pred == "TOO_COOL":
                client.publish(MQTT_TOPIC_OUTPUT, "RAISE_TEMP")
            else:
                client.publish(MQTT_TOPIC_OUTPUT, "ALERT_OFF")
                
        except Exception as e:
            print(f"⚠️ Error MQTT: {e}")

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except Exception as e:
        print(f"❌ MQTT Connection Failed: {e}")

if 'mqtt_thread' not in st.session_state:
    init_db()
    mqtt_thread = threading.Thread(target=run_mqtt_client, daemon=True)
    mqtt_thread.start()
    st.session_state.mqtt_thread = True

# -----------------------------------------------------------------------------
# 4. FRONTEND LOGIC (Helper Calculation)
# -----------------------------------------------------------------------------
def calculate_status_duration(df):
    if df.empty: return False, None, None, None
    latest_row = df.iloc[-1]
    current_status = latest_row['prediction']
    latest_time = latest_row['timestamp']

    if current_status not in ['TOO_WARM', 'TOO_COOL']:
        return False, None, None, None

    start_time = latest_time
    for i in range(len(df) - 2, -1, -1):
        row = df.iloc[i]
        if row['prediction'] == current_status:
            start_time = row['timestamp']
        else:
            break

    duration = latest_time - start_time
    total_seconds = int(duration.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    
    time_str = ""
    if hours > 0: time_str += f"{hours}h "
    time_str += f"{minutes}m"
    if total_seconds < 60: time_str = "< 1m"

    return True, current_status, time_str.strip(), start_time.strftime("%H:%M")

# -----------------------------------------------------------------------------
# 5. UI LAYOUT & FRAGMENT
# -----------------------------------------------------------------------------

# Header
col_h1, col_h2 = st.columns([8, 2], vertical_alignment="bottom")
with col_h1: st.markdown("<h1 style='padding-bottom: 0px; margin-bottom: 0px;'>Dashboard</h1>", unsafe_allow_html=True)
with col_h2:
    today = datetime.now().date()
    selected_date = st.date_input("Pilih Tanggal", value=today, label_visibility="collapsed", format="DD/MM/YYYY")
st.divider()

model_status = "✅ Loaded" if load_ai_model() else "❌ Missing"

# Main Live Dashboard Fragment
@st.fragment(run_every=2) 
def show_live_dashboard(current_date):
    
    # Fetch Data
    df = get_data_from_db(current_date)
    
    # Init Default Value
    h_temp = l_temp = avg_temp = 0
    h_hum = l_hum = avg_hum = 0
    t_h_temp = t_l_temp = t_h_hum = t_l_hum = "-"
    count_warm = count_cold = 0
    last_update = "-"

    if not df.empty:
        h_temp = df['temp'].max(); l_temp = df['temp'].min(); avg_temp = df['temp'].mean()
        t_h_temp = df.loc[df['temp'].idxmax(), 'timestamp'].strftime("%H:%M")
        t_l_temp = df.loc[df['temp'].idxmin(), 'timestamp'].strftime("%H:%M")

        h_hum = df['hum'].max(); l_hum = df['hum'].min(); avg_hum = df['hum'].mean()
        t_h_hum = df.loc[df['hum'].idxmax(), 'timestamp'].strftime("%H:%M")
        t_l_hum = df.loc[df['hum'].idxmin(), 'timestamp'].strftime("%H:%M")
        
        count_warm = df[df['prediction'] == 'TOO_WARM'].shape[0]
        count_cold = df[df['prediction'] == 'TOO_COOL'].shape[0]
        last_update = df['timestamp'].iloc[-1].strftime("%H:%M:%S")

    # Alert Banner
    is_alert, status_type, duration_str, start_time_str = calculate_status_duration(df)
    if is_alert:
        if status_type == 'TOO_WARM':
            alert_class = "alert-banner-red"; icon = "🔥"; title = "CRITICAL ALERT: HIGH TEMPERATURE"
            msg = f"System <b>TOO WARM</b> for <b>{duration_str}</b> (since {start_time_str})."
        else:
            alert_class = "alert-banner-blue"; icon = "❄️"; title = "WARNING: LOW TEMPERATURE"
            msg = f"System <b>TOO COLD</b> for <b>{duration_str}</b> (since {start_time_str})."
        st.markdown(f"""<div class="alert-banner {alert_class}"><div class="alert-icon">{icon}</div><div class="alert-content"><strong>{title}</strong><span>{msg}</span></div></div>""", unsafe_allow_html=True)

    # Metrics Row
    def create_metric_card(label, value, subtext, color_class):
        return f"""<div class="css-card"><div class="metric-label {color_class}">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{subtext}</div></div>"""

    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(create_metric_card("Highest Temp", f"{h_temp:.1f} °C", f"At {t_h_temp}", "text-red"), unsafe_allow_html=True)
    with c2: st.markdown(create_metric_card("Lowest Temp", f"{l_temp:.1f} °C", f"At {t_l_temp}", "text-blue"), unsafe_allow_html=True)
    with c3: st.markdown(create_metric_card("Average Temp", f"{avg_temp:.1f} °C", f"Upd: {last_update}", "text-green"), unsafe_allow_html=True)
    
    c4, c5, c6 = st.columns(3)
    with c4: st.markdown(create_metric_card("Highest Hum", f"{h_hum:.1f} %", f"At {t_h_hum}", "text-orange"), unsafe_allow_html=True)
    with c5: st.markdown(create_metric_card("Lowest Hum", f"{l_hum:.1f} %", f"At {t_l_hum}", "text-purple"), unsafe_allow_html=True)
    with c6: st.markdown(create_metric_card("Average Hum", f"{avg_hum:.1f} %", f"Upd: {last_update}", "text-green"), unsafe_allow_html=True)

    # Main Layout
    col_main, col_side = st.columns([2.2, 0.8])

    with col_main:
        with st.container(): 
            st.write("#### 📈 Real-time Monitoring")
            
            if 'time_filter' not in st.session_state: st.session_state.time_filter = 'All Day'
            filter_options = ['1 Min', '1 Jam', '3 Jam', '6 Jam', '12 Jam', 'All Day']
            selected_filter = st.radio("Interval Grid & Zoom:", filter_options, horizontal=True, key="time_filter", label_visibility="collapsed")

            max_time = df['timestamp'].max() if not df.empty else datetime.now()
            min_time = df['timestamp'].min() if not df.empty else datetime.now()
            dtick_val = None; tick_fmt = "%H:%M"; start_range = min_time

            if not df.empty:
                if selected_filter == '1 Min': start_range = max_time - pd.Timedelta(minutes=10); dtick_val = 60000; tick_fmt = "%H:%M"
                elif selected_filter == '1 Jam': start_range = max_time - pd.Timedelta(hours=5); dtick_val = 3600000
                elif selected_filter == '3 Jam': start_range = max_time - pd.Timedelta(hours=12); dtick_val = 3600000 * 3
                elif selected_filter == '6 Jam': start_range = max_time - pd.Timedelta(hours=24); dtick_val = 3600000 * 6
                elif selected_filter == '12 Jam': start_range = min_time; dtick_val = 3600000 * 12
                else: start_range = min_time; dtick_val = None 

            if not df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['temp'], mode='lines', name='Temp (°C)', line=dict(color='#d9534f', width=2, shape='linear'), fill='tozeroy', fillcolor='rgba(217, 83, 79, 0.1)'))
                fig.add_trace(go.Scatter(x=df['timestamp'], y=df['hum'], mode='lines', name='Humidity (%)', line=dict(color='#0275d8', width=2, shape='linear'), fill='tozeroy', fillcolor='rgba(2, 117, 216, 0.1)'))
                fig.update_layout(dragmode='pan', paper_bgcolor='white', plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=10, l=0, r=0, b=0), height=400, hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    xaxis=dict(showgrid=True, gridcolor='#eee', tickformat=tick_fmt, dtick=dtick_val, range=[start_range, max_time]), 
                    yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
                )
                st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displayModeBar': False})
            else:
                st.warning("Menunggu data dari backend MQTT...")

    with col_side:
        st.write("#### 🔔 Notifications")
        s1, s2 = st.columns(2)
        with s1: st.markdown(f"""<div class="summary-card" style="border-bottom: 3px solid #f44336; padding:10px;"><div class="summary-val text-red">{count_warm}</div><div class="summary-label">Too Warm</div></div>""", unsafe_allow_html=True)
        with s2: st.markdown(f"""<div class="summary-card" style="border-bottom: 3px solid #2196F3; padding:10px;"><div class="summary-val text-blue">{count_cold}</div><div class="summary-label">Too Cold</div></div>""", unsafe_allow_html=True)
        
        st.write("") 
        notif_html = ""
        if not df.empty:
            alerts = df[df['prediction'].isin(['TOO_WARM', 'TOO_COOL'])].sort_values(by='timestamp', ascending=False).head(50)
            for _, row in alerts.iterrows():
                waktu = row['timestamp'].strftime("%H:%M")
                status = row['prediction']
                suhu = row['temp']
                css = "notif-red" if status == 'TOO_WARM' else "notif-blue"
                icon = "🔥" if status == 'TOO_WARM' else "❄️"
                desc = f"Alert: {status} detected ({suhu:.1f}°C)"
                notif_html += f"""<div class="notif-item {css}" style="padding:10px;"><div style="display:flex; justify-content:space-between; align-items:center;"><h4 style="margin:0; font-size:14px;">{icon} {status}</h4><span class="notif-time">{waktu}</span></div><div class="notif-desc" style="font-size:12px; margin-top:2px;">{desc}</div></div>"""
            if notif_html == "": notif_html = "<div style='padding:10px; color:#ccc; text-align:center;'>System normal.</div>"
        else:
            notif_html = "<div style='padding:10px; color:#ccc; text-align:center;'>No Data.</div>"
        st.markdown(f"""<div class="css-card notif-container">{notif_html}</div>""", unsafe_allow_html=True)

    with st.container():
        st.write("#### 📋 Data Log & Export")
        cols_tbl = st.columns([4, 1])
        with cols_tbl[0]: st.caption(f"Menampilkan 10 data terakhir. Total: {len(df)} baris.")
        with cols_tbl[1]:
            if not df.empty:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(label="📥 Export CSV", data=csv, file_name=f'iot_data_{current_date}.csv', mime='text/csv', use_container_width=True)
        
        if not df.empty:
            st.dataframe(df.sort_values(by='timestamp', ascending=False).head(10), column_config={"timestamp": st.column_config.DatetimeColumn("Timestamp", format="D MMM, HH:mm:ss"), "temp": st.column_config.NumberColumn("Temp (°C)", format="%.1f"), "hum": st.column_config.NumberColumn("Hum (%)", format="%.1f"), "prediction": "Status"}, use_container_width=True, hide_index=True)
        else:
            st.info("Belum ada data terekam hari ini.")

# -----------------------------------------------------------------------------
# 6. RUN
# -----------------------------------------------------------------------------
show_live_dashboard(selected_date)