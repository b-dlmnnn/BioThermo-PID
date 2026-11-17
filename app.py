import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json

st.set_page_config(page_title="BioThermo-PID | Portable Detection Cycler", page_icon="🌡️", layout="wide")


def simulate_pid_thermal_cycler(kp, ki, kd, ambient_temp, fault_mode, seed=42):
    rng = np.random.default_rng(seed)
    dt = 0.5
    total_steps = 480
    time_s = np.arange(0, total_steps * dt, dt)

    # 0-180s: 65.0°C LAMP Amplification | 180-215s: 95.0°C Enzyme Inactivation | 215-240s: 25.0°C Active Cooling
    setpoints = np.where(time_s < 180.0, 65.0, np.where(time_s < 215.0, 95.0, 25.0))

    temp = float(ambient_temp)
    integral = 0.0
    prev_error = 0.0

    # CNC Aluminum Micro-Heater Block Physical Parameters
    thermal_mass = 18.0
    heat_loss_coeff = 0.28
    max_heater_power = 45.0 if fault_mode != "Heater Degradation (-40% Power)" else 27.0
    fan_cooling_coeff = 1.45 if fault_mode != "Cooling Fan Failure" else 0.30

    measured_temps = list()
    true_temps = list()
    pwm_duties = list()
    fan_states = list()

    for step_idx, sp in enumerate(setpoints):
        sensor_bias = (0.015 * step_idx) if fault_mode == "Thermistor Drift (+0.015°C/s)" else 0.0
        meas_temp = temp + float(rng.normal(0, 0.08)) + sensor_bias

        error = sp - meas_temp
        integral = np.clip(integral + error * dt, -150.0, 150.0)
        derivative = (error - prev_error) / dt
        prev_error = error

        raw_output = (kp * error) + (ki * integral) + (kd * derivative)
        pwm_pct = float(np.clip(raw_output, 0.0, 100.0))

        fan_on = 100.0 if (sp < temp - 2.0) else 0.0

        heat_in = (pwm_pct / 100.0) * max_heater_power
        heat_out = heat_loss_coeff * (temp - ambient_temp) + (fan_on / 100.0) * fan_cooling_coeff * (temp - ambient_temp)
        temp += ((heat_in - heat_out) / thermal_mass) * dt

        true_temps.append(round(temp, 2))
        measured_temps.append(round(meas_temp, 2))
        pwm_duties.append(round(pwm_pct, 1))
        fan_states.append(fan_on)

    return pd.DataFrame({
        "Time (s)": time_s,
        "Target Setpoint (°C)": setpoints,
        "Measured Temp (°C)": measured_temps,
        "True Block Temp (°C)": true_temps,
        "Heater PWM (%)": pwm_duties,
        "Cooling Fan (%)": fan_states
    })


st.sidebar.header("🎛️ Biomechatronic PID Controls")
kp = st.sidebar.slider("Proportional Gain (Kp)", 1.0, 25.0, 11.5, 0.5)
ki = st.sidebar.slider("Integral Gain (Ki)", 0.0, 5.0, 1.2, 0.1)
kd = st.sidebar.slider("Derivative Gain (Kd)", 0.0, 15.0, 4.0, 0.5)

st.sidebar.markdown("---")
ambient_c = st.sidebar.slider("Ambient Lab Temp (°C)", 15.0, 35.0, 25.0, 0.5)
fault_sim = st.sidebar.selectbox(
    "Hardware Fault Injection (Diagnostics)",
    (
        "Nominal Hardware (Healthy)",
        "Heater Degradation (-40% Power)",
        "Thermistor Drift (+0.015°C/s)",
        "Cooling Fan Failure"
    )
)
sim_seed = st.sidebar.slider("Thermistor Noise Seed", 1, 50, 12)

st.sidebar.markdown("---")
st.sidebar.info("**Academic Transparency Note:** Lumped-capacitance thermal simulation for portable CNC aluminum block LAMP/PCR cyclers & predictive fault diagnostics.")

st.title("🌡️ BioThermo-PID: Portable Detection Cycler & Edge Controller")
st.markdown("**Closed-Loop PID Temperature Control, PWM Actuation & Predictive Hardware Diagnostics**")

df_sim = simulate_pid_thermal_cycler(kp, ki, kd, ambient_c, fault_sim, sim_seed)

lamp_win = df_sim[(df_sim["Time (s)"] >= 60.0) & (df_sim["Time (s)"] <= 175.0)]
overshoot_c = round(max(0.0, float(df_sim[df_sim["Time (s)"] < 180.0]["Measured Temp (°C)"].max()) - 65.0), 2)
rmse_lamp = round(float(np.sqrt(np.mean((lamp_win["Measured Temp (°C)"] - 65.0) ** 2))), 3)
avg_pwm = round(float(lamp_win["Heater PWM (%)"].mean()), 1)
max_temp_reached = round(float(df_sim["Measured Temp (°C)"].max()), 1)

if fault_sim != "Nominal Hardware (Healthy)" or rmse_lamp > 0.6 or overshoot_c > 2.5:
    st.error(f"🚨 **HARDWARE / TUNING ALERT:** Mode = **{fault_sim}** | LAMP RMSE = **{rmse_lamp} °C** | Overshoot = **{overshoot_c} °C**. Inspection or PID retuning required.")
else:
    st.success(f"✅ **THERMAL CYCLER OPTIMAL:** LAMP Stability RMSE = **{rmse_lamp} °C** (Spec < 0.5 °C) | Overshoot = **{overshoot_c} °C** | Avg Holding PWM = **{avg_pwm}%**")

m1, m2, m3, m4 = st.columns(4)
m1.metric("LAMP Hold RMSE", f"{rmse_lamp} °C", delta="Spec < 0.50 °C", delta_color="inverse")
m2.metric("Peak Overshoot", f"{overshoot_c} °C", delta="Max +2.0 °C", delta_color="inverse")
m3.metric("Holding PWM", f"{avg_pwm} %", delta="45W Micro-Heater")
m4.metric("Inactivation Peak", f"{max_temp_reached} °C", delta="Target 95.0 °C")

fig_temp = go.Figure()
fig_temp.add_trace(go.Scatter(x=df_sim["Time (s)"], y=df_sim["Target Setpoint (°C)"], name="Target Setpoint (°C)", line=dict(color="#94a3b8", dash="dash", width=2)))
fig_temp.add_trace(go.Scatter(x=df_sim["Time (s)"], y=df_sim["Measured Temp (°C)"], name="Thermistor Measured (°C)", line=dict(color="#ef4444", width=2.5)))
if fault_sim == "Thermistor Drift (+0.015°C/s)":
    fig_temp.add_trace(go.Scatter(x=df_sim["Time (s)"], y=df_sim["True Block Temp (°C)"], name="True Block Temp (°C)", line=dict(color="#3b82f6", dash="dot", width=2)))

fig_temp.update_layout(
    title="Closed-Loop Thermal Profile: 65°C LAMP Amplification → 95°C Enzyme Inactivation → Active Fan Cooling",
    xaxis_title="Time (seconds)",
    yaxis_title="Temperature (°C)",
    height=350,
    margin=dict(l=10, r=10, t=40, b=10)
)
st.plotly_chart(fig_temp, use_container_width=True)

c_act, c_exp = st.columns((1.4, 1.0))
with c_act:
    st.subheader("⚡ Actuator Telemetry (Heater PWM & Cooling Fan)")
    fig_pwm = go.Figure()
    fig_pwm.add_trace(go.Scatter(x=df_sim["Time (s)"], y=df_sim["Heater PWM (%)"], name="Heater PWM Duty (%)", fill="tozeroy", line=dict(color="#f59e0b")))
    fig_pwm.add_trace(go.Scatter(x=df_sim["Time (s)"], y=df_sim["Cooling Fan (%)"], name="Cooling Fan State (%)", line=dict(color="#06b6d4")))
    fig_pwm.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Duty Cycle (%)")
    st.plotly_chart(fig_pwm, use_container_width=True)

with c_exp:
    st.subheader("📤 Edge Firmware Telemetry Export")
    st.dataframe(df_sim.tail(7), use_container_width=True, hide_index=True)
    csv_data = df_sim.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Thermal Log (CSV)", data=csv_data, file_name="biothermo_pid_log.csv", mime="text/csv")
