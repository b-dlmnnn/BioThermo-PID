import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="BioThermo-PID | Closed-Loop Controller Simulation", page_icon="🌡️", layout="wide")


def run_thermal_simulation(kp, ki, kd, ambient_temp, fault_mode, use_anti_windup=True, use_ema=True, seed=12):
    rng = np.random.default_rng(seed)
    dt = 0.5
    total_steps = 480
    time_s = np.arange(0, total_steps * dt, dt)
    setpoints = np.where(time_s < 180.0, 65.0, np.where(time_s < 215.0, 95.0, 25.0))

    temp = float(ambient_temp)
    ema_temp = float(ambient_temp)
    prev_meas = float(ambient_temp)
    integral = 0.0

    thermal_mass = 18.0
    heat_loss_coeff = 0.28
    max_heater_power = 45.0 if fault_mode != "Heater Degradation (-40% Power)" else 27.0
    fan_cooling_coeff = 1.45 if fault_mode != "Cooling Fan Failure" else 0.30
    ema_alpha = 0.35 if use_ema else 1.0

    true_temps = list()
    raw_meas_temps = list()
    filt_temps = list()
    pwm_duties = list()
    fan_states = list()

    for step_idx, sp in enumerate(setpoints):
        sensor_bias = (0.015 * step_idx) if fault_mode == "Thermistor Drift (+0.015°C/s)" else 0.0
        raw_meas = temp + float(rng.normal(0, 0.12)) + sensor_bias
        ema_temp = (ema_alpha * raw_meas) + ((1.0 - ema_alpha) * ema_temp)

        error = sp - ema_temp
        derivative_meas = -(ema_temp - prev_meas) / dt
        prev_meas = ema_temp

        candidate_integral = integral + error * dt
        raw_u = (kp * error) + (ki * candidate_integral) + (kd * derivative_meas)

        if use_anti_windup:
            if not ((raw_u > 100.0 and error > 0) or (raw_u < 0.0 and error < 0)):
                integral = np.clip(candidate_integral, -120.0, 120.0)
            pwm_pct = float(np.clip((kp * error) + (ki * integral) + (kd * derivative_meas), 0.0, 100.0))
        else:
            integral = candidate_integral
            pwm_pct = float(np.clip(raw_u, 0.0, 100.0))

        fan_on = 100.0 if (sp < temp - 2.0) else 0.0
        heat_in = (pwm_pct / 100.0) * max_heater_power
        heat_out = heat_loss_coeff * (temp - ambient_temp) + (fan_on / 100.0) * fan_cooling_coeff * (temp - ambient_temp)
        temp += ((heat_in - heat_out) / thermal_mass) * dt

        true_temps.append(round(temp, 2))
        raw_meas_temps.append(round(raw_meas, 2))
        filt_temps.append(round(ema_temp, 2))
        pwm_duties.append(round(pwm_pct, 1))
        fan_states.append(fan_on)

    return pd.DataFrame({
        "Time (s)": time_s,
        "Target Setpoint (°C)": setpoints,
        "True Block Temp (°C)": true_temps,
        "Raw Thermistor (°C)": raw_meas_temps,
        "Filtered Temp (°C)": filt_temps,
        "Heater PWM (%)": pwm_duties,
        "Cooling Fan (%)": fan_states
    })


def compute_step_metrics(df):
    lamp_all = df[df["Time (s)"] < 180.0]
    steady_win = df[(df["Time (s)"] >= 75.0) & (df["Time (s)"] <= 175.0)]

    t10 = lamp_all[lamp_all["True Block Temp (°C)"] >= 29.0]["Time (s)"]
    t90 = lamp_all[lamp_all["True Block Temp (°C)"] >= 61.0]["Time (s)"]
    rise_time = round(float(t90.values.item(0) - t10.values.item(0)), 1) if (len(t10) > 0 and len(t90) > 0) else 180.0

    outside = lamp_all[np.abs(lamp_all["True Block Temp (°C)"] - 65.0) > 0.5]["Time (s)"]
    settling_time = round(float(outside.values.item(-1)), 1) if len(outside) > 0 else 0.0

    overshoot = round(max(0.0, float(lamp_all["True Block Temp (°C)"].max()) - 65.0), 2)
    ss_error = round(abs(float(steady_win["True Block Temp (°C)"].mean()) - 65.0), 3)
    true_rmse = round(float(np.sqrt(np.mean((steady_win["True Block Temp (°C)"] - 65.0) ** 2))), 3)
    meas_rmse = round(float(np.sqrt(np.mean((steady_win["Filtered Temp (°C)"] - 65.0) ** 2))), 3)
    avg_pwm = round(float(steady_win["Heater PWM (%)"].mean()), 1)

    return {
        "Rise Time (s)": rise_time,
        "Settling Time (s)": settling_time,
        "Overshoot (°C)": overshoot,
        "Steady-State Err (°C)": ss_error,
        "True Block RMSE (°C)": true_rmse,
        "Measured RMSE (°C)": meas_rmse,
        "Hold PWM (%)": avg_pwm
    }

st.sidebar.header("🎛️ PID & Filter Controls")
tuning_preset = st.sidebar.radio("PID Tuning Mode", ("Auto-Tuned (Anti-Windup + D-Filter)", "Untuned Aggressive Baseline", "Manual Custom Gains"))

if tuning_preset == "Auto-Tuned (Anti-Windup + D-Filter)":
    kp, ki, kd, use_aw, use_ema = 6.2, 0.22, 4.5, True, True
elif tuning_preset == "Untuned Aggressive Baseline":
    kp, ki, kd, use_aw, use_ema = 12.0, 1.40, 1.0, False, False
else:
    kp = st.sidebar.slider("Proportional Gain (Kp)", 1.0, 20.0, 6.2, 0.2)
    ki = st.sidebar.slider("Integral Gain (Ki)", 0.0, 3.0, 0.22, 0.02)
    kd = st.sidebar.slider("Derivative Gain (Kd)", 0.0, 12.0, 4.5, 0.5)
    use_aw = st.sidebar.checkbox("Enable Anti-Windup Clamping", value=True)
    use_ema = st.sidebar.checkbox("Enable EMA Low-Pass Filter", value=True)

st.sidebar.markdown("---")
ambient_c = st.sidebar.slider("Ambient Lab Temp (°C)", 15.0, 35.0, 25.0, 0.5)
fault_sim = st.sidebar.selectbox("Hardware Fault Injection", ("Nominal Hardware (Healthy)", "Heater Degradation (-40% Power)", "Thermistor Drift (+0.015°C/s)", "Cooling Fan Failure"))
sim_seed = st.sidebar.slider("Thermistor Noise Seed", 1, 50, 12)
st.sidebar.markdown("---")
st.sidebar.info("**Note:** Software simulation only (time-compressed 240s thermal model) used to test PID parameters before connecting physical heater hardware.")

st.title("🌡️ BioThermo-PID: Closed-Loop Controller Simulation")
st.markdown("**Time-Compressed LAMP/PCR Thermal Cycler Model, Anti-Windup PID & Hardware Fault Diagnostics**")

df_curr = run_thermal_simulation(kp, ki, kd, ambient_c, fault_sim, use_aw, use_ema, sim_seed)
m_curr = compute_step_metrics(df_curr)

if fault_sim == "Nominal Hardware (Healthy)" and m_curr.get("True Block RMSE (°C)") < 0.50 and m_curr.get("Overshoot (°C)") <= 2.0:
    st.success(f"✅ **CONTROLLER WITHIN SPEC:** True Block RMSE = **{m_curr.get('True Block RMSE (°C)')} °C** (< 0.50 °C) | Overshoot = **{m_curr.get('Overshoot (°C)')} °C** | Hold Power = **{round((m_curr.get('Hold PWM (%)')/100)*45, 1)} W**")
else:
    st.error(f"🚨 **SPEC / HARDWARE ALERT:** Mode = **{fault_sim}** | True RMSE = **{m_curr.get('True Block RMSE (°C)')} °C** | Overshoot = **{m_curr.get('Overshoot (°C)')} °C**.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("True Block RMSE", f"{m_curr.get('True Block RMSE (°C)')} °C", delta="Spec < 0.50 °C", delta_color="inverse")
c2.metric("Measured RMSE", f"{m_curr.get('Measured RMSE (°C)')} °C", delta="Incl. Sensor Noise", delta_color="off")
c3.metric("Peak Overshoot", f"{m_curr.get('Overshoot (°C)')} °C", delta="Limit +2.0 °C", delta_color="inverse")
c4.metric("Settling Time", f"{m_curr.get('Settling Time (s)')} s", delta=f"Rise: {m_curr.get('Rise Time (s)')} s", delta_color="off")
c5.metric("Hold Heater PWM", f"{m_curr.get('Hold PWM (%)')} %", delta=f"~{round((m_curr.get('Hold PWM (%)')/100)*45, 1)} W / 45 W", delta_color="off")

fig_temp = go.Figure()
fig_temp.add_trace(go.Scatter(x=df_curr["Time (s)"], y=df_curr["Target Setpoint (°C)"], name="Target Setpoint (°C)", line=dict(color="#94a3b8", dash="dash", width=2)))
fig_temp.add_trace(go.Scatter(x=df_curr["Time (s)"], y=df_curr["Raw Thermistor (°C)"], name="Raw Thermistor (°C)", line=dict(color="#64748b", width=1), opacity=0.55))
fig_temp.add_trace(go.Scatter(x=df_curr["Time (s)"], y=df_curr["True Block Temp (°C)"], name="True Block Temp (°C)", line=dict(color="#ef4444", width=2.5)))
if fault_sim == "Thermistor Drift (+0.015°C/s)":
    fig_temp.add_trace(go.Scatter(x=df_curr["Time (s)"], y=df_curr["Filtered Temp (°C)"], name="Drifted Sensor (°C)", line=dict(color="#f59e0b", dash="dot", width=2)))
fig_temp.update_layout(title="Time-Compressed Simulation: 65°C LAMP Hold → 95°C Inactivation → Cooling", xaxis_title="Compressed Simulation Time (s)", yaxis_title="Temperature (°C)", height=340, margin=dict(l=10, r=10, t=40, b=10))
st.plotly_chart(fig_temp, use_container_width=True)

st.markdown("---")
col_comp, col_phys = st.columns((1.25, 1.0))
with col_comp:
    st.subheader("📊 Untuned vs. Auto-Tuned PID Benchmark")
    m_un = compute_step_metrics(run_thermal_simulation(12.0, 1.40, 1.0, ambient_c, "Nominal Hardware (Healthy)", False, False, sim_seed))
    m_tu = compute_step_metrics(run_thermal_simulation(6.2, 0.22, 4.5, ambient_c, "Nominal Hardware (Healthy)", True, True, sim_seed))
    bench_df = pd.DataFrame({
        "Metric": ("Rise Time (10-90%)", "Settling Time (±0.5°C)", "Peak Overshoot", "Steady-State Err", "True Block RMSE"),
        "Untuned Baseline": (f"{m_un.get('Rise Time (s)')} s", f"{m_un.get('Settling Time (s)')} s", f"{m_un.get('Overshoot (°C)')} °C", f"{m_un.get('Steady-State Err (°C)')} °C", f"{m_un.get('True Block RMSE (°C)')} °C"),
        "Auto-Tuned (AW+EMA)": (f"{m_tu.get('Rise Time (s)')} s", f"{m_tu.get('Settling Time (s)')} s", f"{m_tu.get('Overshoot (°C)')} °C", f"{m_tu.get('Steady-State Err (°C)')} °C", f"{m_tu.get('True Block RMSE (°C)')} °C")
    })
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

with col_phys:
    st.subheader("⚙️ Lumped-Capacitance Physical Specs")
    exp_w = round(0.28 * (65.0 - ambient_c), 2)
    phys_df = pd.DataFrame({
        "Parameter": ("Thermal Mass (C_th)", "Heat Loss Coeff (hA)", "Max Heater Power", "Hold Power (65°C)", "Theoretical Hold PWM"),
        "Value": ("18.0 J/°C", "0.28 W/°C", "45.0 W", f"{exp_w} W", f"{round((exp_w/45.0)*100.0, 1)} %")
    })
    st.dataframe(phys_df, use_container_width=True, hide_index=True)

st.markdown("---")
c_act, c_exp = st.columns((1.4, 1.0))
with c_act:
    st.subheader("⚡ Actuator Telemetry (Heater PWM & Cooling Fan)")
    fig_pwm = go.Figure()
    fig_pwm.add_trace(go.Scatter(x=df_curr["Time (s)"], y=df_curr["Heater PWM (%)"], name="Heater PWM (%)", fill="tozeroy", line=dict(color="#f59e0b")))
    fig_pwm.add_trace(go.Scatter(x=df_curr["Time (s)"], y=df_curr["Cooling Fan (%)"], name="Cooling Fan (%)", line=dict(color="#06b6d4")))
    fig_pwm.update_layout(height=230, margin=dict(l=10, r=10, t=20, b=10), yaxis_title="Duty (%)")
    st.plotly_chart(fig_pwm, use_container_width=True)

with c_exp:
    st.subheader("📤 Simulation Telemetry Log")
    st.dataframe(df_curr.tail(6), use_container_width=True, hide_index=True)
    st.download_button("📥 Download Thermal Log (CSV)", data=df_curr.to_csv(index=False).encode("utf-8"), file_name="biothermo_pid_log.csv", mime="text/csv")
