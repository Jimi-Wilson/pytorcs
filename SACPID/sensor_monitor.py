"""
Sensor monitor: connects to TORCS and displays ALL raw sensor data in a
structured, live-updating console view while a basic autopilot keeps
the car on the track.  Logs every timestep to a CSV file that is saved
on Ctrl+C.

Usage:
    1. Launch TORCS:  torcs -nofuel -nodamage -nolaptime &
    2. Quick Race → add 'scr_server' → New Race
    3. Run:  python sensor_monitor.py
"""

import os
import sys
import time
import numpy as np
from datetime import datetime
from gym_torcs import TorcsEnv


def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')


def format_bar(value, width=20, vmin=-1.0, vmax=1.0):
    """Create a simple ASCII bar for a normalized value."""
    ratio = (value - vmin) / (vmax - vmin + 1e-9)
    ratio = max(0.0, min(1.0, ratio))
    filled = int(ratio * width)
    return '█' * filled + '░' * (width - filled)


def build_sensor_dashboard(raw, step, log_file, ema_ax=0.0, ema_ay=0.0, ema_az=0.0, prev_trackPos=0.0):
    """Build the dashboard as a string and return it."""
    lines = []
    a = lines.append

    a("=" * 80)
    a(f"  TORCS SENSOR MONITOR — Step {step}  |  Logging to: {log_file}")
    a("=" * 80)

    # ---- Car Kinematics ----
    speedX = raw.get('speedX', 0.0)
    speedY = raw.get('speedY', 0.0)
    speedZ = raw.get('speedZ', 0.0)
    rpm    = raw.get('rpm', 0.0)
    gear   = raw.get('gear', 0)

    a(f"")
    a(f"  ┌─── CAR KINEMATICS ───────────────────────────────────────┐")
    a(f"  │  Speed X (fwd):   {speedX:+8.2f} km/h   {format_bar(speedX, 20, 0, 300)}  │")
    a(f"  │  Speed Y (lat):   {speedY:+8.2f} km/h   {format_bar(speedY, 20, -50, 50)}  │")
    a(f"  │  Speed Z (vert):  {speedZ:+8.2f} km/h   {format_bar(speedZ, 20, -20, 20)}  │")
    a(f"  │  RPM:             {rpm:8.0f}           {format_bar(rpm, 20, 0, 10000)}  │")
    a(f"  │  Gear:            {gear:8.0f}                                    │")
    a(f"  └─────────────────────────────────────────────────────────┘")

    # ---- Track Position ----
    angle    = raw.get('angle', 0.0)
    trackPos = raw.get('trackPos', 0.0)
    z_height = raw.get('z', 0.0)
    damage   = raw.get('damage', 0.0)

    a(f"")
    a(f"  ┌─── TRACK POSITION ──────────────────────────────────────┐")
    a(f"  │  Angle to track:  {angle:+8.4f} rad   {format_bar(angle, 20, -3.14, 3.14)}  │")
    a(f"  │  Track position:  {trackPos:+8.4f}       {format_bar(trackPos, 20, -1.5, 1.5)}  │")
    a(f"  │  Height (z):      {z_height:+8.4f} m                                  │")
    a(f"  │  Damage:          {damage:8.0f}                                    │")
    a(f"  └─────────────────────────────────────────────────────────┘")

    # ---- Track Edge Sensors (19 rays) ----
    track = raw.get('track', [0]*19)
    a(f"")
    a(f"  ┌─── TRACK EDGE SENSORS (19 rays, max 200m) ────────────┐")
    angles_deg = [-90, -80, -70, -60, -50, -40, -30, -20, -10, 0,
                   10,  20,  30,  40,  50,  60,  70,  80,  90]
    for i in range(19):
        dist = track[i]
        bar = format_bar(dist, 15, 0, 200)
        label = f"{angles_deg[i]:+4d}°"
        marker = " ◄── CLOSEST" if dist == min(track) else ""
        a(f"  │  {label}: {dist:6.1f}m  {bar}{marker:>20s}  │")
    a(f"  └─────────────────────────────────────────────────────────┘")

    # ---- Wheel Spin Velocities ----
    wsv = raw.get('wheelSpinVel', [0]*4)
    a(f"")
    a(f"  ┌─── WHEEL SPIN VELOCITIES (rad/s) ─────────────────────┐")
    a(f"  │  Front Left:   {wsv[0]:+8.1f}    Front Right:  {wsv[1]:+8.1f}      │")
    a(f"  │  Rear  Left:   {wsv[2]:+8.1f}    Rear  Right:  {wsv[3]:+8.1f}      │")
    a(f"  └─────────────────────────────────────────────────────────┘")

    # ---- Derived Values (what the NN will see) ----
    slip_angle = np.arctan2(speedY, speedX + 1e-8) / (np.pi / 2.0)
    curve = (track[18] - track[0]) / 200.0

    # Longitudinal tire slip (rear-wheel drive assumption)
    WHEEL_RADIUS = 0.33       # metres — typical TORCS rear wheel
    avg_rear_rads = (wsv[2] + wsv[3]) / 2.0
    wheel_speed_ms = avg_rear_rads * WHEEL_RADIUS
    car_speed_ms = speedX / 3.6
    # Low-speed deadband: below 5 km/h the wheel signals are noisy junk
    if abs(speedX) < 5.0:
        tire_slip = 0.0
    else:
        denom = max(abs(wheel_speed_ms), abs(car_speed_ms), 1.0)
        tire_slip = float(np.clip((wheel_speed_ms - car_speed_ms) / denom, -1.0, 1.0))

    # Time-to-boundary panic — uses d(trackPos)/dt to capture ALL drift
    # (heading angle, curvature, lateral velocity) not just speedY.
    trackPos_rate = trackPos - prev_trackPos
    TRACK_LIMIT = 1.05  # must match the wrapper's track_limit
    dist_to_edge = max(TRACK_LIMIT - abs(trackPos), 0.0)
    moving_outward = trackPos * trackPos_rate > 0   # drifting toward ±limit

    if dist_to_edge <= 0.0:
        ttb = 0.0
    elif moving_outward and abs(trackPos_rate) > 0.001:
        steps_to_edge = dist_to_edge / (abs(trackPos_rate) + 1e-8)
        # Panic TTB: 0.0 at edge, 1.0 when ≥100 steps (~5 s) away
        ttb = float(np.clip(steps_to_edge / 100.0, 0.0, 1.0))
    else:
        # Moving inward or stationary → safe
        ttb = 1.0

    a(f"")
    a(f"  ┌─── DERIVED FEATURES (what the NN sees) ────────────────┐")
    a(f"  │  Chassis Slip Angle: {slip_angle:+.4f}                              │")
    a(f"  │  Tire Slip Delta:    {tire_slip:+.4f}  (>0 burnout, <0 locked) │")
    a(f"  │  Track Curvature:    {curve:+.4f}  (+ = right, - = left)       │")
    a(f"  │  Panic TTB:          {ttb:+.4f}  (1=safe, 0=at edge)       │")
    a(f"  │  G-Force X (lon):    {ema_ax:+.4f}  (EMA accel)                │")
    a(f"  │  G-Force Y (lat):    {ema_ay:+.4f}  (EMA accel)                │")
    a(f"  │  G-Force Z (vert):   {ema_az:+.4f}  (suspension unload)        │")
    a(f"  └─────────────────────────────────────────────────────────┘")

    a(f"")
    a(f"  Press Ctrl+C to stop and save log.")

    return "\n".join(lines)





def main():
    # Generate a timestamped filename
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'sensor_log_{ts}.txt'
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), log_file)

    print("Connecting to TORCS on port 3101...")
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    obs, info = env.reset()

    print(f"Connected! Logging to {log_file}")

    log_frames = []  # accumulate dashboard text frames

    # EMA state tracking for G-force derived features
    alpha = 0.2
    ema_accelX  = 0.0
    ema_accelY  = 0.0
    ema_accelZ  = 0.0

    # Seed prev-state trackers from the actual spawn state so the first
    # step's deltas (G-force, TTB) are zero instead of a false spike.
    raw_init = env.client.S.d
    prev_speedX   = raw_init.get('speedX', 0.0)
    prev_speedY   = raw_init.get('speedY', 0.0)
    prev_speedZ   = raw_init.get('speedZ', 0.0)
    prev_trackPos = raw_init.get('trackPos', 0.0)

    step = 0
    try:
        while True:
            raw = env.client.S.d

            # Simple proportional autopilot to stay on track
            trackPos = raw.get('trackPos', 0.0)
            angle    = raw.get('angle', 0.0)
            speedX   = raw.get('speedX', 0.0)
            speedY   = raw.get('speedY', 0.0)
            speedZ   = raw.get('speedZ', 0.0)

            # EMA G-force calculations
            raw_ax = speedX - prev_speedX
            ema_accelX = (1.0 - alpha) * ema_accelX + alpha * raw_ax
            prev_speedX = speedX

            raw_ay = speedY - prev_speedY
            ema_accelY = (1.0 - alpha) * ema_accelY + alpha * raw_ay
            prev_speedY = speedY

            raw_az = speedZ - prev_speedZ
            ema_accelZ = (1.0 - alpha) * ema_accelZ + alpha * raw_az
            prev_speedZ = speedZ

            norm_ax = float(np.clip(ema_accelX / 10.0, -1.0, 1.0))
            norm_ay = float(np.clip(ema_accelY / 10.0, -1.0, 1.0))
            norm_az = float(np.clip(ema_accelZ / 5.0, -1.0, 1.0))

            # Straight-line autopilot (no steering, gentle throttle)
            steer = 0.0
            accel = 0.5 if speedX < 80 else 0.1

            # Build dashboard text for every step
            frame = build_sensor_dashboard(raw, step, log_file,
                                           ema_ax=norm_ax, ema_ay=norm_ay, ema_az=norm_az,
                                           prev_trackPos=prev_trackPos)
            prev_trackPos = trackPos
            log_frames.append(frame)

            # Print to console every 5 steps
            if step % 5 == 0:
                clear_screen()
                print(frame)

            action = np.array([steer, accel])
            obs, reward, done, truncated, info = env.step(action)

            if done:
                obs, info = env.reset()

            step += 1

    except KeyboardInterrupt:
        # Save all frames to .txt
        print(f"\n\n  Saving {len(log_frames)} frames to {log_path} ...")
        with open(log_path, 'w') as f:
            f.write(('\n\n').join(log_frames))
        print(f"  ✓ Saved! ({len(log_frames)} timesteps logged)")
        env.end()


if __name__ == '__main__':
    main()
