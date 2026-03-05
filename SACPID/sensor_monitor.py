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


def build_sensor_dashboard(raw, step, log_file):
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

    # ---- Focus Sensors ----
    focus = raw.get('focus', [0]*5)
    a(f"")
    a(f"  ┌─── FOCUS SENSORS (5 rays, max 200m) ──────────────────┐")
    focus_str = "  ".join(f"{f:6.1f}m" for f in focus)
    a(f"  │  {focus_str}      │")
    a(f"  └─────────────────────────────────────────────────────────┘")

    # ---- Derived Values (what the NN will see) ----
    slip_angle = np.arctan2(speedY, speedX + 1e-8) / (np.pi / 2.0)
    avg_rear = (wsv[2] + wsv[3]) / 2.0
    curve = (track[18] - track[0]) / 200.0

    a(f"")
    a(f"  ┌─── DERIVED FEATURES (what the NN sees) ────────────────┐")
    a(f"  │  Chassis Slip Angle: {slip_angle:+.4f}                              │")
    a(f"  │  Rear Wheel Avg:     {avg_rear:+.1f} rad/s                          │")
    a(f"  │  Track Curvature:    {curve:+.4f}  (+ = right, - = left)       │")
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

    step = 0
    try:
        while True:
            raw = env.client.S.d

            # Simple proportional autopilot to stay on track
            trackPos = raw.get('trackPos', 0.0)
            angle    = raw.get('angle', 0.0)
            speedX   = raw.get('speedX', 0.0)

            # Straight-line autopilot (no steering, gentle throttle)
            steer = 0.0
            accel = 0.5 if speedX < 80 else 0.1

            # Build dashboard text for every step
            frame = build_sensor_dashboard(raw, step, log_file)
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
