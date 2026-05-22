from gymnasium import spaces
import numpy as np
# from os import path
import snakeoil3_gym as snakeoil3
import copy
import collections as col
import os
import time
import json


# =============================================================================
# ARCHIVED: OW1 / advanced auto-shifter (commented out — do not delete).
# To re-enable: uncomment this block, add ``import math``, uncomment the matching
# sections in ``TorcsEnv.__init__``, ``_debounced_gear_command``, ``reset``, and
# ``step`` marked ``# --- OW1 SHIFTER ---`` below, and comment the active RPM block.
# =============================================================================
# import math
#
# # --- Automatic gear selection (SCR has no gear="auto"; 0 is neutral.) ---
# # TORCS human ``drive_at`` (``src/drivers/human/human.cpp``) + OW1 XML ratios.
# # Tuned for short-shift / traction; knobs: TORCS_OW1_DOWNHYST_MS, TORCS_GEAR_DEBOUNCE_STEPS, etc.
# _CAR1_OW1_AUTO = {
#     "redline_rpm": 18700.0,
#     "rear_wheel_radius_m": 0.315,
#     "final_drive": 4.5,
#     "gear_ratios": (3.9, 2.9, 2.3, 1.87, 1.68, 1.54),
# }
# _CAR1_OW1_OVERALL = tuple(r * _CAR1_OW1_AUTO["final_drive"] for r in _CAR1_OW1_AUTO["gear_ratios"])
# _TWO_PI_OVER_60 = (2.0 * math.pi) / 60.0
# _OW1_UPSHIFT_CUSHIONS = (0.18, 0.26, 0.38, 0.48, 0.54, 0.58)
#
#
# def _torcs_speed_scalar_m_s(state: dict) -> float:
#     try:
#         sx = float(state.get("speedX", 0.0))
#         sy = float(state.get("speedY", 0.0))
#         sz = float(state.get("speedZ", 0.0))
#         return float(math.sqrt(sx * sx + sy * sy + sz * sz)) / 3.6
#     except (TypeError, ValueError):
#         return 0.0
#
#
# def _ow1_rear_spin_trim(state: dict | None) -> float:
#     if state is None:
#         return 1.0
#     try:
#         w = state.get("wheelSpinVel")
#         if not isinstance(w, (list, tuple)) or len(w) < 4:
#             return 1.0
#         rear = (float(w[2]) + float(w[3])) * 0.5
#         front = (float(w[0]) + float(w[1])) * 0.5
#         excess = rear - front
#         if excess > 12.0:
#             return 0.84
#         if excess > 8.0:
#             return 0.90
#         if excess > 5.0:
#             return 0.94
#     except (TypeError, ValueError):
#         pass
#     return 1.0
#
#
# def auto_gear_human_torcs_car1_ow1(
#     current_gear: int, speed_m_s: float, state: dict | None = None
# ) -> int:
#     try:
#         redline_frac = float(os.environ.get("TORCS_OW1_SHIFT_REDLINE_FRAC", "0.72"))
#     except ValueError:
#         redline_frac = 0.72
#     redline_frac = float(np.clip(redline_frac, 0.45, 1.0))
#     redline = _CAR1_OW1_AUTO["redline_rpm"] * redline_frac
#     rw = _CAR1_OW1_AUTO["rear_wheel_radius_m"]
#     ratios = _CAR1_OW1_OVERALL
#     max_g = len(ratios)
#     cushions = _OW1_UPSHIFT_CUSHIONS
#     try:
#         down_m = float(os.environ.get("TORCS_OW1_DOWNHYST_MS", "14.0"))
#     except ValueError:
#         down_m = 14.0
#     spin_trim = _ow1_rear_spin_trim(state)
#     g = int(current_gear)
#     if g < 1:
#         return 1
#     if g > max_g:
#         g = max_g
#
#     def upshift_threshold(for_gear: int, *, apply_spin_trim: bool) -> float:
#         t = (
#             redline * _TWO_PI_OVER_60 * rw * cushions[for_gear - 1] / ratios[for_gear - 1]
#         )
#         if apply_spin_trim:
#             t *= spin_trim
#         return t
#
#     if g < max_g and speed_m_s > upshift_threshold(g, apply_spin_trim=True):
#         return g + 1
#     if g > 1 and speed_m_s < upshift_threshold(g - 1, apply_spin_trim=False) - down_m:
#         return g - 1
#     return g
#
#
# def auto_gear_legacy_rpm_speed(current_gear: int, rpm: float, speed_x: float) -> int:
#     gear = int(current_gear)
#     if gear < 1:
#         gear = 1
#     elif gear < 6 and rpm > 8000:
#         gear += 1
#     elif gear > 1 and (rpm < 4000 or (gear >= 4 and speed_x < 20.0)):
#         gear -= 1
#     return gear
# =============================================================================


class TorcsEnv:
    """Low-level SCR client. ``terminate_on_off_track`` controls whether ``track.min() < 0``
    ends the episode (SCR ``meta``). Disabling it keeps Python driving after excursions;
    TORCS physics/UI may still behave oddly off-track.

    On the **first** ``reset()``, you can optionally send an extra ``meta`` round-trip after the
    initial handshake (set ``TORCS_FIRST_RESET_META_SYNC=1``) so SCR race state matches later
    episode resets. **Default is off:** many SCR/TORCS builds loop on ``***restart***`` during
    that sync; leave disabled unless you know you need it.

    Automatic gear uses the **original RPM shifter** (8000 / 4000). An archived **OW1** shifter
    lives in comments at the top of this file and in ``# --- OW1 SHIFTER (archived) ---`` sections."""

    terminal_judge_start = 200  # Speed limit is applied after this step
    termination_limit_progress = 10  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 50

    initial_reset = True


    def _debug_log(self, hypothesis_id: str, location: str, message: str, data: dict) -> None:
        # #region agent log
        try:
            payload = {
                "sessionId": "2e77ba",
                "runId": "scr-precheck",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(time.time() * 1000),
            }
            candidates = [
                os.path.join(os.path.dirname(__file__), ".cursor", "debug-2e77ba.log"),
                os.path.join(os.getcwd(), ".cursor", "debug-2e77ba.log"),
            ]
            line = json.dumps(payload, separators=(",", ":")) + "\n"
            for path in candidates:
                try:
                    parent = os.path.dirname(path)
                    if parent:
                        os.makedirs(parent, exist_ok=True)
                    with open(path, "a", encoding="utf-8") as fp:
                        fp.write(line)
                    break
                except Exception:
                    continue
        except Exception:
            pass
        # #endregion


    def __init__(
        self,
        vision=False,
        throttle=False,
        gear_change=False,
        port=3001,
        terminate_on_off_track=None,
    ):
       #print("Init")
        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change
        self.port = int(port)
        # Default True (RL episodes). If False, leaving the track does not set SCR meta / done
        # (hotlap / eval). Override via arg or env TORCS_OFFTRACK_TERMINATION=0|false|no.
        if terminate_on_off_track is not None:
            self.terminate_on_off_track = bool(terminate_on_off_track)
        else:
            raw = os.environ.get("TORCS_OFFTRACK_TERMINATION", "").strip().lower()
            if raw in ("0", "false", "no"):
                self.terminate_on_off_track = False
            elif raw in ("1", "true", "yes"):
                self.terminate_on_off_track = True
            else:
                self.terminate_on_off_track = True

        self.initial_run = True

        # --- OW1 SHIFTER (archived) — uncomment together with module + step + reset + method below:
        # self._auto_shift_mode = os.environ.get("TORCS_AUTO_SHIFT", "human_car1_ow1").strip().lower()
        # try:
        #     self._gear_cmd_debounce_steps = max(1, int(os.environ.get("TORCS_GEAR_DEBOUNCE_STEPS", "18")))
        # except ValueError:
        #     self._gear_cmd_debounce_steps = 18
        # self._gear_cmd_smoothed: int | None = None
        # self._gear_cmd_pending: int | None = None
        # self._gear_cmd_streak = 0
        # self._last_sent_gear_cmd: int | None = None
        # try:
        #     self._shift_clutch_max = max(0, int(os.environ.get("TORCS_OW1_SHIFT_CLUTCH_STEPS", "5")))
        # except ValueError:
        #     self._shift_clutch_max = 5
        # self._shift_clutch_n = 0

        ##print("launch torcs")
        # os.system('pkill torcs')
        # time.sleep(0.5)
        # if self.vision is True:
        #     os.system('torcs -nofuel -nodamage -nolaptime  -vision &')
        # else:
        #     os.system('torcs  -nofuel -nodamage -nolaptime &')
        # time.sleep(0.5)
        # Launch race manually in TORCS (Corkscrew + scr_server1 selected).
        # time.sleep(0.5)

        """
        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3001, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        obs = client.S.d  # Get the current full-observation from torcs
        """
        if throttle is False:
            # Steering only.
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        elif gear_change is True:
            # Full manual controls: [steer, accel, brake, gear_cmd]
            # gear_cmd is continuous here and snapped to valid gears downstream.
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            )
        else:
            # Explicit 3-pedal control when throttle mode is enabled:
            # [steer, accel, brake] where steer in [-1,1], pedals in [0,1].
            self.action_space = spaces.Box(
                low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            )

        if vision is False:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf])
            self.observation_space = spaces.Box(low=low, high=high)
        else:
            high = np.array([1., np.inf, np.inf, np.inf, 1., np.inf, 1., np.inf, 255])
            low = np.array([0., -np.inf, -np.inf, -np.inf, 0., -np.inf, 0., -np.inf, 0])
            self.observation_space = spaces.Box(low=low, high=high)

    # --- OW1 SHIFTER (archived) ---
    # def _debounced_gear_command(self, ideal_gear: int) -> int:
    #     if self._gear_cmd_smoothed is None:
    #         self._gear_cmd_smoothed = int(ideal_gear)
    #         self._gear_cmd_pending = None
    #         self._gear_cmd_streak = 0
    #         return self._gear_cmd_smoothed
    #     if int(ideal_gear) == self._gear_cmd_smoothed:
    #         self._gear_cmd_pending = None
    #         self._gear_cmd_streak = 0
    #         return self._gear_cmd_smoothed
    #     if self._gear_cmd_pending != int(ideal_gear):
    #         self._gear_cmd_pending = int(ideal_gear)
    #         self._gear_cmd_streak = 1
    #     else:
    #         self._gear_cmd_streak += 1
    #     if self._gear_cmd_streak >= self._gear_cmd_debounce_steps:
    #         self._gear_cmd_smoothed = int(ideal_gear)
    #         self._gear_cmd_pending = None
    #         self._gear_cmd_streak = 0
    #     return self._gear_cmd_smoothed

    def step(self, u):
       #print("Step")
        # convert thisAction to the actual torcs actionstr
        client = self.client

        this_action = self.agent_to_torcs(u)

        # Apply Action
        action_torcs = client.R.d

        # Steering
        action_torcs['steer'] = this_action['steer']  # in [-1, 1]

        #  Simple Autnmatic Throttle Control by Snakeoil
        if self.throttle is False:
            target_speed = self.default_speed
            if client.S.d['speedX'] < target_speed - (client.R.d['steer']*50):
                client.R.d['accel'] += .01
            else:
                client.R.d['accel'] -= .01

            if client.R.d['accel'] > 0.2:
                client.R.d['accel'] = 0.2

            if client.S.d['speedX'] < 10:
                client.R.d['accel'] += 1/(client.S.d['speedX']+.1)

            # Traction Control System
            if ((client.S.d['wheelSpinVel'][2]+client.S.d['wheelSpinVel'][3]) -
               (client.S.d['wheelSpinVel'][0]+client.S.d['wheelSpinVel'][1]) > 5):
                action_torcs['accel'] -= .2
        else:
            action_torcs['accel'] = this_action['accel']
            action_torcs['brake'] = this_action['brake']

        #  Automatic Gear Change (RPM-based, like a real racer) — original simple shifter.
        if self.gear_change is True:
            action_torcs['gear'] = this_action['gear']
        else:
            rpm = client.S.d['rpm']
            gear = client.S.d['gear']

            # Upshift near redline, downshift when RPM drops too low
            if gear < 1:
                gear = 1
            elif gear < 6 and rpm > 8000:
                gear += 1
            elif gear > 1 and rpm < 4000:
                gear -= 1

            action_torcs['gear'] = gear

        # --- OW1 SHIFTER (archived) — replace the ``else`` block above when re-enabling:
        # if self.gear_change is True:
        #     action_torcs["gear"] = this_action["gear"]
        # else:
        #     gear = int(client.S.d["gear"])
        #     if self._auto_shift_mode in ("legacy", "rpm", "torcs_single_instance_training"):
        #         rpm = float(client.S.d["rpm"])
        #         try:
        #             speed_x = float(client.S.d.get("speedX", 0.0))
        #         except (TypeError, ValueError):
        #             speed_x = 0.0
        #         action_torcs["gear"] = auto_gear_legacy_rpm_speed(gear, rpm, speed_x)
        #     else:
        #         speed = _torcs_speed_scalar_m_s(client.S.d)
        #         ideal = auto_gear_human_torcs_car1_ow1(gear, speed, client.S.d)
        #         cmd = self._debounced_gear_command(ideal)
        #         prev = self._last_sent_gear_cmd
        #         if prev is not None and cmd != prev and self._shift_clutch_max > 0:
        #             self._shift_clutch_n = max(self._shift_clutch_n, self._shift_clutch_max)
        #         self._last_sent_gear_cmd = cmd
        #         action_torcs["gear"] = cmd
        #         if self._shift_clutch_n > 0:
        #             action_torcs["clutch"] = 0.26
        #             self._shift_clutch_n -= 1
        #         else:
        #             action_torcs["clutch"] = 0.0

        # Save the privious full-obs from torcs for the reward calculation
        obs_pre = copy.deepcopy(client.S.d)

        # One-Step Dynamics Update #################################
        # Apply the Agent's action into torcs
        client.respond_to_server()
        # Get the response of TORCS
        client.get_servers_input()

        # TORCS SCR sent ***restart***/***shutdown*** or hit server limits: snakeoil closed the UDP socket.
        if client.so is None:
            info = {"termination_reason": "snakeoil_disconnected_recoverable"}
            if os.environ.get("TORCS_VERBOSE_TERMINATION", "0") == "1":
                print(
                    "--> episode terminated: SCR socket closed (TORCS restart/shutdown or server limit); "
                    "reset will open a new client."
                )
            obs = client.S.d
            self.observation = self.make_observation(obs)
            self.initial_run = False
            self.time_step += 1
            return self.get_obs(), 0.0, True, False, info

        # Get the current full-observation from torcs
        obs = client.S.d

        # Make an observation from a raw observation vector from TORCS
        self.observation = self.make_observation(obs)

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        sp = np.array(obs['speedX'])
        progress = sp*np.cos(obs['angle'])
        reward = progress

        # collision detection
        if obs['damage'] - obs_pre['damage'] > 0:
            reward = -1

        # Termination judgement #########################
        # When any of these happen we must set meta=True so step() returns done=True
        # and the trainer (OmniSafe) calls reset() to start a new episode.
        info = {}
        verbose = os.environ.get('TORCS_VERBOSE_TERMINATION', '0') == '1'

        # Off-track: optional terminal (disabled for native hotlap when terminate_on_off_track is False).
        if self.terminate_on_off_track and track.min() < 0:
            if verbose:
                print(f"--> episode terminated: Car has run off the track (track.min: {track.min():.2f})")
            reward = -1
            client.R.d['meta'] = True
            info['termination_reason'] = 'off_track'

        # Disabled: too_slow termination cut off RL episodes early when progress stayed low after
        # terminal_judge_start steps (e.g. exploration / bad policy). Re-enable by uncommenting below.
        # if self.terminal_judge_start < self.time_step:  # Episode terminates if the progress of agent is small
        #     if progress < self.termination_limit_progress and 'termination_reason' not in info:
        #         if verbose:
        #             print(f"--> episode terminated: Car is driving too slow (progress: {progress:.2f})")
        #         client.R.d['meta'] = True
        #         info['termination_reason'] = 'too_slow'

        if np.cos(obs['angle']) < 0:  # Episode is terminated if the agent runs backward
            if 'termination_reason' not in info:
                if verbose:
                    print(f"--> episode terminated: Car is running backwards")
                client.R.d['meta'] = True
                info['termination_reason'] = 'backwards'

        if obs['damage'] - obs_pre['damage'] > 0 and 'termination_reason' not in info:
            info['termination_reason'] = 'damage'

        if client.R.d['meta']:  # truthy: 1 or True after clip — send reset / post-meta packet
            self.initial_run = False
            client.respond_to_server()

        self.time_step += 1

        return self.get_obs(), reward, client.R.d['meta'], False, info

    def reset(self, relaunch=False):
        #print("Reset")
        # #region agent log
        self._debug_log(
            "H8",
            "gym_torcs.py:TorcsEnv.reset:entry",
            "TorcsEnv reset begin",
            {
                "port": int(self.port),
                "relaunch": bool(relaunch),
                "initial_reset": bool(self.initial_reset),
            },
        )
        # #endregion

        self.time_step = 0

        # --- OW1 SHIFTER (archived) — uncomment with OW1 step/__init__/module:
        # self._gear_cmd_smoothed = None
        # self._gear_cmd_pending = None
        # self._gear_cmd_streak = 0
        # self._last_sent_gear_cmd = None
        # self._shift_clutch_n = 0

        first_connect_reset = self.initial_reset is True

        if self.initial_reset is not True:
            self.client.R.d['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        # #region agent log
        self._debug_log(
            "H8",
            "gym_torcs.py:TorcsEnv.reset:client_create_begin",
            "Creating snakeoil client",
            {
                "port": int(self.port),
            },
        )
        # #endregion
        try:
            self.client = snakeoil3.Client(
                p=self.port,
                vision=self.vision,
                reconnect_timeout_s=os.environ.get("TORCS_RECONNECT_TIMEOUT_S", "10.0"),
                enforce_reconnect_timeout=not first_connect_reset,
            )  # Open new UDP in vtorcs
        except snakeoil3.TorcsReconnectTimeout as exc:
            raise RuntimeError(f"race_finished_reconnect_timeout: {exc}") from exc
        # snakeoil uses attribute maxSteps (not MAX_STEPS); keep high to avoid misleading shutdown prints / limits
        self.client.maxSteps = 10**12
        # #region agent log
        self._debug_log(
            "H8",
            "gym_torcs.py:TorcsEnv.reset:client_create_done",
            "Created snakeoil client",
            {
                "port": int(self.port),
            },
        )
        # #endregion

        client = self.client
        # #region agent log
        t0 = time.perf_counter()
        self._debug_log(
            "H8",
            "gym_torcs.py:TorcsEnv.reset:get_servers_input_begin",
            "Calling get_servers_input",
            {
                "port": int(self.port),
            },
        )
        # #endregion
        client.get_servers_input()  # Get the initial input from torcs
        # #region agent log
        self._debug_log(
            "H8",
            "gym_torcs.py:TorcsEnv.reset:get_servers_input_done",
            "get_servers_input returned",
            {
                "port": int(self.port),
                "elapsed_ms": int((time.perf_counter() - t0) * 1000),
            },
        )
        # #endregion

        obs = client.S.d  # Get the current full-observation from torcs
        # Log track name once if the server sends it (SCR state may include trackName/trackname)
        if getattr(self, '_logged_track', None) is None:
            self._logged_track = True
            for key in ('trackName', 'trackname', 'track_name'):
                if key in obs and isinstance(obs[key], (str, bytes)):
                    name = obs[key].decode() if isinstance(obs[key], bytes) else obs[key]
                    print(f"[gym_torcs] Track (from TORCS server state): {name}")
                    break
            else:
                # Optionally log state keys so user can see what the server sends
                if os.environ.get('TORCS_LOG_STATE_KEYS'):
                    print(f"[gym_torcs] Server state keys (first connection): {list(obs.keys())}")
        self.observation = self.make_observation(obs)

        self.last_u = None

        # Optional: extra meta round-trip so first session matches post-episode reset path.
        # Default OFF: several vtorcs/SCR stacks answer with repeated ***restart*** here and
        # never return telemetry until this block is skipped (set TORCS_FIRST_RESET_META_SYNC=1 to enable).
        _raw_sync = os.environ.get("TORCS_FIRST_RESET_META_SYNC", "0").strip().lower()
        _sync = _raw_sync in ("1", "true", "yes")
        if first_connect_reset and _sync and getattr(client, "so", None) is not None:
            try:
                client.R.d["meta"] = 1
                client.respond_to_server()
                client.get_servers_input()
                obs = client.S.d
                self.observation = self.make_observation(obs)
            except Exception:
                pass

        self.initial_reset = False
        # #region agent log
        self._debug_log(
            "H8",
            "gym_torcs.py:TorcsEnv.reset:success",
            "TorcsEnv reset success",
            {
                "port": int(self.port),
            },
        )
        # #endregion
        return self.get_obs(), {}

    def end(self):
        # os.system('pkill torcs')
        pass

    def get_obs(self):
        return self.observation

    def reset_torcs(self):
       #print("relaunch torcs disabled")
        pass

    def agent_to_torcs(self, u):
        if self.throttle is False and len(u) < 1:
            raise ValueError(f"Expected at least 1 action element [steer], got {len(u)}")
        if self.throttle is True and self.gear_change is False and len(u) < 3:
            raise ValueError(
                f"Expected 3 action elements [steer, accel, brake], got {len(u)}"
            )
        if self.throttle is True and self.gear_change is True and len(u) < 4:
            raise ValueError(
                f"Expected 4 action elements [steer, accel, brake, gear], got {len(u)}"
            )

        torcs_action = {'steer': float(np.clip(u[0], -1.0, 1.0))}

        if self.throttle is True:  # throttle action is enabled
            torcs_action.update({
                'accel': float(np.clip(u[1], 0.0, 1.0)),
                'brake': float(np.clip(u[2], 0.0, 1.0)),
            })

        if self.gear_change is True: # gear change action is enabled
            torcs_action.update({'gear': int(np.rint(u[3]))})

        return torcs_action


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        rgb = []
        temp = []
        # convert size 64x64x3 = 12288 to 64x64=4096 2-D list 
        # with rgb values grouped together.
        # Format similar to the observation in openai gym
        for i in range(0,12286,3):
            temp.append(image_vec[i])
            temp.append(image_vec[i+1])
            temp.append(image_vec[i+2])
            rgb.append(temp)
            temp = []
        return np.array(rgb, dtype=np.uint8)

    def make_observation(self, raw_obs):
        if self.vision is False:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ',
                     'opponents',
                     'rpm',
                     'track',
                     'wheelSpinVel']
            Observation = col.namedtuple('Observation', names)
            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32))
        else:
            names = ['focus',
                     'speedX', 'speedY', 'speedZ',
                     'opponents',
                     'rpm',
                     'track',
                     'wheelSpinVel',
                     'img']
            Observation = col.namedtuple('Observation', names)

            # Get RGB from observation
            image_rgb = self.obs_vision_to_image_rgb(raw_obs[names[8]])

            return Observation(focus=np.array(raw_obs['focus'], dtype=np.float32)/200.,
                               speedX=np.array(raw_obs['speedX'], dtype=np.float32)/self.default_speed,
                               speedY=np.array(raw_obs['speedY'], dtype=np.float32)/self.default_speed,
                               speedZ=np.array(raw_obs['speedZ'], dtype=np.float32)/self.default_speed,
                               opponents=np.array(raw_obs['opponents'], dtype=np.float32)/200.,
                               rpm=np.array(raw_obs['rpm'], dtype=np.float32),
                               track=np.array(raw_obs['track'], dtype=np.float32)/200.,
                               wheelSpinVel=np.array(raw_obs['wheelSpinVel'], dtype=np.float32),
                               img=image_rgb)
