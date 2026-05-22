import socket
import time
from dataclasses import dataclass


class TorcsClient:
    def __init__(self, port: int = 3001, host = "localhost", driver_id="SCR", viewing_angles="-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"):
        self.sensors = Sensor()
        self.actions = Action()

        self.host = host
        self.port = port
        self.driver_id = driver_id
        self.viewing_angles = viewing_angles
        self.sock = None

        self.connect()


    def connect(self, max_retries=500):
        init_msg = f"{self.driver_id}(init {self.viewing_angles})"

        if self.sock is not None:
            try:
                self.sock.close()
            except Exception:
                pass

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(1)

        for attempt in range(max_retries):
            self.sock.sendto(init_msg.encode(), (self.host, self.port))

            try:
                response, _ = self.sock.recvfrom(1024)
                response_str = response.decode()

                if "***identified***" in response_str:
                    return
            except socket.timeout:
                print(f"Waiting for TORCS response from {self.port}... ({attempt+1}/{max_retries})")
            except ConnectionResetError:
                print(f"TORCS server not listening on {self.host}:{self.port} Please start TORCS.")
                time.sleep(1)

        print(f"Failed to connect to TORCS after {max_retries} attempts.")

    def get_sensors(self):
        last_valid = None

        self.sock.setblocking(False)
        try:
            while True:
                try:
                    response, _ = self.sock.recvfrom(65536)
                    response_str = response.decode()

                    if "***shutdown***" in response_str or "***restart***" in response_str:
                        return None

                    if response_str and "***identified***" not in response_str:
                        last_valid = response_str

                except BlockingIOError:
                    break
                except ConnectionResetError:
                    print("Server closed the connection.")
                    return None
        finally:
            self.sock.setblocking(True)
            self.sock.settimeout(1)

        if last_valid is not None:
            return last_valid

        try:
            response, _ = self.sock.recvfrom(65536)
            response_str = response.decode()

            if "***shutdown***" in response_str or "***restart***" in response_str:
                return None

            if not response_str or "***identified***" in response_str:
                return None

            return response_str

        except socket.timeout:
            print("No data received from server recently...")
            return None
        except ConnectionResetError:
            print("Server closed the connection.")
            return None

    def send_action(self):
        try:
            self.sock.sendto(self.actions.to_msg().encode(), (self.host, self.port))
        except socket.error as e:
            print(f"Failed to send action to TORCS: {e}")


class Action:
    def __init__(self):
        self.accel = 0.0
        self.brake = 0.0
        self.clutch = 0.0
        self.gear = 0
        self.steering = 0.0
        self.focus = 0
        self.meta = 0

    def to_msg(self):
        accel = max(0.0, min(1.0, self.accel))
        brake = max(0.0, min(1.0, self.brake))
        steer = max(-1.0, min(1.0, self.steering))
        return f"(accel {accel})(brake {brake})(gear {self.gear})(steer {steer})(clutch {self.clutch})(focus {self.focus})(meta {self.meta})"

class Sensor:
    def __init__(self):
        self.angle = 0.0
        self.curLapTime = 0.0
        self.damage = 0.0
        self.distFromStart = 0.0
        self.distRaced = 0.0
        self.focus = [0.0] * 5
        self.fuel = 0.0
        self.gear = 0
        self.lastLapTime = 0.0
        self.opponents = [0.0] * 36
        self.racePos = 0
        self.rpm = 0.0
        self.speedX = 0.0
        self.speedY = 0.0
        self.speedZ = 0.0
        self.track = [0.0] * 19
        self.trackPos = 0.0
        self.wheelSpinVel = [0.0] * 4
        self.z = 0.0


    def update(self, raw_string: str):
        clean_string = raw_string.strip()
        sensors = clean_string.strip('()').split(")(")

        for sensor in sensors:
            parts = sensor.split()
            if not parts:
                continue

            name = parts[0]

            try:
                numeric_values = [float(v.replace(')', '')) for v in parts[1:]]
            except (ValueError, IndexError):
                continue

            if hasattr(self, name):
                if len(numeric_values) == 1:
                    val = numeric_values[0]
                    setattr(self, name, int(val) if name == "gear" else val)
                else:
                    setattr(self, name, numeric_values)

@dataclass
class TorcsState:
    sensors: Sensor
    actions: Action