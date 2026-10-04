"""Arduino UNO R4 WiFi 계측 데이터 UDP 수신.

Arduino AP: BH_Sensing / UDP 12345
RPi -> Arduino: '1' 시작, '0' 정지
Arduino -> broadcast: 'pureCurrent,pureHall'

두 값은 현재 영점 보정 전압이며 절대 H[A/m], B[T] 환산은 센서/코일 보정계수 확정 후 추가한다.
"""
from __future__ import annotations

import socket
import time
import numpy as np

ARDUINO_IP = "192.168.4.1"
UDP_PORT = 12345
TIMEOUT = 2.0
DEFAULT_MEASUREMENT_SECONDS = 15.0


def parse_packet(data):
    if isinstance(data, bytes):
        text = data.decode("utf-8", errors="ignore")
    else:
        text = str(data)
    parts = text.strip().split(",")
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


class HallSensorClient:
    def __init__(self, arduino_ip=ARDUINO_IP, port=UDP_PORT, timeout=TIMEOUT):
        self.arduino_ip = arduino_ip
        self.port = int(port)
        self.timeout = float(timeout)
        self.sock = None

    def connect(self):
        if self.sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind(("", self.port))
        sock.settimeout(self.timeout)
        self.sock = sock

    def _send_command(self, command):
        if self.sock is None:
            self.connect()
        if command not in ("0", "1"):
            raise ValueError("Arduino command는 '0' 또는 '1'이어야 합니다.")
        self.sock.sendto(command.encode("ascii"), (self.arduino_ip, self.port))

    def start_measurement(self):
        self._send_command("1")

    def stop_measurement(self):
        if self.sock is not None:
            try:
                self._send_command("0")
            except OSError:
                pass

    def close(self):
        if self.sock is not None:
            self.stop_measurement()
            self.sock.close()
            self.sock = None

    def read_measurement(self, duration=DEFAULT_MEASUREMENT_SECONDS):
        if self.sock is None:
            self.connect()
        times, h_values, b_values = [], [], []
        self.start_measurement()
        first_t = None
        try:
            while True:
                try:
                    data, _addr = self.sock.recvfrom(256)
                except socket.timeout:
                    if first_t is None:
                        raise TimeoutError(
                            "Arduino UDP 패킷을 받지 못했습니다. Raspberry Pi가 "
                            "BH_Sensing Wi-Fi에 연결됐는지 확인하세요."
                        )
                    break
                pair = parse_packet(data)
                if pair is None:
                    continue
                now = time.monotonic()
                if first_t is None:
                    first_t = now
                t = now - first_t
                h, b = pair
                times.append(t)
                h_values.append(h)
                b_values.append(b)
                if t >= float(duration):
                    break
        finally:
            self.stop_measurement()
        if len(times) < 20:
            raise ValueError(f"수신 데이터가 너무 적습니다: {len(times)}개")
        return np.asarray(times), np.asarray(h_values), np.asarray(b_values)
