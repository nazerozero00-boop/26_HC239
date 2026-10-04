"""GUI용 UDP 측정 수신 + CSV 저장.

저장 CSV는 sample-level elapsed_time을 포함하므로 실측 fs를 다시 계산할 수 있다.
"""
from __future__ import annotations

import csv
import os
import threading
import time
from datetime import datetime
import numpy as np

from sensor_udp_io import HallSensorClient, parse_packet

HERE = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(HERE, "measurements")


def ensure_save_dir():
    os.makedirs(SAVE_DIR, exist_ok=True)


def save_measurement_csv(times, H, B, path=None):
    ensure_save_dir()
    if path is None:
        name = "bh_udp_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv"
        path = os.path.join(SAVE_DIR, name)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["elapsed_time", "H_Voltage", "B_Voltage"])
        for t, h, b in zip(times, H, B):
            w.writerow([f"{t:.6f}", f"{h:.6f}", f"{b:.6f}"])
    return path


class MeasurementReceiver:
    def __init__(self, on_status=None, on_done=None):
        self.client = None
        self.thread = None
        self.running = False
        self.times, self.H, self.B = [], [], []
        self.on_status = on_status
        self.on_done = on_done
        self._first_t = None

    def _status(self, msg):
        if self.on_status:
            self.on_status(msg)

    def start(self):
        if self.running:
            return
        self.times, self.H, self.B = [], [], []
        self._first_t = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        import socket
        try:
            self.client = HallSensorClient()
            self.client.connect()
            self.client.start_measurement()
            self._status("측정 중... UDP 데이터 수신")
        except Exception as e:
            self._status(f"연결 실패: {e}")
            self.running = False
            return

        while self.running:
            try:
                data, _ = self.client.sock.recvfrom(256)
            except socket.timeout:
                if not self.times:
                    self._status("데이터 없음 - Wi-Fi(BH_Sensing) 연결 확인")
                continue
            except OSError:
                break
            pair = parse_packet(data)
            if pair is None:
                continue
            now = time.monotonic()
            if self._first_t is None:
                self._first_t = now
            t = now - self._first_t
            h, b = pair
            self.times.append(t)
            self.H.append(h)
            self.B.append(b)
            if len(self.times) % 20 == 0:
                self._status(f"측정 중... {len(self.times)}점 수신")

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
        n = len(self.times)
        if n < 5:
            self._status(f"수신 데이터 부족 ({n}점) - 저장 안 함")
            if self.on_done:
                self.on_done(None, n)
            return None, n
        path = save_measurement_csv(
            np.asarray(self.times), np.asarray(self.H), np.asarray(self.B)
        )
        self._status(f"저장 완료: {os.path.basename(path)} ({n}점)")
        if self.on_done:
            self.on_done(path, n)
        return path, n
