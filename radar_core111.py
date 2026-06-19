"""
UAV Precision Landing -- Final Flight Core Mode
Hardware: TI IWR6843 mmWave Radar + Telemetry
Target Node: Field Laptop (Headless Ubuntu)
* 具備: UART防死鎖、Body_NED座標系修正、天地視角鏡像反轉 *
"""

import math
import struct
import time
import sys
import numpy as np
import serial
from sklearn.cluster import DBSCAN
from pymavlink import mavutil

# ============================================================
# 0. MAVLink / Telemetry Port Settings
# ============================================================
MAVLINK_PORT    = '/dev/ttyUSB0'  # 🌟 數傳電臺
MAVLINK_BAUD    = 115200           

# ============================================================
# 1. Centralised Parameter Settings
# ============================================================
CMD_PORT        = '/dev/ttyUSB1'  # 🌟 雷達 Command Port
CMD_BAUD        = 115200
DATA_PORT       = '/dev/ttyUSB2'  # 🌟 雷達 Data Port
DATA_BAUD       = 921600

# --- 測量與過濾範圍 ---
LIMIT_X     = 2.0       
LIMIT_Z     = 2.0       
MIN_Y_DIST  = 1.0      # 雷達盲區 (0.5m以下飛控自動盲降)
MAX_Y_DIST  = 3.0       # 最大追蹤高度放寬至 3m

# --- 戶外實飛專用過濾器 (防雜訊) ---
MIN_VELOCITY    = 0.07  # 濾除地面靜止與風吹草動雜訊
MAX_VELOCITY    = 2.0   

CLUSTER_EPS         = 0.2   
CLUSTER_MIN_SAMPLES = 2     # 確保物體夠大 (至少2個反射點)

# --- Alpha-Beta 追蹤器參數 (平滑軌跡) ---
MAX_LOST_FRAMES  = 15        
TRACKER_ALPHA    = 0.05     
TRACKER_BETA     = 0.01     
MAX_ALLOWED_JUMP = 0.25      

# --- 雷達封包解析常數 ---
MAGIC_WORD      = b'\x02\x01\x04\x03\x06\x05\x08\x07'
MAGIC_LEN       = 8
HEADER_LEN      = 40
TLV_HEADER_LEN  = 8
POINT_UNIT_LEN  = 16

# ============================================================
# 2. Core Class: TargetTracker
# ============================================================
class TargetTracker:
    def __init__(self):
        self.current_x = 0.0; self.current_y = 0.0; self.current_z = 0.0
        self.vx = 0.0; self.vy = 0.0; self.vz = 0.0
        self.last_time = 0.0
        self.lost_count = 0
        self.is_active = False
        self._initialized = False

    def update(self, new_x: float, new_y: float, new_z: float) -> None:
        now = time.monotonic()
        if not self.is_active:
            self.current_x, self.current_y, self.current_z = new_x, new_y, new_z
            self.vx, self.vy, self.vz = 0.0, 0.0, 0.0
            self.last_time, self._initialized, self.lost_count, self.is_active = now, True, 0, True
            return

        dt = max(min(now - self.last_time, 0.5), 0.005)
        pred_x = self.current_x + self.vx * dt
        pred_y = self.current_y + self.vy * dt
        pred_z = self.current_z + self.vz * dt

        dx, dy, dz = new_x - pred_x, new_y - pred_y, new_z - pred_z
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        if dist > MAX_ALLOWED_JUMP:
            if self.lost_count > 3:
                self.current_x, self.current_y, self.current_z = new_x, new_y, new_z
                self.vx, self.vy, self.vz = 0.0, 0.0, 0.0
                self.last_time, self.lost_count = now, 0
            else:
                self.predict_or_wait()
            return

        self.current_x = pred_x + TRACKER_ALPHA * dx
        self.current_y = pred_y + TRACKER_ALPHA * dy
        self.current_z = pred_z + TRACKER_ALPHA * dz

        beta_dt = TRACKER_BETA / dt
        self.vx += beta_dt * dx
        self.vy += beta_dt * dy
        self.vz += beta_dt * dz

        self.last_time, self.lost_count = now, 0

    def predict_or_wait(self) -> None:
        if not self._initialized:
            self.is_active = False
            return
        now = time.monotonic()
        dt  = max(min(now - self.last_time, 0.5), 0.005)
        self.current_x += self.vx * dt
        self.current_y += self.vy * dt
        self.current_z += self.vz * dt
        self.vx *= 0.5; self.vy *= 0.5; self.vz *= 0.5
        self.last_time = now
        self.lost_count += 1
        if self.lost_count > MAX_LOST_FRAMES:
            self.is_active = False

    @property
    def position(self):
        return self.current_x, self.current_y, self.current_z

# ============================================================
# 3. Hardware Connection
# ============================================================
def open_serial_safe(port: str, baud: int) -> serial.Serial:
    ser = serial.Serial(port=port, baudrate=baud, timeout=0.05)
    ser.dtr, ser.rts = False, False
    return ser

def send_config(cmd_serial: serial.Serial, cfg_path: str = 'profile.cfg') -> None:
    print(f'[CONFIG] Loading config file: {cfg_path}')
    try:
        with open(cfg_path, 'r') as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip() and not ln.startswith('%')]
    except FileNotFoundError:
        print(f"❌ [錯誤] 找不到設定檔 {cfg_path}，請確認位置。")
        sys.exit(1)

    HANDSHAKE_TOKENS = (b'Done', b'Error', b'Ignored', b'mmwDemo:/>')
    for line in lines:
        cmd_serial.write((line + '\n').encode('ascii'))
        response_buf = b''
        while True:
            chunk = cmd_serial.read(256)
            if chunk:
                response_buf += chunk
                if any(tok in response_buf for tok in HANDSHAKE_TOKENS):
                    break
            time.sleep(0.005)
    print('[CONFIG] 雷達底層設定成功載入！')

# ============================================================
# 4. TLV Decode & Filtering
# ============================================================
def find_magic_word(buf: bytes) -> int: 
    return buf.find(MAGIC_WORD)

def parse_frame(buf: bytes):
    if len(buf) < HEADER_LEN: return np.empty((0, 4), dtype=np.float32)
    try:
        total_len, num_tlvs = struct.unpack_from('<I', buf, 12)[0], struct.unpack_from('<I', buf, 32)[0]
    except struct.error: return np.empty((0, 4), dtype=np.float32)
    if len(buf) < total_len: return np.empty((0, 4), dtype=np.float32)

    offset, points = HEADER_LEN, []
    for _ in range(num_tlvs):
        if offset + TLV_HEADER_LEN > total_len: break
        try:
            tlv_type, tlv_length = struct.unpack_from('<I', buf, offset)[0], struct.unpack_from('<I', buf, offset + 4)[0]
        except struct.error: break
        offset += TLV_HEADER_LEN
        if tlv_type == 1:
            for i in range(tlv_length // POINT_UNIT_LEN):
                p_offset = offset + i * POINT_UNIT_LEN
                if p_offset + POINT_UNIT_LEN > len(buf): break
                try: points.append(struct.unpack_from('<ffff', buf, p_offset))
                except struct.error: break
        offset += tlv_length
    return np.array(points, dtype=np.float32) if points else np.empty((0, 4), dtype=np.float32)

def filter_points(raw_points: np.ndarray) -> np.ndarray:
    if raw_points.shape[0] == 0: return raw_points
    mask = ((np.abs(raw_points[:, 0]) < LIMIT_X) & (np.abs(raw_points[:, 2]) < LIMIT_Z) &
            (raw_points[:, 1] > MIN_Y_DIST) & (raw_points[:, 1] < MAX_Y_DIST) &
            (np.abs(raw_points[:, 3]) >= MIN_VELOCITY) & (np.abs(raw_points[:, 3]) < MAX_VELOCITY))
    return raw_points[mask]

def cluster_and_centroid(filtered_points: np.ndarray):
    if filtered_points.shape[0] < CLUSTER_MIN_SAMPLES: return None
    labels = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES).fit(filtered_points[:, :3]).labels_
    unique_labels = set(labels) - {-1}
    if not unique_labels: return None
    best_centroid, best_y = None, float('inf')
    for lbl in unique_labels:
        cx, cy, cz = filtered_points[:, :3][labels == lbl].mean(axis=0)
        if cy < best_y: best_y, best_centroid = cy, (float(cx), float(cy), float(cz))
    return best_centroid

# ============================================================
# Main entry point
# ============================================================
def main():
    print('=' * 60)
    print('  UAV Flight Core -- Ready for Real Flight')
    print('=' * 60)

    print(f'[SERIAL] 正在連接雷達 ({CMD_PORT} & {DATA_PORT})...')
    try:
        cmd_serial  = open_serial_safe(CMD_PORT,  CMD_BAUD)
        data_serial = open_serial_safe(DATA_PORT, DATA_BAUD)
        send_config(cmd_serial, cfg_path='profile.cfg')
        data_serial.reset_input_buffer()
    except Exception as e:
        print(f"❌ [錯誤] 雷達連線失敗: {e}"); sys.exit(1)

    print(f'[MAVLINK] 正在連接飛控數傳 ({MAVLINK_PORT})...')
    try:
        mav_vehicle = mavutil.mavlink_connection(MAVLINK_PORT, baud=MAVLINK_BAUD, source_system=2)
        print("✅ [MAVLINK] 數傳連線成功！")
    except Exception as e:
        print(f"❌ [錯誤] MAVLink 連線失敗: {e}"); mav_vehicle = None

    tracker = TargetTracker()
    recv_buf = b''
    last_debug_time = time.time()
    bytes_received = 0

    print('\n🚀 [SYSTEM] 實機飛行運算核心已啟動... (按 Ctrl+C 結束)\n')

    try:
        while True:
            chunk = data_serial.read(2048)
            if chunk: 
                recv_buf += chunk
                bytes_received += len(chunk)
                
            if time.time() - last_debug_time > 2.0:
                if bytes_received == 0: print("\n⚠️ [警告] 雷達資料流中斷。")
                bytes_received = 0; last_debug_time = time.time()

            while True:
                idx = find_magic_word(recv_buf)
                if idx < 0:
                    recv_buf = recv_buf[-(MAGIC_LEN - 1):] if len(recv_buf) >= MAGIC_LEN else recv_buf
                    break

                recv_buf = recv_buf[idx:]
                if len(recv_buf) < HEADER_LEN: break

                try: total_len = struct.unpack_from('<I', recv_buf, 12)[0]
                except struct.error: recv_buf = recv_buf[MAGIC_LEN:]; continue
                
                if total_len > 100000 or total_len < HEADER_LEN:
                    recv_buf = recv_buf[MAGIC_LEN:]; continue
                if len(recv_buf) < total_len: break

                frame_buf = recv_buf[:total_len]
                recv_buf  = recv_buf[total_len:]

                raw_pts = parse_frame(frame_buf)
                if raw_pts.shape[0] > 0: raw_pts[:, 1] -= 0.13
                filt_pts = filter_points(raw_pts)
                
                y_dist = filt_pts[:, 1].mean() if len(filt_pts) > 0 else 0.0
                sys.stdout.write(f"\r[SYS] 原始點: {len(raw_pts):>3} | 特徵點: {len(filt_pts):>3} | 鎖定高度: {y_dist:.2f}m    ")
                sys.stdout.flush()

                centroid = cluster_and_centroid(filt_pts)
                if centroid is not None: tracker.update(*centroid)
                else: tracker.predict_or_wait()

                tx, ty, tz = tracker.position
                
                if mav_vehicle is not None:
                    mav_vehicle.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0
                    )
                    
                    if tracker.is_active and ty > 0.1:
                        distance = ty
                        
                        # 💡 [極度關鍵：天地視角反轉與坐標映射]
                        # 飛控 LANDING_TARGET 預設 angle_x 是前後(Pitch)，angle_y 是左右(Roll)
                        # 因為雷達在地面往上看，必須將偵測到的無人機偏移量加上「負號」，才能讓無人機往正確的方向推回去。
                        angle_x_pitch = math.atan2(tz, distance)  # 控制前後
                        angle_y_roll  = math.atan2(-tx, distance)  # 控制左右
                        
                        time_boot_us = int(time.time() * 1_000_000)
                        
                        # 💡 [極度關鍵：修正為 BODY_NED]
                        mav_vehicle.mav.landing_target_send(
                            time_boot_us, 0, 
                            mavutil.mavlink.MAV_FRAME_BODY_NED,  # 告訴飛控這是相對機身的偏移
                            angle_x_pitch, angle_y_roll, distance, 0, 0            
                        )
                        print(f"\n🚀 [引導送出] 高度: {distance:.2f}m | 前後補償(Pitch): {math.degrees(angle_x_pitch):+05.1f}° | 左右補償(Roll): {math.degrees(angle_y_roll):+05.1f}°")
                        
            time.sleep(0.01)

    except KeyboardInterrupt:
        print('\n[EXIT] 收到終止指令，系統安全關閉中...')
    finally:
        if cmd_serial is not None and cmd_serial.is_open: cmd_serial.close()
        if data_serial is not None and data_serial.is_open: data_serial.close()
        if mav_vehicle is not None: mav_vehicle.close()
        print('\n[EXIT] 任務結束，祝起落平安！')

if __name__ == '__main__':
    main()
