#!/usr/bin/env python3
"""
Highway Vehicle Detection — Jetson Orin Nano + DeepStream + YOLO + MQTT

MQTT topic yapısı:
- highway/telemetry/<sensor_id>/detections   (active-only, QoS0)
- highway/telemetry/<sensor_id>/stats        (QoS0)
- highway/sensors/<sensor_id>/status         (QoS1 retained + LWT)
- highway/sensors/<sensor_id>/meta           (QoS1 retained)
- highway/sensors/<sensor_id>/heartbeat      (QoS0)
- highway/events/<sensor_id>/vehicle/enter   (QoS1)
- highway/events/<sensor_id>/vehicle/exit    (QoS1)
- highway/commands/<sensor_id>/request       (subscribe)
- highway/commands/<sensor_id>/response      (QoS1)
"""

import argparse
import os
import sys
import time
from collections import Counter, defaultdict, deque

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer

import numpy as np
import pyds
from mqtt_publisher import MqttPublisher
from anomaly_detector import AnomalyDetector, AnomalyConfig, TrackInfo

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CAMERA_DEVICE = "/dev/v4l/by-id/usb-Arducam_Arducam_B0495__USB3_2.3MP__Arducam_202500915_0001-video-index0"
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1200
CAMERA_FPS = 30

RTSP_PORT = 8554
RTSP_MOUNT = "/stream"
RTSP_UDP_PORT = 5400

ENCODER_BITRATE_KBPS = 9000
STREAMMUX_BUFFER_POOL_SIZE = 16
NVVIDEOCONVERT_COPY_HW = 2  # JP6.2 workaround

TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
TRACKER_WIDTH = 960
TRACKER_HEIGHT = 544

DEFAULT_MQTT_HOST = "100.84.29.29"
DEFAULT_MQTT_PORT = 1883
DEFAULT_SENSOR_ID = "jetson01"

DEFAULT_RTSP_BIND = "0.0.0.0"

PUBLISH_EVERY_N_FRAMES = 3
TRACK_ID_UNASSIGNED = 0xFFFFFFFFFFFFFFFF

# Yeni 5-class model mapping
CLASS_NAMES = ["car", "van", "bus", "motorcycle", "truck"]

TRACK_CONFIRM_MIN_HITS = 1
TRACK_LOST_TTL_FRAMES = 45
TRACK_END_TTL_FRAMES = 90
TRACK_HISTORY_MAXLEN = 30

# Speed estimation params
SPEED_EMA_ALPHA = 0.35
SPEED_MIN_DT_S = 1.0 / (CAMERA_FPS * 1.5)   # çok küçük dt jitter'ı engelle
SPEED_MAX_KMH = 260.0                       # outlier clamp
SPEED_MIN_MOVE_M = 0.05                     # çok küçük hareketi ignore et

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
HOMOGRAPHY_PATH = os.path.join(REPO_ROOT, "configs", "homography.npy")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Highway araç tespit pipeline'ı + MQTT telemetri",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=["s", "n"], default="s")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", default=CAMERA_DEVICE)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--no-rtsp", action="store_true")
    parser.add_argument("--rtsp-port", type=int, default=RTSP_PORT)
    parser.add_argument("--rtsp-bind", default=DEFAULT_RTSP_BIND)
    parser.add_argument("--no-mqtt", action="store_true")
    parser.add_argument("--mqtt-host", default=DEFAULT_MQTT_HOST)
    parser.add_argument("--mqtt-port", type=int, default=DEFAULT_MQTT_PORT)
    parser.add_argument("--sensor-id", default=DEFAULT_SENSOR_ID)
    parser.add_argument("--no-anomaly", action="store_true",
                        help="Anomali tespitini devre dışı bırak")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Homography + Speed helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_homography(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Homography dosyası bulunamadı: {path}")
    H = np.load(path)
    if H.shape != (3, 3):
        raise ValueError(f"Homography shape hatalı: {H.shape}, beklenen (3,3)")
    return H.astype(np.float64)


def pixel_to_world(H, u, v):
    p = np.array([float(u), float(v), 1.0], dtype=np.float64)
    w = H @ p
    if abs(w[2]) < 1e-9:
        return None
    return (float(w[0] / w[2]), float(w[1] / w[2]))


def bbox_bottom_center(rect_params):
    u = float(rect_params.left) + float(rect_params.width) * 0.5
    v = float(rect_params.top) + float(rect_params.height)
    return u, v


# ─────────────────────────────────────────────────────────────────────────────
# Stats / Track lifecycle
# ─────────────────────────────────────────────────────────────────────────────

class Stats:
    def __init__(self, mqtt_publisher=None):
        self.frame_count = 0
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self._frames_at_last_report = 0
        self._mqtt = mqtt_publisher
        self._current_fps = 0.0
        self.total_raw_detections = 0

        # tid -> track data
        self.tracks = {}

        self._last_class_counts_unique = defaultdict(int)
        self._last_confirmed_count = 0

        self._pending_enter_events = []
        self._pending_exit_events = []

    def _ensure_track(self, tid, cls_id, frame_num):
        if tid not in self.tracks:
            self.tracks[tid] = {
                "state": "tentative",
                "first_seen": frame_num,
                "last_seen": frame_num,
                "hits": 1,
                "misses": 0,
                "class_recent": deque([cls_id], maxlen=TRACK_HISTORY_MAXLEN),
                "majority_class": cls_id,
                "entered_emitted": False,
                "exited_emitted": False,
            }
            return

        tr = self.tracks[tid]
        tr["last_seen"] = frame_num
        tr["hits"] += 1
        tr["misses"] = 0
        tr["class_recent"].append(cls_id)
        tr["majority_class"] = Counter(tr["class_recent"]).most_common(1)[0][0]

    def _transition_states(self, current_frame_num, seen_tids):
        for tid, tr in list(self.tracks.items()):
            if tid in seen_tids:
                if tr["state"] == "tentative" and tr["hits"] >= TRACK_CONFIRM_MIN_HITS:
                    tr["state"] = "confirmed"
                    if not tr["entered_emitted"]:
                        self._pending_enter_events.append({
                            "type": "enter",
                            "track_id": int(tid),
                            "frame": int(current_frame_num),
                            "class_id": int(tr["majority_class"]),
                        })
                        tr["entered_emitted"] = True
                elif tr["state"] == "lost":
                    tr["state"] = "confirmed"
                continue

            tr["misses"] = current_frame_num - tr["last_seen"]

            if tr["state"] in ("tentative", "confirmed") and tr["misses"] >= TRACK_LOST_TTL_FRAMES:
                tr["state"] = "lost"

            if tr["misses"] >= TRACK_END_TTL_FRAMES:
                tr["state"] = "ended"
                if tr["entered_emitted"] and not tr["exited_emitted"]:
                    self._pending_exit_events.append({
                        "type": "exit",
                        "track_id": int(tid),
                        "frame": int(current_frame_num),
                        "class_id": int(tr["majority_class"]),
                    })
                    tr["exited_emitted"] = True
                del self.tracks[tid]

    def on_frame(self, frame_num, raw_detection_count, track_id_class_pairs):
        self.frame_count += 1
        self.total_raw_detections += raw_detection_count

        seen_tids = set()
        for tid, cls_id in track_id_class_pairs:
            self._ensure_track(tid, cls_id, frame_num)
            seen_tids.add(tid)

        self._transition_states(frame_num, seen_tids)

        now = time.time()
        if now - self.last_report_time >= 1.0:
            elapsed_total = now - self.start_time
            fps_avg = self.frame_count / elapsed_total if elapsed_total > 0 else 0.0
            self._current_fps = (
                (self.frame_count - self._frames_at_last_report) / (now - self.last_report_time)
                if (now - self.last_report_time) > 0 else 0.0
            )

            class_counts_unique = defaultdict(int)
            confirmed_count = 0
            for tr in self.tracks.values():
                if tr["state"] == "confirmed":
                    confirmed_count += 1
                    class_counts_unique[tr["majority_class"]] += 1

            self._last_class_counts_unique = class_counts_unique
            self._last_confirmed_count = confirmed_count

            # stats topic publish (1Hz)
            if self._mqtt:
                life = self.get_lifecycle_snapshot()
                self._mqtt.publish_stats(
                    fps=self._current_fps,
                    queue_size=self._mqtt.queue_size,
                    extra={
                        "tracks_confirmed": life["confirmed"],
                        "tracks_lost": life["lost"],
                        "tracks_tentative": life["tentative"],
                        "tracks_active_total": life["active_total"],
                    },
                )

            cls_summary = " ".join(
                f"{CLASS_NAMES[i]}={class_counts_unique[i]}" for i in range(len(CLASS_NAMES))
            )

            mqtt_info = ""
            if self._mqtt:
                mstats = self._mqtt.get_stats()
                conn = "✓" if self._mqtt.is_connected else "✗"
                mqtt_info = (
                    f" | MQTT {conn} pub={mstats['published']} "
                    f"evt={mstats['events_published']} "
                    f"q={self._mqtt.queue_size} drop={mstats['dropped']}"
                )

            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"FPS: {self._current_fps:5.1f}/{fps_avg:5.1f} | "
                f"Frame: {self.frame_count:6d} | "
                f"Araç: {confirmed_count:4d} | "
                f"{cls_summary}{mqtt_info}"
            )

            self.last_report_time = now
            self._frames_at_last_report = self.frame_count

    def drain_pending_enter_events(self):
        evts = self._pending_enter_events[:]
        self._pending_enter_events.clear()
        return evts

    def drain_pending_exit_events(self):
        evts = self._pending_exit_events[:]
        self._pending_exit_events.clear()
        return evts

    @property
    def current_fps(self):
        return self._current_fps

    @property
    def unique_vehicle_count(self):
        return self._last_confirmed_count

    def get_class_summary(self):
        return dict(self._last_class_counts_unique)

    def get_lifecycle_snapshot(self):
        tentative = confirmed = lost = 0
        for tr in self.tracks.values():
            st = tr["state"]
            if st == "tentative":
                tentative += 1
            elif st == "confirmed":
                confirmed += 1
            elif st == "lost":
                lost += 1
        return {
            "tentative": tentative,
            "confirmed": confirmed,
            "lost": lost,
            "active_total": len(self.tracks),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Probe
# ─────────────────────────────────────────────────────────────────────────────

def make_osd_sink_pad_probe(stats, mqtt_publisher=None, debug=False, homography=None, anomaly_detector=None):
    # tid -> speed state
    speed_state = {}

    def _estimate_speed_kmh(track_id, world_xy, ts_sec):
        st = speed_state.get(track_id)
        if st is None:
            speed_state[track_id] = {
                "last_world": world_xy,
                "last_ts": ts_sec,
                "speed_ema_kmh": 0.0,
            }
            return 0.0, 0.0  # raw, ema

        dt = ts_sec - st["last_ts"]
        if dt < SPEED_MIN_DT_S:
            return st["speed_ema_kmh"], st["speed_ema_kmh"]

        dx = world_xy[0] - st["last_world"][0]
        dy = world_xy[1] - st["last_world"][1]
        dist_m = (dx * dx + dy * dy) ** 0.5

        if dist_m < SPEED_MIN_MOVE_M:
            raw_kmh = 0.0
        else:
            raw_kmh = (dist_m / dt) * 3.6

        if raw_kmh > SPEED_MAX_KMH:
            raw_kmh = SPEED_MAX_KMH

        ema = SPEED_EMA_ALPHA * raw_kmh + (1.0 - SPEED_EMA_ALPHA) * st["speed_ema_kmh"]

        st["last_world"] = world_xy
        st["last_ts"] = ts_sec
        st["speed_ema_kmh"] = ema
        return raw_kmh, ema

    def probe(pad, info, u_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if batch_meta is None:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            track_id_class_pairs = []
            raw_detection_count = 0
            mqtt_detections = []   # active object candidates
            debug_lines = []

            # timestamp: PTS varsa onu kullan, yoksa fallback wall clock
            ts_sec = 0.0
            if hasattr(frame_meta, "buf_pts") and frame_meta.buf_pts and int(frame_meta.buf_pts) > 0:
                ts_sec = float(frame_meta.buf_pts) / 1e9
            else:
                ts_sec = time.time()

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                cls_id = obj_meta.class_id
                track_id = obj_meta.object_id
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
                raw_detection_count += 1

                if track_id != TRACK_ID_UNASSIGNED:
                    track_id_class_pairs.append((track_id, cls_id))
                    r = obj_meta.rect_params

                    speed_kmh = 0.0
                    if homography is not None:
                        u, v = bbox_bottom_center(r)
                        world_xy = pixel_to_world(homography, u, v)
                        if world_xy is not None:
                            _, speed_ema = _estimate_speed_kmh(int(track_id), world_xy, ts_sec)
                            speed_kmh = max(0.0, float(speed_ema))

                    # OSD label: ID yanında hız
                    obj_meta.text_params.display_text = (
                        f"{cls_name}[id={int(track_id)}] {speed_kmh:.1f} km/h"
                    )

                    mqtt_detections.append({
                        "track_id": int(track_id),
                        "class": cls_name,
                        "confidence": round(float(obj_meta.confidence), 2),
                        "bbox": [int(r.left), int(r.top), int(r.width), int(r.height)],
                        "track_state": "active",
                        "speed_kmh": round(speed_kmh, 1),
                    })

                    if debug:
                        debug_lines.append(
                            f"{cls_name}[id={track_id}] speed={speed_kmh:.1f}km/h "
                            f"conf={obj_meta.confidence:.2f} "
                            f"bbox=({int(r.left)},{int(r.top)},{int(r.width)},{int(r.height)})"
                        )

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            stats.on_frame(int(frame_meta.frame_num), raw_detection_count, track_id_class_pairs)

            # enter/exit event objeleri
            event_objects = []

            for e in stats.drain_pending_enter_events():
                cls_id = e["class_id"]
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
                event_objects.append({
                    "track_id": e["track_id"],
                    "class": cls_name,
                    "confidence": 0.0,
                    "bbox": [0, 0, 0, 0],
                    "track_state": "enter",
                })

            for e in stats.drain_pending_exit_events():
                cls_id = e["class_id"]
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
                event_objects.append({
                    "track_id": e["track_id"],
                    "class": cls_name,
                    "confidence": 0.0,
                    "bbox": [0, 0, 0, 0],
                    "track_state": "exit",
                })

            has_events = len(event_objects) > 0
            should_publish = (frame_meta.frame_num % PUBLISH_EVERY_N_FRAMES == 0) or has_events

            # ACTIVE-ONLY detections
            active_only = [d for d in mqtt_detections if d.get("track_state") == "active"]

            if mqtt_publisher is not None and should_publish:
                mqtt_publisher.publish_detections(
                    frame_id=int(frame_meta.frame_num),
                    fps=stats.current_fps,
                    detections=active_only,
                    event_objects=event_objects,
                )

            # ── Anomaly detection ──
            if anomaly_detector is not None:
                track_infos = []
                for d in mqtt_detections:
                    tid = d["track_id"]
                    ss = speed_state.get(tid)
                    if ss is not None:
                        track_infos.append(TrackInfo(
                            track_id=tid,
                            class_name=d["class"],
                            speed_kmh=d["speed_kmh"],
                            world_x=ss["last_world"][0],
                            world_y=ss["last_world"][1],
                            bbox=d["bbox"],
                        ))
                anomalies = anomaly_detector.update(
                    frame_id=int(frame_meta.frame_num),
                    timestamp=ts_sec,
                    tracks=track_infos,
                )
                if anomalies:
                    if mqtt_publisher is not None:
                        mqtt_publisher.publish_anomalies(
                            frame_id=int(frame_meta.frame_num),
                            anomalies=[a.to_dict() for a in anomalies],
                        )
                    if debug:
                        for a in anomalies:
                            print(f"  ⚠ ANOMALY: {a.anomaly_type.value} "
                                  f"severity={a.severity.value} "
                                  f"track={a.track_id} "
                                  f"dur={a.duration_sec:.1f}s "
                                  f"msg={a.message}")

            if debug and debug_lines:
                print(f"  └─ frame#{frame_meta.frame_num}: " + ", ".join(debug_lines))

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    return probe


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(args):
    if args.no_display and args.no_rtsp:
        print("HATA: --no-display ve --no-rtsp birlikte kullanılamaz")
        sys.exit(1)

    config_file = os.path.join(REPO_ROOT, "configs", f"config_infer_primary_yolo26{args.model}.txt")
    tracker_config = os.path.join(REPO_ROOT, "configs", "tracker_config.yml")

    for path, label in [
        (config_file, "Inference config"),
        (tracker_config, "Tracker config"),
        (TRACKER_LIB, "Tracker library"),
    ]:
        if not os.path.exists(path):
            print(f"HATA: {label} bulunamadı: {path}")
            sys.exit(1)

    chw = NVVIDEOCONVERT_COPY_HW

    common = f"""
        v4l2src device={args.device} do-timestamp=true !
        video/x-raw,format=YUY2,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},framerate={CAMERA_FPS}/1 !
        videoconvert !
        video/x-raw,format=NV12 !
        nvvideoconvert copy-hw={chw} !
        video/x-raw(memory:NVMM),format=NV12 !
        mux.sink_0 nvstreammux name=mux
                    batch-size=1
                    width={CAMERA_WIDTH} height={CAMERA_HEIGHT}
                    batched-push-timeout=40000
                    live-source=1
                    sync-inputs=0
                    buffer-pool-size={STREAMMUX_BUFFER_POOL_SIZE}
                    nvbuf-memory-type=0 !
        nvinfer config-file-path={config_file} name=primary-inference !
        nvtracker name=tracker
                    ll-lib-file={TRACKER_LIB}
                    ll-config-file={tracker_config}
                    tracker-width={TRACKER_WIDTH}
                    tracker-height={TRACKER_HEIGHT}
                    display-tracking-id=1 !
        nvvideoconvert copy-hw={chw} !
        nvdsosd name=osd !
        nvvideoconvert copy-hw={chw} !
        video/x-raw,format=RGBA !
        tee name=t
    """

    display_branch = """
        t. ! queue leaky=downstream max-size-buffers=8 max-size-time=0 max-size-bytes=0 !
        videoconvert !
        autovideosink sync=false
    """ if not args.no_display else ""

    rtsp_branch = f"""
        t. ! queue leaky=downstream max-size-buffers=8 max-size-time=0 max-size-bytes=0 !
        videoconvert !
        video/x-raw,format=I420 !
        x264enc bitrate={ENCODER_BITRATE_KBPS} tune=zerolatency speed-preset=ultrafast
                threads=2 key-int-max=60 b-adapt=false bframes=0
                option-string="no-mbtree:sliced-threads:rc-lookahead=0" !
        h264parse !
        rtph264pay config-interval=1 pt=96 !
        udpsink host=127.0.0.1 port={RTSP_UDP_PORT} sync=false async=false buffer-size=2097152
    """ if not args.no_rtsp else ""

    pipeline_str = common + display_branch + rtsp_branch

    print("Pipeline kuruluyor:")
    print(f"  Model:    yolo26{args.model}")
    print(f"  Tracker:  NvSORT ({tracker_config})")
    print(f"  Display:  {'kapalı' if args.no_display else 'aktif'}")
    if args.no_rtsp:
        rtsp_info = "kapalı"
    else:
        rtsp_host = args.rtsp_bind if args.rtsp_bind != "0.0.0.0" else "<jetson-ip>"
        rtsp_info = f"rtsp://{rtsp_host}:{args.rtsp_port}{RTSP_MOUNT}"

    print(f"  RTSP:     {rtsp_info}")
    print(f"  MQTT:     {'kapalı' if args.no_mqtt else f'{args.mqtt_host}:{args.mqtt_port}'}")
    if not args.no_mqtt:
        print(f"  Topic(detections): highway/telemetry/{args.sensor_id}/detections")
        print(f"  Topic(stats):      highway/telemetry/{args.sensor_id}/stats")
        print(f"  Topic(status):     highway/sensors/{args.sensor_id}/status")
        print(f"  Topic(meta):       highway/sensors/{args.sensor_id}/meta")
        print(f"  Topic(heartbeat):  highway/sensors/{args.sensor_id}/heartbeat")
        print(f"  Topic(enter):      highway/events/{args.sensor_id}/vehicle/enter")
        print(f"  Topic(exit):       highway/events/{args.sensor_id}/vehicle/exit")
        print(f"  Topic(cmd req):    highway/commands/{args.sensor_id}/request")
        print(f"  Topic(cmd resp):   highway/commands/{args.sensor_id}/response")
    print(f"  Publish:  her {PUBLISH_EVERY_N_FRAMES} frame (~{CAMERA_FPS // PUBLISH_EVERY_N_FRAMES} msg/sn)")
    print(f"  Confirm:  min_hits={TRACK_CONFIRM_MIN_HITS}")
    print(f"  TTL:      lost={TRACK_LOST_TTL_FRAMES} end={TRACK_END_TTL_FRAMES}")
    print(f"  Debug:    {args.debug}")

    try:
        return Gst.parse_launch(pipeline_str)
    except GLib.Error as e:
        print(f"Pipeline parse hatası: {e}")
        sys.exit(1)


def start_rtsp_server(port, mount_path, udp_port, bind_address="0.0.0.0"):
    server = GstRtspServer.RTSPServer()
    server.props.service = str(port)
    server.props.address = bind_address

    factory = GstRtspServer.RTSPMediaFactory()
    factory.set_launch(
        f'( udpsrc name=pay0 port={udp_port} buffer-size=2097152 '
        f'caps="application/x-rtp, media=video, clock-rate=90000, encoding-name=H264, payload=96" )'
    )
    factory.set_shared(True)

    mounts = server.get_mount_points()
    mounts.add_factory(mount_path, factory)
    server.attach(None)
    return server


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    Gst.init(None)

    # Homography yükle
    try:
        H = load_homography(HOMOGRAPHY_PATH)
        print(f"[bilgi] Homography yüklendi: {HOMOGRAPHY_PATH}")
    except Exception as e:
        print(f"[HATA] Homography yüklenemedi: {e}")
        sys.exit(1)

    mqtt_publisher = None
    if not args.no_mqtt:
        mqtt_publisher = MqttPublisher(
            host=args.mqtt_host,
            port=args.mqtt_port,
            sensor_id=args.sensor_id,
        )
        mqtt_publisher.start()

    # Anomaly detector
    anomaly_det = None
    if not args.no_anomaly:
        anomaly_det = AnomalyDetector(
            config=AnomalyConfig(),
            sensor_id=args.sensor_id,
        )
        print(f"[bilgi] Anomaly detection aktif")
        print(f"  Topic(anomaly): highway/anomalies/{args.sensor_id}/detections")
    else:
        print(f"[bilgi] Anomaly detection devre dışı (--no-anomaly)")

    pipeline = build_pipeline(args)
    stats = Stats(mqtt_publisher=mqtt_publisher)

    osd = pipeline.get_by_name("osd")
    if osd is None:
        print("HATA: nvdsosd bulunamadı")
        sys.exit(1)

    osd_sink_pad = osd.get_static_pad("sink")
    if osd_sink_pad is None:
        print("HATA: osd sink pad alınamadı")
        sys.exit(1)

    probe_fn = make_osd_sink_pad_probe(
        stats,
        mqtt_publisher=mqtt_publisher,
        debug=args.debug,
        homography=H,
        anomaly_detector=anomaly_det,
    )
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, probe_fn, 0)

    loop = GLib.MainLoop()

    def on_message(bus, msg, loop_ref):
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, debug_msg = msg.parse_error()
            print(f"\n[HATA] {err.message}")
            if debug_msg:
                print(f"[DEBUG] {debug_msg}")
            loop_ref.quit()
        elif t == Gst.MessageType.EOS:
            print("\n[bilgi] Stream bitti")
            loop_ref.quit()
        elif t == Gst.MessageType.WARNING:
            warn, _ = msg.parse_warning()
            print(f"[uyarı] {warn.message}")
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    if not args.no_rtsp:
        _ = start_rtsp_server(args.rtsp_port, RTSP_MOUNT, RTSP_UDP_PORT, args.rtsp_bind)

    print("[bilgi] Engine yükleniyor...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[bilgi] Kapatılıyor...")

    pipeline.set_state(Gst.State.NULL)

    if mqtt_publisher:
        mqtt_publisher.stop()

    total_time = time.time() - stats.start_time
    if stats.frame_count > 0:
        print("\n─── Oturum özeti ───")
        print(f"Toplam süre:         {total_time:.1f} sn")
        print(f"İşlenen frame:       {stats.frame_count}")
        print(f"Ortalama FPS:        {stats.frame_count / total_time:.2f}")
        print(f"Toplam tespit (raw): {stats.total_raw_detections}")
        print(f"Benzersiz araç:      {stats.unique_vehicle_count}")
        print("Sınıf dağılımı (confirmed, unique):")
        cls_summary = stats.get_class_summary()
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name:10s}: {cls_summary.get(i, 0)}")


if __name__ == "__main__":
    main()
