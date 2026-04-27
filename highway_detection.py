#!/usr/bin/env python3
"""
Highway Vehicle Detection — Jetson Orin Nano + DeepStream + YOLO26 + MQTT

Pipeline:
    B0495C USB kamera
        -> nvinfer (YOLO26 TensorRT)
        -> nvtracker (NvSORT — Kalman filter tabanlı tracker)
        -> nvdsosd
        -> tee
            ├─ (opsiyonel) yerel ekran
            └─ (opsiyonel) x264enc + RTSP yayını -> rtsp://jetson:8554/stream

Telemetri:
    Pad probe -> queue -> MQTT publisher thread -> broker (default: localhost)
    Topic:   highway/detections/<sensor_id>
    Sıklık:  5 mesaj/saniye (her 6 frame'de bir, 30 FPS varsayımı)

NOT: Orin Nano'nun NVENC donanımı yok, software H.264 encoder (x264enc)
     kullanılıyor. Encode CPU'da yapılır, tipik yük ~%25.

Kullanım:
    python3 highway_detection.py                          # default ayarlar
    python3 highway_detection.py --model n                # yolo26n
    python3 highway_detection.py --debug                  # her frame detayı
    python3 highway_detection.py --no-display             # monitörsüz, sadece RTSP
    python3 highway_detection.py --no-rtsp                # sadece yerel ekran
    python3 highway_detection.py --no-mqtt                # MQTT publishing kapalı
    python3 highway_detection.py --mqtt-host pi5.local    # broker'ı Pi5'e yönlendir

Test:
    # Ayrı bir terminalden mesajları görmek için
    mosquitto_sub -h localhost -t 'highway/#' -v

VLC ile RTSP izleme (aynı LAN'dan):
    vlc rtsp://<JETSON_IP>:8554/stream

Çıkış: Ctrl+C
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import gi
gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import Gst, GLib, GstRtspServer

import pyds

from mqtt_publisher import MqttPublisher


# ─── Sabitler ─────────────────────────────────────────────────────────────────

CAMERA_DEVICE = "/dev/video0"
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1200
CAMERA_FPS = 30

RTSP_PORT = 8554
RTSP_MOUNT = "/stream"
RTSP_UDP_PORT = 5400

ENCODER_BITRATE_KBPS = 4000
STREAMMUX_BUFFER_POOL_SIZE = 16

# Tracker
TRACKER_LIB = "/opt/nvidia/deepstream/deepstream/lib/libnvds_nvmultiobjecttracker.so"
TRACKER_WIDTH = 960
TRACKER_HEIGHT = 544

# MQTT defaults
DEFAULT_MQTT_HOST = "localhost"
DEFAULT_MQTT_PORT = 1883
DEFAULT_SENSOR_ID = "yhnt-jetson-01"

# 30 FPS / 5 msg/s = 6. Yani her 6 frame'de bir publish.
PUBLISH_EVERY_N_FRAMES = 6

# Tracker henüz ID atamamış nesneler için sentinel
TRACK_ID_UNASSIGNED = 0xFFFFFFFFFFFFFFFF

CLASS_NAMES = ["others", "car", "van", "bus"]

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


# ─── Parametre parsing ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Highway araç tespit pipeline'ı + MQTT telemetri",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=["s", "n"], default="s",
                        help="Model boyutu: s (yolo26s) veya n (yolo26n)")
    parser.add_argument("--debug", action="store_true",
                        help="Her frame'in tespit listesini konsola yaz")
    parser.add_argument("--device", default=CAMERA_DEVICE,
                        help="V4L2 kamera aygıtı")
    parser.add_argument("--no-display", action="store_true",
                        help="Yerel monitöre çizme (saha modu)")
    parser.add_argument("--no-rtsp", action="store_true",
                        help="RTSP yayını yapma")
    parser.add_argument("--rtsp-port", type=int, default=RTSP_PORT,
                        help="RTSP sunucu portu")

    # MQTT seçenekleri
    parser.add_argument("--no-mqtt", action="store_true",
                        help="MQTT publishing kapalı")
    parser.add_argument("--mqtt-host", default=DEFAULT_MQTT_HOST,
                        help="MQTT broker hostname/IP")
    parser.add_argument("--mqtt-port", type=int, default=DEFAULT_MQTT_PORT,
                        help="MQTT broker portu")
    parser.add_argument("--sensor-id", default=DEFAULT_SENSOR_ID,
                        help="Bu Jetson'ın benzersiz kimliği (topic'te kullanılır)")

    return parser.parse_args()


# ─── İstatistik toplayıcı ─────────────────────────────────────────────────────

class Stats:
    def __init__(self, mqtt_publisher=None):
        self.frame_count = 0
        self.total_detections = 0
        self.class_counts = defaultdict(int)
        self.unique_track_ids = set()
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self._frames_at_last_report = 0
        self._mqtt = mqtt_publisher
        self._current_fps = 0.0   # Son saniyenin FPS'i, MQTT payload'a giriyor

    def on_frame(self, num_detections, per_class, track_ids_this_frame):
        self.frame_count += 1
        self.total_detections += num_detections
        for cls_id, count in per_class.items():
            self.class_counts[cls_id] += count
        self.unique_track_ids.update(track_ids_this_frame)

        now = time.time()
        if now - self.last_report_time >= 1.0:
            elapsed_total = now - self.start_time
            fps_avg = self.frame_count / elapsed_total
            self._current_fps = (self.frame_count - self._frames_at_last_report) / (now - self.last_report_time)

            cls_summary = " ".join(
                f"{CLASS_NAMES[i]}={self.class_counts[i]}"
                for i in range(len(CLASS_NAMES))
            )

            mqtt_info = ""
            if self._mqtt:
                stats = self._mqtt.get_stats()
                conn = "✓" if self._mqtt.is_connected else "✗"
                mqtt_info = (f" | MQTT {conn} pub={stats['published']} "
                             f"q={self._mqtt.queue_size} drop={stats['dropped']}")

            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"FPS: {self._current_fps:5.1f}/{fps_avg:5.1f} | "
                f"Frame: {self.frame_count:6d} | "
                f"Araç: {len(self.unique_track_ids):4d} | "
                f"{cls_summary}"
                f"{mqtt_info}"
            )
            self.last_report_time = now
            self._frames_at_last_report = self.frame_count

    @property
    def current_fps(self) -> float:
        return self._current_fps


# ─── Pad probe: frame metadata okuma ──────────────────────────────────────────

def make_osd_sink_pad_probe(stats, mqtt_publisher=None, debug=False):
    """
    Probe her frame'de:
      - Tespitleri sayar (Stats için)
      - PUBLISH_EVERY_N_FRAMES'de bir MQTT publisher'a tespit listesini verir
      - Debug modunda detayları konsola basar
    """
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

            per_class = defaultdict(int)
            track_ids_this_frame = []
            detections_in_frame = []   # debug için
            mqtt_detections = []        # MQTT payload için

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                cls_id = obj_meta.class_id
                track_id = obj_meta.object_id
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"

                per_class[cls_id] += 1

                if track_id != TRACK_ID_UNASSIGNED:
                    track_ids_this_frame.append(track_id)

                # MQTT payload — sadece track_id'si atanmış olanları gönder
                if track_id != TRACK_ID_UNASSIGNED:
                    r = obj_meta.rect_params
                    mqtt_detections.append({
                        "track_id": int(track_id),
                        "class": cls_name,
                        "confidence": round(float(obj_meta.confidence), 3),
                        "bbox": [int(r.left), int(r.top), int(r.width), int(r.height)],
                    })

                if debug:
                    r = obj_meta.rect_params
                    track_str = f"id={track_id}" if track_id != TRACK_ID_UNASSIGNED else "id=yeni"
                    detections_in_frame.append(
                        f"{cls_name}[{track_str}] conf={obj_meta.confidence:.2f} "
                        f"bbox=({int(r.left)},{int(r.top)},{int(r.width)},{int(r.height)})"
                    )

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            total = sum(per_class.values())
            stats.on_frame(total, per_class, track_ids_this_frame)

            # Throttled MQTT publish — her 6 frame'de bir
            if mqtt_publisher is not None and frame_meta.frame_num % PUBLISH_EVERY_N_FRAMES == 0:
                mqtt_publisher.publish_detections(
                    frame_id=int(frame_meta.frame_num),
                    fps=stats.current_fps,
                    detections=mqtt_detections,
                )

            if debug and detections_in_frame:
                print(f"  └─ frame#{frame_meta.frame_num}: " + ", ".join(detections_in_frame))

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    return probe


# ─── Pipeline kurulum ─────────────────────────────────────────────────────────

def build_pipeline(args):
    if args.no_display and args.no_rtsp:
        print("HATA: --no-display ve --no-rtsp birlikte kullanılamaz")
        sys.exit(1)

    config_file = os.path.join(
        REPO_ROOT, "configs", f"config_infer_primary_yolo26{args.model}.txt"
    )
    tracker_config = os.path.join(REPO_ROOT, "configs", "tracker_config.yml")

    for path, label in [(config_file, "Inference config"),
                        (tracker_config, "Tracker config"),
                        (TRACKER_LIB, "Tracker library")]:
        if not os.path.exists(path):
            print(f"HATA: {label} bulunamadı: {path}")
            sys.exit(1)

    common = f"""
        v4l2src device={args.device} do-timestamp=true !
        video/x-raw,format=YUY2,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},framerate={CAMERA_FPS}/1 !
        videoconvert !
        video/x-raw,format=NV12 !
        nvvideoconvert copy-hw=2 !
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
        nvvideoconvert copy-hw=2 !
        nvdsosd name=osd !
        nvvideoconvert copy-hw=2 !
        video/x-raw,format=RGBA !
        tee name=t
    """

    display_branch = """
        t. ! queue leaky=downstream max-size-buffers=4 max-size-time=0 max-size-bytes=0 !
        videoconvert !
        autovideosink sync=false
    """ if not args.no_display else ""

    rtsp_branch = f"""
        t. ! queue leaky=downstream max-size-buffers=4 max-size-time=0 max-size-bytes=0 !
        videoconvert !
        video/x-raw,format=I420 !
        x264enc bitrate={ENCODER_BITRATE_KBPS} tune=zerolatency speed-preset=ultrafast key-int-max=30 !
        h264parse !
        rtph264pay config-interval=1 pt=96 !
        udpsink host=127.0.0.1 port={RTSP_UDP_PORT} sync=false async=false
    """ if not args.no_rtsp else ""

    pipeline_str = common + display_branch + rtsp_branch

    print(f"Pipeline kuruluyor:")
    print(f"  Model:    yolo26{args.model}")
    print(f"  Tracker:  NvSORT ({tracker_config})")
    print(f"  Display:  {'kapalı' if args.no_display else 'aktif'}")
    print(f"  RTSP:     {'kapalı' if args.no_rtsp else f'rtsp://<jetson-ip>:{args.rtsp_port}{RTSP_MOUNT}'}")
    if args.no_mqtt:
        print(f"  MQTT:     kapalı")
    else:
        print(f"  MQTT:     {args.mqtt_host}:{args.mqtt_port}")
        print(f"  Topic:    highway/detections/{args.sensor_id}")
        print(f"  Sıklık:   ~{CAMERA_FPS // PUBLISH_EVERY_N_FRAMES} mesaj/sn (her {PUBLISH_EVERY_N_FRAMES} frame'de)")
    print(f"  Debug:    {args.debug}")

    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except GLib.Error as e:
        print(f"Pipeline parse hatası: {e}")
        sys.exit(1)

    return pipeline


# ─── RTSP sunucu ──────────────────────────────────────────────────────────────

def start_rtsp_server(port, mount_path, udp_port):
    server = GstRtspServer.RTSPServer()
    server.props.service = str(port)

    factory = GstRtspServer.RTSPMediaFactory()
    factory.set_launch(
        f"( udpsrc name=pay0 port={udp_port} buffer-size=524288 "
        f'caps="application/x-rtp, media=video, clock-rate=90000, '
        f'encoding-name=H264, payload=96" )'
    )
    factory.set_shared(True)

    mounts = server.get_mount_points()
    mounts.add_factory(mount_path, factory)

    server.attach(None)
    return server


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    Gst.init(None)

    # MQTT publisher'ı pipeline'dan ÖNCE başlat — pipeline başlatıldığında
    # ilk frame'ler de publish edilebilsin
    mqtt_publisher = None
    if not args.no_mqtt:
        mqtt_publisher = MqttPublisher(
            host=args.mqtt_host,
            port=args.mqtt_port,
            sensor_id=args.sensor_id,
        )
        mqtt_publisher.start()

    pipeline = build_pipeline(args)
    stats = Stats(mqtt_publisher=mqtt_publisher)

    osd = pipeline.get_by_name("osd")
    if osd is None:
        print("HATA: nvdsosd bulunamadı pipeline'da")
        sys.exit(1)

    osd_sink_pad = osd.get_static_pad("sink")
    probe_fn = make_osd_sink_pad_probe(stats, mqtt_publisher=mqtt_publisher, debug=args.debug)
    osd_sink_pad.add_probe(Gst.PadProbeType.BUFFER, probe_fn, 0)

    loop = GLib.MainLoop()

    def on_message(bus, msg, loop):
        t = msg.type
        if t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            print(f"\n[HATA] {err.message}")
            if debug:
                print(f"[DEBUG] {debug}")
            loop.quit()
        elif t == Gst.MessageType.EOS:
            print("\n[bilgi] Stream bitti")
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            warn, _ = msg.parse_warning()
            print(f"[uyarı] {warn.message}")
        elif t == Gst.MessageType.STATE_CHANGED:
            if msg.src == pipeline:
                old, new, _ = msg.parse_state_changed()
                if new == Gst.State.PLAYING:
                    print(f"[bilgi] Pipeline aktif — Ctrl+C ile kapat\n")
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message, loop)

    rtsp_server = None
    if not args.no_rtsp:
        rtsp_server = start_rtsp_server(args.rtsp_port, RTSP_MOUNT, RTSP_UDP_PORT)
        print(f"[bilgi] RTSP sunucu hazır: rtsp://<jetson-ip>:{args.rtsp_port}{RTSP_MOUNT}")

    print("[bilgi] Engine yükleniyor (ilk açılışta 10-20 sn sürebilir)...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[bilgi] Kullanıcı kesintisi, kapatılıyor...")

    # Cleanup
    pipeline.set_state(Gst.State.NULL)

    if mqtt_publisher:
        mqtt_publisher.stop()

    # Özet
    total_time = time.time() - stats.start_time
    if stats.frame_count > 0:
        print(f"\n─── Oturum özeti ───")
        print(f"Toplam süre:         {total_time:.1f} sn")
        print(f"İşlenen frame:       {stats.frame_count}")
        print(f"Ortalama FPS:        {stats.frame_count / total_time:.2f}")
        print(f"Toplam tespit:       {stats.total_detections}")
        print(f"Benzersiz araç:      {len(stats.unique_track_ids)}")
        print(f"Sınıf dağılımı:")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name:10s}: {stats.class_counts[i]}")


if __name__ == "__main__":
    main()
