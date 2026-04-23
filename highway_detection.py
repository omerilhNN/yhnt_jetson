#!/usr/bin/env python3
"""
Highway Vehicle Detection — Jetson Orin Nano + DeepStream + YOLO26

Pipeline:
    B0495C USB kamera
        -> nvinfer (YOLO26 TensorRT)
        -> nvdsosd
        -> tee
            ├─ (opsiyonel) yerel ekran
            └─ (opsiyonel) RTSP yayını -> rtsp://jetson:8554/stream

Çalıştırma yeri: /home/dev/Desktop/yhnt/yhnt_jetson/
Repo yapısı:
    highway_detection.py      <- bu dosya
    configs/                  <- inference config dosyaları
    models/                   <- .engine, labels.txt
    lib/                      <- libnvdsinfer_custom_impl_Yolo.so

Kullanım:
    python3 highway_detection.py                   # ekran + RTSP (default)
    python3 highway_detection.py --model n         # yolo26n
    python3 highway_detection.py --debug           # her frame detayı
    python3 highway_detection.py --no-display      # monitörsüz, sadece RTSP
    python3 highway_detection.py --no-rtsp         # sadece yerel ekran

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


# ─── Sabitler ─────────────────────────────────────────────────────────────────

CAMERA_DEVICE = "/dev/video0"
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1200
CAMERA_FPS = 30

RTSP_PORT = 8554
RTSP_MOUNT = "/stream"
RTSP_UDP_PORT = 5400   # Pipeline -> RTSP server arası dahili UDP portu

# Encoder bitrate (bps). 4 Mbps 1080p highway için yeterli.
ENCODER_BITRATE = 4_000_000

# Senin eğittiğin sınıflar (models/labels.txt ile aynı sırada olmalı)
CLASS_NAMES = ["others", "car", "van", "bus"]

# Repo kökü
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


# ─── Parametre parsing ────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Highway araç tespit pipeline'ı",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", choices=["s", "n"], default="s",
        help="Model boyutu: s (yolo26s) veya n (yolo26n)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Her frame'in tespit listesini konsola yaz",
    )
    parser.add_argument(
        "--device", default=CAMERA_DEVICE,
        help="V4L2 kamera aygıtı",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Yerel monitöre çizme (saha modu)",
    )
    parser.add_argument(
        "--no-rtsp", action="store_true",
        help="RTSP yayını yapma",
    )
    parser.add_argument(
        "--rtsp-port", type=int, default=RTSP_PORT,
        help="RTSP sunucu portu",
    )
    return parser.parse_args()


# ─── İstatistik toplayıcı ─────────────────────────────────────────────────────

class Stats:
    def __init__(self):
        self.frame_count = 0
        self.total_detections = 0
        self.class_counts = defaultdict(int)
        self.start_time = time.time()
        self.last_report_time = self.start_time
        self._frames_at_last_report = 0

    def on_frame(self, num_detections, per_class):
        self.frame_count += 1
        self.total_detections += num_detections
        for cls_id, count in per_class.items():
            self.class_counts[cls_id] += count

        now = time.time()
        if now - self.last_report_time >= 1.0:
            elapsed_total = now - self.start_time
            fps_avg = self.frame_count / elapsed_total
            fps_recent = (self.frame_count - self._frames_at_last_report) / (now - self.last_report_time)

            cls_summary = " ".join(
                f"{CLASS_NAMES[i]}={self.class_counts[i]}"
                for i in range(len(CLASS_NAMES))
            )

            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"FPS anlık: {fps_recent:5.1f} | ort: {fps_avg:5.1f} | "
                f"Frame: {self.frame_count:6d} | "
                f"Tespit: {self.total_detections:6d} | "
                f"{cls_summary}"
            )
            self.last_report_time = now
            self._frames_at_last_report = self.frame_count


# ─── Pad probe: frame metadata okuma ──────────────────────────────────────────

def make_osd_sink_pad_probe(stats, debug=False):
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
            detections_in_frame = []

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                cls_id = obj_meta.class_id
                per_class[cls_id] += 1

                if debug:
                    r = obj_meta.rect_params
                    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}"
                    detections_in_frame.append(
                        f"{cls_name} conf={obj_meta.confidence:.2f} "
                        f"bbox=({int(r.left)},{int(r.top)},{int(r.width)},{int(r.height)})"
                    )

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            total = sum(per_class.values())
            stats.on_frame(total, per_class)

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
    """
    Pipeline yapısı:

    [v4l2src ... nvinfer ... nvdsosd] -> tee
                                         ├─ queue -> nvvideoconvert -> videoconvert -> autovideosink
                                         └─ queue -> nvvideoconvert -> nvv4l2h264enc -> rtph264pay -> udpsink

    Display ve/veya RTSP branch'ları bayraklara göre dahil edilir.
    En az biri aktif olmalı.
    """

    if args.no_display and args.no_rtsp:
        print("HATA: --no-display ve --no-rtsp birlikte kullanılamaz (hiç çıkış kalmaz)")
        sys.exit(1)

    config_file = os.path.join(
        REPO_ROOT, "configs", f"config_infer_primary_yolo26{args.model}.txt"
    )

    if not os.path.exists(config_file):
        print(f"HATA: Config dosyası bulunamadı: {config_file}")
        sys.exit(1)

    # Common: kamera -> inference -> OSD -> tee
    common = f"""
        v4l2src device={args.device} !
        video/x-raw,format=YUY2,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},framerate={CAMERA_FPS}/1 !
        videoconvert !
        video/x-raw,format=NV12 !
        nvvideoconvert !
        video/x-raw(memory:NVMM),format=NV12 !
        mux.sink_0 nvstreammux name=mux batch-size=1
                    width={CAMERA_WIDTH} height={CAMERA_HEIGHT}
                    batched-push-timeout=40000 live-source=1 !
        nvinfer config-file-path={config_file} name=primary-inference !
        nvvideoconvert !
        nvdsosd name=osd !
        tee name=t
    """

    # Yerel ekran branch
    display_branch = """
        t. ! queue !
        nvvideoconvert !
        video/x-raw,format=RGBA !
        videoconvert !
        autovideosink sync=false
    """ if not args.no_display else ""

    # RTSP branch — HW H.264 encoder + UDP push to local RTSP server
    rtsp_branch = f"""
        t. ! queue !
        nvvideoconvert !
        video/x-raw,format=I420 !
        x264enc bitrate=4000 tune=zerolatency speed-preset=ultrafast key-int-max=30 !
        h264parse !
        rtph264pay config-interval=1 pt=96 !
        udpsink host=127.0.0.1 port={RTSP_UDP_PORT} sync=false async=false
    """ if not args.no_rtsp else ""

    pipeline_str = common + display_branch + rtsp_branch

    print(f"Pipeline kuruluyor:")
    print(f"  Model:    yolo26{args.model}")
    print(f"  Display:  {'kapalı' if args.no_display else 'aktif'}")
    print(f"  RTSP:     {'kapalı' if args.no_rtsp else f'rtsp://<jetson-ip>:{args.rtsp_port}{RTSP_MOUNT}'}")
    print(f"  Debug:    {args.debug}")

    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except GLib.Error as e:
        print(f"Pipeline parse hatası: {e}")
        sys.exit(1)

    return pipeline


# ─── RTSP sunucu ──────────────────────────────────────────────────────────────

def start_rtsp_server(port, mount_path, udp_port):
    """
    udpsink'in gönderdiği H.264 stream'i alıp RTSP olarak yayınlar.
    İstemciler rtsp://<ip>:<port><mount_path> ile bağlanır.
    """
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

    pipeline = build_pipeline(args)
    stats = Stats()

    osd = pipeline.get_by_name("osd")
    if osd is None:
        print("HATA: nvdsosd bulunamadı pipeline'da")
        sys.exit(1)

    osd_sink_pad = osd.get_static_pad("sink")
    probe_fn = make_osd_sink_pad_probe(stats, debug=args.debug)
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
        print(f"        (VLC ile izle: vlc rtsp://<jetson-ip>:{args.rtsp_port}{RTSP_MOUNT})")

    print("[bilgi] Engine yükleniyor (ilk açılışta 10-20 sn sürebilir)...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[bilgi] Kullanıcı kesintisi, kapatılıyor...")

    total_time = time.time() - stats.start_time
    if stats.frame_count > 0:
        print(f"\n─── Oturum özeti ───")
        print(f"Toplam süre:      {total_time:.1f} sn")
        print(f"İşlenen frame:    {stats.frame_count}")
        print(f"Ortalama FPS:     {stats.frame_count / total_time:.2f}")
        print(f"Toplam tespit:    {stats.total_detections}")
        for i, name in enumerate(CLASS_NAMES):
            print(f"  {name:10s}: {stats.class_counts[i]}")

    pipeline.set_state(Gst.State.NULL)


if __name__ == "__main__":
    main()
