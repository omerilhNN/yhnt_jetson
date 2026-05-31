#!/usr/bin/env python3
"""
Homografi Kalibrasyon Araci — Tek Serit, 4 Nokta

Kameradan tek frame yakalar, kullanicidan 4 nokta alir,
serit genisligi ve periyot parametrelerini GUI'den sorar,
configs/homography.npy uretir.

Referans noktasi secimi:
  Tek seridin (tercihen orta serit) iki ardisik kesik cizgi baslangiclarinin
  sol ve sag kenarlari. Orta serit lens distorsiyonunun en az oldugu bolgedir.

Tiklama sirasi (saat yonunde):
  P1 = yakin sol   (sana en yakin kesik cizginin sol kenari)
  P2 = yakin sag   (sana en yakin kesik cizginin sag kenari)
  P3 = uzak sag    (bir sonraki kesik cizginin sag kenari)
  P4 = uzak sol    (bir sonraki kesik cizginin sol kenari)

        Yol goruntusu:              Dunya koordinatlari:
     P4 ──────── P3                P4 ──────── P3
      \\          /                 |            |
       \\        /                  | periyot(m) |
        \\      /                   |            |
     P1 ──── P2                   P1 ──────── P2
     (yakin, genis)                 genislik(m)

Kullanim:
    python3 calibrate_homography.py
    python3 calibrate_homography.py --device /dev/video0
    python3 calibrate_homography.py --image frame.jpg
    python3 calibrate_homography.py --video demo1.mp4
    python3 calibrate_homography.py --video demo1.mp4 --video-time 5.0
"""

import argparse
import os
import sys

import cv2
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst


# ─── Sabitler ───────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(REPO_ROOT, "configs")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "homography.npy")

CAMERA_DEVICE = "/dev/video0"
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1200
CAMERA_FPS = 30

WARMUP_FRAMES = 30

DEFAULT_LANE_WIDTH_M = 3.75   # Otoyol standart serit genisligi
DEFAULT_LANE_PERIOD_M = 18.0  # Otoyol kesik cizgi periyodu (9m cizgi + 9m bosluk)

# Reprojection hata esigi (metre): bunun uzerindeyse uyari ver
REPROJECTION_WARN_THRESHOLD_M = 0.10

# Kondisyon sayisi esigi: bunun uzerindeyse matris sayisal olarak guvenilmez
CONDITION_NUMBER_WARN = 1e6


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Homografi kalibrasyon araci — tek serit, 4 nokta",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--device", default=CAMERA_DEVICE,
        help="V4L2 kamera cihazi",
    )
    parser.add_argument(
        "--image", default=None,
        help="Kamera yerine mevcut bir goruntu dosyasi kullan (.jpg/.png)",
    )
    parser.add_argument(
        "--video", default=None,
        help="Video dosyasindan frame al (MP4/MKV/AVI). "
             "--video-time ile hangi saniyeden alinacagi belirtilebilir.",
    )
    parser.add_argument(
        "--video-time", type=float, default=2.0,
        help="--video ile kullanilir: frame'in alinacagi saniye (varsayilan: 2.0s)",
    )
    parser.add_argument(
        "--video-browse", action="store_true",
        help="--video ile kullanilir: video uzerinde interaktif gezinerek "
             "frame sec (ok tuslari / trackbar ile)",
    )
    parser.add_argument(
        "--width", type=float, default=DEFAULT_LANE_WIDTH_M,
        help="Varsayilan serit genisligi (m)",
    )
    parser.add_argument(
        "--period", type=float, default=DEFAULT_LANE_PERIOD_M,
        help="Varsayilan periyot (m)",
    )
    return parser.parse_args()


# ─── Frame yakalama ─────────────────────────────────────────────────────────

def grab_frame_from_camera(device: str) -> np.ndarray:
    """GStreamer ile kameradan tek frame yakala (warmup sonrasi)."""
    Gst.init(None)

    pipeline_str = (
        f"v4l2src device={device} num-buffers={WARMUP_FRAMES + 5} ! "
        f"video/x-raw,format=YUY2,width={CAMERA_WIDTH},height={CAMERA_HEIGHT},"
        f"framerate={CAMERA_FPS}/1 ! "
        f"videoconvert ! "
        f"video/x-raw,format=BGR ! "
        f"appsink name=sink emit-signals=false max-buffers=5 drop=false sync=false"
    )

    pipeline = Gst.parse_launch(pipeline_str)
    sink = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)

    last_frame = None
    frame_count = 0

    try:
        while frame_count < WARMUP_FRAMES + 1:
            sample = sink.emit("pull-sample")
            if sample is None:
                break

            buf = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = structure.get_value("width")
            height = structure.get_value("height")

            success, mapinfo = buf.map(Gst.MapFlags.READ)
            if not success:
                continue

            try:
                arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
                arr = arr.reshape((height, width, 3))
                last_frame = arr.copy()
            finally:
                buf.unmap(mapinfo)

            frame_count += 1
    finally:
        pipeline.set_state(Gst.State.NULL)

    if last_frame is None:
        raise RuntimeError("Kameradan frame alinamadi")

    return last_frame


def load_frame_from_file(path: str) -> np.ndarray:
    """Dosyadan frame yukle."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Goruntu dosyasi bulunamadi: {path}")
    frame = cv2.imread(path)
    if frame is None:
        raise RuntimeError(f"Goruntu okunamadi: {path}")
    return frame


def grab_frame_from_video(path: str, time_sec: float) -> np.ndarray:
    """
    Video dosyasindan belirli bir saniyedeki frame'i al.
    OpenCV VideoCapture kullanir (codec bagimsiz).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video dosyasi bulunamadi: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    if time_sec < 0:
        time_sec = 0
    if duration_sec > 0 and time_sec > duration_sec:
        print(f"[uyari] İstenen zaman ({time_sec:.1f}s) video süresinden ({duration_sec:.1f}s) büyük, "
              f"son saniyeye ayarlanıyor.")
        time_sec = max(0, duration_sec - 0.1)

    target_ms = time_sec * 1000.0
    cap.set(cv2.CAP_PROP_POS_MSEC, target_ms)

    ret, frame = cap.read()
    actual_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    cap.release()

    if not ret or frame is None:
        raise RuntimeError(f"Video'dan frame okunamadı (t={time_sec:.1f}s)")

    print(f"[bilgi] Video: {path}")
    print(f"[bilgi]   FPS={fps:.1f}, toplam={total_frames} frame, süre={duration_sec:.1f}s")
    print(f"[bilgi]   Frame alındı: t={actual_ms/1000:.2f}s")

    return frame


def browse_video_frames(path: str) -> np.ndarray | None:
    """
    Video dosyasini interaktif olarak gezinerek frame sec.

    Kontroller:
      ← / → : 1 frame geri/ileri
      A / D  : 1 saniye geri/ileri
      Q / E  : 10 saniye geri/ileri
      Space/Enter : Bu frame'i sec
      ESC    : İptal
      Trackbar ile de gezinilebilir.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video dosyasi bulunamadi: {path}")

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Video acilamadi: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video frame sayısı alınamadı")

    print(f"[bilgi] Video: {path}")
    print(f"[bilgi]   FPS={fps:.1f}, toplam={total_frames} frame, süre={duration_sec:.1f}s")
    print()
    print("─── Video gezgini ───")
    print("  ← / →     : 1 frame geri/ileri")
    print("  A / D      : 1 saniye geri/ileri")
    print("  Q / E      : 10 saniye geri/ileri")
    print("  Space/Enter: Bu frame'i seç")
    print("  ESC        : İptal")
    print()

    win_name = "Video gezgini  |  ←→=frame  AD=1s  QE=10s  Space=sec  ESC=iptal"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1280, 800)

    current_frame_idx = 0
    selected_frame = None

    # Trackbar callback
    def on_trackbar(val):
        nonlocal current_frame_idx
        current_frame_idx = val

    cv2.createTrackbar("Frame", win_name, 0, max(total_frames - 1, 1), on_trackbar)

    def read_frame_at(idx):
        idx = max(0, min(idx, total_frames - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        return frame

    last_displayed_idx = -1

    while True:
        # Frame'i oku ve göster
        if current_frame_idx != last_displayed_idx:
            frame = read_frame_at(current_frame_idx)
            if frame is None:
                current_frame_idx = max(0, current_frame_idx - 1)
                continue
            last_displayed_idx = current_frame_idx

            # Bilgi overlay
            display = frame.copy()
            t_sec = current_frame_idx / fps if fps > 0 else 0
            info = f"Frame {current_frame_idx}/{total_frames-1}  |  t={t_sec:.2f}s / {duration_sec:.1f}s"
            cv2.putText(display, info, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(display, "Space/Enter = sec   ESC = iptal", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow(win_name, display)

            # Trackbar'ı güncelle
            cv2.setTrackbarPos("Frame", win_name, current_frame_idx)

        key = cv2.waitKeyEx(30)
        if key < 0:
            continue

        k = key & 0xFF

        # ESC
        if k == 27:
            selected_frame = None
            break

        # Space veya Enter
        if k in (32, 13, 10):
            selected_frame = read_frame_at(current_frame_idx)
            t_sec = current_frame_idx / fps if fps > 0 else 0
            print(f"[bilgi] Frame seçildi: #{current_frame_idx}, t={t_sec:.2f}s")
            break

        # Ok tuşları (← = 81/65361, → = 83/65363)
        if key == 65361 or k == 81:  # ← sol ok
            current_frame_idx = max(0, current_frame_idx - 1)
        elif key == 65363 or k == 83:  # → sağ ok
            current_frame_idx = min(total_frames - 1, current_frame_idx + 1)

        # A/D = 1 saniye
        elif k in (ord("a"), ord("A")):
            jump = int(fps) if fps > 0 else 30
            current_frame_idx = max(0, current_frame_idx - jump)
        elif k in (ord("d"), ord("D")):
            jump = int(fps) if fps > 0 else 30
            current_frame_idx = min(total_frames - 1, current_frame_idx + jump)

        # Q/E = 10 saniye
        elif k in (ord("q"), ord("Q")):
            jump = int(fps * 10) if fps > 0 else 300
            current_frame_idx = max(0, current_frame_idx - jump)
        elif k in (ord("e"), ord("E")):
            jump = int(fps * 10) if fps > 0 else 300
            current_frame_idx = min(total_frames - 1, current_frame_idx + jump)

    cap.release()
    cv2.destroyWindow(win_name)
    cv2.waitKey(1)
    return selected_frame


# ─── Geometri dogrulama ─────────────────────────────────────────────────────

def polygon_area_signed(pts):
    """Shoelace formuluyle isaret alan hesapla. Pozitif = saat yonunun tersi."""
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return area / 2.0


def is_convex_quadrilateral(pts):
    """4 noktanin konveks dortgen olusturup olusturamadigini kontrol et."""
    n = len(pts)
    if n != 4:
        return False

    signs = []
    for i in range(n):
        p0 = pts[i]
        p1 = pts[(i + 1) % n]
        p2 = pts[(i + 2) % n]
        cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
        signs.append(cross)

    # Tum cross product'lar ayni isarette olmali
    all_pos = all(s > 0 for s in signs)
    all_neg = all(s < 0 for s in signs)
    return all_pos or all_neg


def validate_point_order(pts):
    """
    Nokta sirasini dogrula:
    - 4 nokta konveks dortgen olusturmali
    - P1-P2 (yakin kenar) P3-P4'ten (uzak kenar) genis olmali (perspektif)
    - Saat yonunde siralanmali

    Dondurulen: (gecerli_mi, hata_mesaji_veya_None)
    """
    if len(pts) != 4:
        return False, "Tam 4 nokta gerekli"

    if not is_convex_quadrilateral(pts):
        return False, (
            "Noktalar konveks dortgen olusturmuyor.\n"
            "Capraz tiklama yapmis olabilirsin. Sira: P1(yakin sol) -> P2(yakin sag) -> P3(uzak sag) -> P4(uzak sol)"
        )

    # Saat yonu kontrolu (signed area negatif = saat yonu, goruntu koordinatlarinda y asagi)
    area = polygon_area_signed(pts)
    if area > 0:
        return False, (
            "Noktalar saat yonunun tersinde siralanmis.\n"
            "Dogru sira: P1(yakin sol) -> P2(yakin sag) -> P3(uzak sag) -> P4(uzak sol)"
        )

    # Perspektif kontrolu: yakin kenar (P1-P2) uzak kenardan (P3-P4) genis olmali
    dist_near = np.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
    dist_far = np.hypot(pts[2][0] - pts[3][0], pts[2][1] - pts[3][1])
    if dist_far > dist_near * 1.1:  # %10 tolerans
        return False, (
            f"Uzak kenar ({dist_far:.0f}px) yakin kenardan ({dist_near:.0f}px) genis.\n"
            "P1-P2 sana yakin (genis), P3-P4 uzak (dar) olmali."
        )

    # P1-P2 yaklasik yatay olmali
    angle_near = abs(np.degrees(np.arctan2(
        pts[1][1] - pts[0][1], pts[1][0] - pts[0][0]
    )))
    if angle_near > 30:
        return False, (
            f"P1-P2 cizgisi yataydan {angle_near:.1f}° sapiyor.\n"
            "Yakin kenarin iki ucunu yatay olarak tikla."
        )

    return True, None


# ─── 4 nokta tiklama ────────────────────────────────────────────────────────

clicked_points: list[tuple[int, int]] = []
click_image: np.ndarray | None = None
click_original: np.ndarray | None = None
CLICK_WINDOW = "P1(yakin sol) > P2(yakin sag) > P3(uzak sag) > P4(uzak sol)  |  R=sifirla  Z=geri  ESC=cik"

POINT_LABELS = ["P1 yakin sol", "P2 yakin sag", "P3 uzak sag", "P4 uzak sol"]
POINT_COLORS = [(0, 255, 0), (0, 200, 255), (255, 100, 0), (255, 0, 200)]


def _redraw_points():
    """Mevcut noktalari click_image uzerine yeniden ciz."""
    global click_image
    click_image = click_original.copy()

    for i, (x, y) in enumerate(clicked_points):
        color = POINT_COLORS[i]
        cv2.circle(click_image, (x, y), 8, color, -1)
        cv2.circle(click_image, (x, y), 12, color, 2)
        cv2.putText(click_image, f"P{i+1}", (x + 14, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Cizgileri ciz
    if len(clicked_points) >= 2:
        # P1-P2 (yakin kenar, kirmizi)
        cv2.line(click_image, clicked_points[0], clicked_points[1], (0, 0, 255), 2)
    if len(clicked_points) >= 3:
        # P2-P3 (sag kenar)
        cv2.line(click_image, clicked_points[1], clicked_points[2], (200, 200, 0), 2)
    if len(clicked_points) >= 4:
        # P3-P4 (uzak kenar, kirmizi)
        cv2.line(click_image, clicked_points[2], clicked_points[3], (0, 0, 255), 2)
        # P4-P1 (sol kenar)
        cv2.line(click_image, clicked_points[3], clicked_points[0], (200, 200, 0), 2)

    # Sonraki nokta ipucu
    if len(clicked_points) < 4:
        hint = f"Siradaki: {POINT_LABELS[len(clicked_points)]}"
        cv2.putText(click_image, hint, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow(CLICK_WINDOW, click_image)


def on_mouse(event, x, y, flags, param):
    global clicked_points

    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if len(clicked_points) >= 4:
        return

    clicked_points.append((x, y))
    idx = len(clicked_points)
    print(f"  [P{idx}] piksel = ({x}, {y})  — {POINT_LABELS[idx-1]}")

    _redraw_points()

    # 4 nokta tamam, hemen dogrula
    if len(clicked_points) == 4:
        ok, err = validate_point_order(clicked_points)
        if not ok:
            print(f"\n  [!] {err}")
            print("  [bilgi] R ile sifirla ve tekrar dene.\n")
            # Goruntude de uyariyi goster
            cv2.putText(click_image, "HATA: " + err.split("\n")[0],
                        (20, click_image.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow(CLICK_WINDOW, click_image)


def collect_4_points(frame: np.ndarray) -> list[tuple[int, int]] | None:
    """Kullanicidan 4 nokta al. Dogrulama gecene kadar tekrar ettir."""
    global clicked_points, click_image, click_original

    clicked_points = []
    click_original = frame.copy()
    click_image = frame.copy()

    cv2.namedWindow(CLICK_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CLICK_WINDOW, 1280, 800)

    # Baslangic ipucu
    cv2.putText(click_image, f"Siradaki: {POINT_LABELS[0]}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imshow(CLICK_WINDOW, click_image)
    cv2.setMouseCallback(CLICK_WINDOW, on_mouse)

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(CLICK_WINDOW)
            return None

        # R = tamamen sifirla
        if key in (ord("r"), ord("R")):
            clicked_points = []
            _redraw_points()
            print("  [bilgi] Tum noktalar sifirlandi.")
            continue

        # Z = son noktayi geri al
        if key in (ord("z"), ord("Z")) and clicked_points:
            removed = clicked_points.pop()
            _redraw_points()
            print(f"  [bilgi] Son nokta geri alindi: {removed}")
            continue

        # 4 nokta tiklandi ve gecerli mi?
        if len(clicked_points) == 4:
            ok, _ = validate_point_order(clicked_points)
            if ok:
                # Enter ile onayla veya otomatik gecis
                if key in (13, 10, 32):  # Enter veya Space
                    break
                # 4 nokta gecerli, ekranda "Enter ile onayla" goster
                temp = click_image.copy()
                cv2.putText(temp, "OK — Enter veya Space ile onayla",
                            (20, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(CLICK_WINDOW, temp)

    cv2.destroyWindow(CLICK_WINDOW)
    cv2.waitKey(1)
    return list(clicked_points)


# ─── GUI parametre girisi ───────────────────────────────────────────────────

def prompt_params_gui(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    defaults: tuple[float, float] = (DEFAULT_LANE_WIDTH_M, DEFAULT_LANE_PERIOD_M),
) -> tuple[float, float] | None:
    """
    Frame uzerinde tiklanmis 4 noktayi gosterirken, yari-saydam panel
    uzerinden serit genisligi ve periyot degerlerini al.

    Tab=sonraki alan, Enter=onayla, ESC=iptal, Backspace=sil.
    """
    win = "Yol parametreleri  |  Tab=sonraki  Enter=onayla  ESC=iptal"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 800)

    fields = [
        {"label": "Serit genisligi (P1-P2, m)",  "value": str(defaults[0])},
        {"label": "Periyot (P1-P4, m)",           "value": str(defaults[1])},
    ]
    active = 0
    error_msg = ""

    def is_valid_char(ch: int) -> bool:
        return (48 <= ch <= 57) or ch in (ord("."), ord(","))

    while True:
        canvas = frame.copy()

        # 4 noktayi ciz
        for i, (x, y) in enumerate(points):
            color = POINT_COLORS[i]
            cv2.circle(canvas, (x, y), 8, color, -1)
            cv2.putText(canvas, f"P{i+1}", (x + 12, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # P1-P2, P3-P4 (kirmizi: genislik)
        cv2.line(canvas, points[0], points[1], (0, 0, 255), 3)
        cv2.line(canvas, points[2], points[3], (0, 0, 255), 3)
        # P1-P4, P2-P3 (mavi: periyot)
        cv2.line(canvas, points[0], points[3], (255, 100, 0), 3)
        cv2.line(canvas, points[1], points[2], (255, 100, 0), 3)

        # Yari-saydam panel
        h, w = canvas.shape[:2]
        panel_w, panel_h = 760, 380
        x0 = (w - panel_w) // 2
        y0 = (h - panel_h) // 2
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h),
                      (200, 200, 200), 2)

        # Baslik
        cv2.putText(canvas, "Yol parametrelerini gir (metre)",
                    (x0 + 24, y0 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Renk legendi
        cv2.line(canvas, (x0 + 24, y0 + 78), (x0 + 60, y0 + 78), (0, 0, 255), 3)
        cv2.putText(canvas, "Kirmizi = serit genisligi (P1-P2)",
                    (x0 + 70, y0 + 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.line(canvas, (x0 + 24, y0 + 104), (x0 + 60, y0 + 104), (255, 100, 0), 3)
        cv2.putText(canvas, "Mavi = periyot (P1-P4)",
                    (x0 + 70, y0 + 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # Referans degerleri
        cv2.putText(canvas, "Otoyol: 3.75 / 18.0    Devlet yolu: 3.50 / 9.0",
                    (x0 + 24, y0 + 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Input alanlari
        for i, f in enumerate(fields):
            ty = y0 + 190 + i * 80
            color = (0, 255, 100) if i == active else (180, 180, 180)
            cv2.putText(canvas, f["label"], (x0 + 24, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1)

            box_x = x0 + 420
            box_y = ty - 28
            cv2.rectangle(canvas, (box_x, box_y), (box_x + 300, box_y + 40),
                          color, 2)
            text = f["value"]
            if i == active:
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cursor_x = box_x + 12 + text_size[0] + 4
                cv2.line(canvas, (cursor_x, box_y + 8), (cursor_x, box_y + 32),
                         color, 2)
            cv2.putText(canvas, text, (box_x + 12, box_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Hata mesaji
        if error_msg:
            cv2.putText(canvas, error_msg,
                        (x0 + 24, y0 + panel_h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(canvas, "Tab: sonraki alan   Enter: onayla   ESC: iptal",
                    (x0 + 24, y0 + panel_h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        cv2.imshow(win, canvas)

        k = cv2.waitKeyEx(20)
        if k < 0:
            continue
        key = k & 0xFF

        if key == 255 or key == 0:
            continue

        if key == 27:  # ESC
            cv2.destroyWindow(win)
            cv2.waitKey(1)
            return None

        if key == 9:  # Tab
            active = (active + 1) % len(fields)
            error_msg = ""
            continue

        # Enter
        if key in (13, 10):
            error_msg = ""
            try:
                w_val = float(fields[0]["value"].replace(",", "."))
                p_val = float(fields[1]["value"].replace(",", "."))
            except ValueError:
                error_msg = "Gecersiz sayi formati"
                continue

            if w_val <= 0 or p_val <= 0:
                error_msg = "Degerler pozitif olmali"
                continue
            if not (0.5 <= w_val <= 20.0):
                error_msg = f"Serit genisligi mantikli degil: {w_val}m (beklenen 0.5-20m)"
                continue
            if not (1.0 <= p_val <= 100.0):
                error_msg = f"Periyot mantikli degil: {p_val}m (beklenen 1-100m)"
                continue

            cv2.destroyWindow(win)
            cv2.waitKey(1)
            return (w_val, p_val)

        # Backspace
        if key in (8, 127):
            fields[active]["value"] = fields[active]["value"][:-1]
            error_msg = ""
            continue

        if is_valid_char(key):
            ch = chr(key)
            if ch == ",":
                ch = "."
            if ch == "." and "." in fields[active]["value"]:
                continue
            fields[active]["value"] += ch
            error_msg = ""
            continue


# ─── Homografi hesaplama ────────────────────────────────────────────────────

def build_world_points(lane_width_m: float, lane_period_m: float) -> np.ndarray:
    """
    4 noktanin dunya koordinatlarini olustur.
    Orijin = P1 (yakin sol kose).
    X ekseni = seridin enine yonu (sag tarafa dogru).
    Y ekseni = seridin boyuna yonu (uzaga dogru).
    """
    return np.array([
        [0.0,          0.0],            # P1: yakin sol
        [lane_width_m, 0.0],            # P2: yakin sag
        [lane_width_m, lane_period_m],  # P3: uzak sag
        [0.0,          lane_period_m],  # P4: uzak sol
    ], dtype=np.float32)


def compute_homography(
    pixel_points: list[tuple[int, int]],
    world_points: np.ndarray,
) -> tuple[np.ndarray | None, float, list[float]]:
    """
    Homografi hesapla ve dogrula.

    getPerspectiveTransform kullanir (findHomography degil):
    tam 4 nokta var, RANSAC'a gerek yok, RANSAC bazen bir noktayi
    outlier sayip dejenere matris uretebilir.

    Dondurulen: (H_matrisi, kondisyon_sayisi, nokta_basi_hatalar_metre)
    """
    pts_img = np.array(pixel_points, dtype=np.float32)
    pts_world = world_points.copy()

    H = cv2.getPerspectiveTransform(pts_img, pts_world)

    # Kondisyon sayisi kontrolu
    cond = np.linalg.cond(H)

    # Reprojection hatalari
    errors = []
    for i, (u, v) in enumerate(pixel_points):
        p = np.array([float(u), float(v), 1.0], dtype=np.float64)
        w = H @ p
        if abs(w[2]) < 1e-12:
            errors.append(float("inf"))
            continue
        X = w[0] / w[2]
        Y = w[1] / w[2]
        expected = world_points[i]
        err = float(np.hypot(X - expected[0], Y - expected[1]))
        errors.append(err)

    return H, cond, errors


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Kaynak doğrulama: sadece biri seçilmeli
    sources = [args.image, args.video]
    if sum(s is not None for s in sources) > 1:
        print("[hata] --image ve --video aynı anda kullanılamaz. Birini seçin.")
        sys.exit(1)

    # Frame al
    if args.video:
        print(f"[bilgi] Video dosyasından frame alınıyor: {args.video}")
        try:
            if args.video_browse:
                frame = browse_video_frames(args.video)
                if frame is None:
                    print("[bilgi] İptal edildi.")
                    sys.exit(0)
            else:
                frame = grab_frame_from_video(args.video, args.video_time)
        except Exception as e:
            print(f"[hata] {e}")
            sys.exit(1)
    elif args.image:
        print(f"[bilgi] Goruntu dosyasindan yukleniyor: {args.image}")
        try:
            frame = load_frame_from_file(args.image)
        except Exception as e:
            print(f"[hata] {e}")
            sys.exit(1)
    else:
        print(f"[bilgi] Kameradan frame yakalaniyor ({args.device})...")
        try:
            frame = grab_frame_from_camera(args.device)
        except Exception as e:
            print(f"[hata] Frame yakalanamadi: {e}")
            sys.exit(1)

    print(f"[bilgi] Frame boyutu: {frame.shape[1]}x{frame.shape[0]}")

    # 1) 4 nokta tikla
    print()
    print("─── Nokta secimi ───")
    print("Tek seridin (tercihen orta serit) iki ardisik kesik cizgi baslangicini tikla.")
    print("Sira: P1(yakin sol) → P2(yakin sag) → P3(uzak sag) → P4(uzak sol)")
    print("R=sifirla, Z=geri al, ESC=iptal")
    print()

    points = collect_4_points(frame)
    if points is None:
        print("[bilgi] Iptal edildi.")
        sys.exit(0)

    print(f"\n[bilgi] 4 nokta alindi:")
    for i, p in enumerate(points):
        print(f"  P{i+1}: piksel({p[0]}, {p[1]})  — {POINT_LABELS[i]}")

    # 2) GUI'den parametreleri al
    params = prompt_params_gui(frame, points, defaults=(args.width, args.period))
    if params is None:
        print("[bilgi] Iptal edildi.")
        sys.exit(0)

    lane_width, lane_period = params
    world_points = build_world_points(lane_width, lane_period)

    print(f"\n[bilgi] Serit genisligi: {lane_width} m")
    print(f"[bilgi] Periyot:         {lane_period} m")

    # 3) Homografi hesapla
    H, cond, errors = compute_homography(points, world_points)
    if H is None:
        print("[hata] Homografi hesaplanamadi.")
        sys.exit(1)

    # Kondisyon uyarisi
    if cond > CONDITION_NUMBER_WARN:
        print(f"\n[!] UYARI: Kondisyon sayisi cok yuksek ({cond:.0e}).")
        print("    Matris sayisal olarak guvenilmez. Noktalari tekrar sec.")
        print("    Noktalar birbirine cok yakin veya neredeyse ayni cizgi uzerinde olabilir.\n")

    max_err = max(errors)
    if max_err > REPROJECTION_WARN_THRESHOLD_M:
        print(f"\n[!] UYARI: Reprojection hatasi yuksek (max {max_err:.4f} m).")
        print("    getPerspectiveTransform ile bu normalde ~0 olmali.")
        print("    Noktalar veya parametreler tutarsiz olabilir.\n")

    # 4) Kaydet
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_PATH, H)

    backup_jpg = os.path.join(OUTPUT_DIR, "calibration_frame.jpg")
    annotated = frame.copy()
    for i, (x, y) in enumerate(points):
        color = POINT_COLORS[i]
        cv2.circle(annotated, (x, y), 8, color, -1)
        cv2.putText(annotated, f"P{i+1}", (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.line(annotated, points[0], points[1], (0, 0, 255), 2)
    cv2.line(annotated, points[2], points[3], (0, 0, 255), 2)
    cv2.line(annotated, points[0], points[3], (255, 100, 0), 2)
    cv2.line(annotated, points[1], points[2], (255, 100, 0), 2)
    cv2.imwrite(backup_jpg, annotated)

    # Kalibrasyon parametrelerini de kaydet (tekrar uretim icin)
    calib_data = {
        "pixel_points": points,
        "lane_width_m": lane_width,
        "lane_period_m": lane_period,
        "world_points": world_points.tolist(),
        "condition_number": float(cond),
        "reprojection_errors_m": errors,
    }
    calib_path = os.path.join(OUTPUT_DIR, "calibration_params.npy")
    np.save(calib_path, calib_data)

    # 5) Sonuc raporu
    print("\n─── Sonuc ───")
    print(f"Homografi matrisi:\n{H}")
    print(f"\nKondisyon sayisi: {cond:.2e}")
    print(f"Kaydedildi:       {OUTPUT_PATH}")
    print(f"Parametreler:     {calib_path}")
    print(f"Yedek frame:      {backup_jpg}")

    print("\n─── Reprojection dogrulama ───")
    for i, (u, v) in enumerate(points):
        p = np.array([float(u), float(v), 1.0], dtype=np.float64)
        w = H @ p
        X, Y = w[0] / w[2], w[1] / w[2]
        expected = world_points[i]
        status = "OK" if errors[i] < REPROJECTION_WARN_THRESHOLD_M else "YUKSEK"
        print(
            f"  P{i+1}: piksel({u},{v}) → dunya({X:6.3f}, {Y:6.3f})  "
            f"beklenen({expected[0]:.2f}, {expected[1]:.2f})  "
            f"hata={errors[i]:.4f}m [{status}]"
        )

    print(f"\n[bilgi] Artik 'python3 highway_detection.py' ile pipeline'i baslatabilirsin.")


if __name__ == "__main__":
    main()
