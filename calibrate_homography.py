#!/usr/bin/env python3
"""
Canli kameradan tek frame yakala, 4 nokta tikla, parametreleri GUI'den gir,
configs/homography.npy uret.

Kullanim:
    python3 calibrate_homography.py

Sahnede 4 referans noktasi: bir seridin iki ardisik kesik cizgi
baslangiclarinin sol/sag kenarlari.
"""

import os
import sys

import cv2
import numpy as np

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst


# ─── Sabitler ───
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(REPO_ROOT, "configs", "homography.npy")

CAMERA_DEVICE = "/dev/video0"
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1200
CAMERA_FPS = 30

WARMUP_FRAMES = 30

DEFAULT_LANE_WIDTH_M = 3.75
DEFAULT_LANE_PERIOD_M = 18.0


# ─── Frame yakalama ───
def grab_frame_from_camera() -> np.ndarray:
    Gst.init(None)

    pipeline_str = (
        f"v4l2src device={CAMERA_DEVICE} num-buffers={WARMUP_FRAMES + 5} ! "
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


# ─── 4 nokta tiklama ───
clicked_points: list[tuple[int, int]] = []
click_image: np.ndarray | None = None
click_window = "Tikla: P1 (yakin sol) -> P2 (yakin sag) -> P3 (uzak sag) -> P4 (uzak sol)  |  R=sifirla, ESC=cik"


def on_mouse(event, x, y, flags, param):
    global clicked_points, click_image

    if event != cv2.EVENT_LBUTTONDOWN:
        return
    if len(clicked_points) >= 4:
        return

    clicked_points.append((x, y))
    label = f"P{len(clicked_points)}"
    cv2.circle(click_image, (x, y), 6, (0, 255, 0), -1)
    cv2.putText(click_image, label, (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # P1->P2->P3->P4 cizgilerini de ciz, kullaniciya goster
    if len(clicked_points) >= 2:
        for i in range(len(clicked_points) - 1):
            cv2.line(click_image, clicked_points[i], clicked_points[i + 1],
                     (0, 200, 0), 2)
    if len(clicked_points) == 4:
        cv2.line(click_image, clicked_points[3], clicked_points[0],
                 (0, 200, 0), 2)

    cv2.imshow(click_window, click_image)
    print(f"[{label}] piksel = ({x}, {y})")


def collect_4_points(frame: np.ndarray) -> list[tuple[int, int]] | None:
    global clicked_points, click_image

    clicked_points = []
    click_image = frame.copy()

    cv2.namedWindow(click_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(click_window, 1280, 800)
    cv2.imshow(click_window, click_image)
    cv2.setMouseCallback(click_window, on_mouse)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == 27:  # ESC
            cv2.destroyWindow(click_window)
            return None
        if key in (ord("r"), ord("R")):
            clicked_points = []
            click_image = frame.copy()
            cv2.imshow(click_window, click_image)
            print("[bilgi] Sifirlandi.")
            continue
        if len(clicked_points) == 4:
            break

    cv2.destroyWindow(click_window)
    return list(clicked_points)


# ─── GUI input dialog ───
def prompt_params_gui(frame: np.ndarray,
                      points: list[tuple[int, int]],
                      defaults: tuple[float, float] = (DEFAULT_LANE_WIDTH_M, DEFAULT_LANE_PERIOD_M)
                      ) -> tuple[float, float] | None:
    """
    Frame uzerinde tiklanmis 4 noktayi gosterirken, yari-saydam panel uzerinden
    iki float input al. Tab=sonraki alan, Enter=onayla, ESC=iptal, Backspace=sil.
    """
    win = "Yol parametreleri  |  Tab=sonraki, Enter=onayla, ESC=iptal"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 800)

    fields = [
        {"label": "Serit genisligi (P1-P2 arasi, m)",  "value": str(defaults[0])},
        {"label": "Periyot (P1-P4 arasi, m)",          "value": str(defaults[1])},
    ]
    active = 0

    def is_valid_char(ch: int) -> bool:
        return (48 <= ch <= 57) or ch in (ord("."), ord(","))

    while True:
        canvas = frame.copy()

        # 4 noktayi ve baglantilari ciz
        for i, (x, y) in enumerate(points):
            cv2.circle(canvas, (x, y), 8, (0, 255, 0), -1)
            cv2.putText(canvas, f"P{i+1}", (x + 12, y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        # P1-P2 (kirmizi: width)
        cv2.line(canvas, points[0], points[1], (0, 0, 255), 3)
        # P3-P4 (kirmizi: width)
        cv2.line(canvas, points[2], points[3], (0, 0, 255), 3)
        # P1-P4 (mavi: period)
        cv2.line(canvas, points[0], points[3], (255, 100, 0), 3)
        # P2-P3 (mavi: period)
        cv2.line(canvas, points[1], points[2], (255, 100, 0), 3)

        # Yari-saydam panel
        h, w = canvas.shape[:2]
        panel_w, panel_h = 760, 360
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

        # Renk legend'i
        cv2.line(canvas, (x0 + 24, y0 + 78), (x0 + 60, y0 + 78), (0, 0, 255), 3)
        cv2.putText(canvas, "Kirmizi = serit genisligi",
                    (x0 + 70, y0 + 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.line(canvas, (x0 + 360, y0 + 78), (x0 + 396, y0 + 78), (255, 100, 0), 3)
        cv2.putText(canvas, "Mavi = periyot",
                    (x0 + 406, y0 + 84),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        cv2.putText(canvas, "Otoyol: 3.75 / 18.0    Devlet yolu: 3.50 / 9.0",
                    (x0 + 24, y0 + 114),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

        # Alanlar
        for i, f in enumerate(fields):
            ty = y0 + 170 + i * 80
            color = (0, 255, 100) if i == active else (180, 180, 180)
            cv2.putText(canvas, f["label"], (x0 + 24, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1)

            box_x = x0 + 460
            box_y = ty - 28
            cv2.rectangle(canvas, (box_x, box_y), (box_x + 260, box_y + 40),
                          color, 2)
            text = f["value"]
            if i == active:
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                cv2.line(canvas,
                         (box_x + 12 + text_size[0] + 4, box_y + 8),
                         (box_x + 12 + text_size[0] + 4, box_y + 32),
                         color, 2)
            cv2.putText(canvas, text, (box_x + 12, box_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(canvas, "Tab: sonraki alan   Enter: onayla   ESC: iptal",
                    (x0 + 24, y0 + panel_h - 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        cv2.imshow(win, canvas)

        # waitKeyEx: platforma gore degisen keycode'lari daha guvenilir yakalar
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
            continue

        # Enter
        if key in (13, 10):
            try:
                w_val = float(fields[0]["value"].replace(",", "."))
                p_val = float(fields[1]["value"].replace(",", "."))
            except ValueError:
                continue

            if 0.1 <= w_val <= 100 and 0.1 <= p_val <= 100:
                cv2.destroyWindow(win)
                cv2.waitKey(1)  # GUI event flush
                return (w_val, p_val)
            continue

        # Backspace (sisteme gore 8 veya 127 gelebilir)
        if key in (8, 127):
            fields[active]["value"] = fields[active]["value"][:-1]
            continue

        if is_valid_char(key):
            ch = chr(key)
            if ch == ",":
                ch = "."
            if ch == "." and "." in fields[active]["value"]:
                continue
            fields[active]["value"] += ch
            continue


def build_world_points(lane_width_m: float, lane_period_m: float) -> np.ndarray:
    return np.array([
        [0.0,          0.0],            # P1: yakin sol
        [lane_width_m, 0.0],            # P2: yakin sag
        [lane_width_m, lane_period_m],  # P3: uzak sag
        [0.0,          lane_period_m],  # P4: uzak sol
    ], dtype=np.float32)


# ─── Main ───
def main():
    print("[bilgi] Kameradan frame yakalaniyor...")
    try:
        frame = grab_frame_from_camera()
    except Exception as e:
        print(f"[hata] Frame yakalanamadi: {e}")
        sys.exit(1)
    print(f"[bilgi] Frame alindi: {frame.shape[1]}x{frame.shape[0]}")

    # 1) 4 nokta tikla
    print("[bilgi] 4 noktayi sirayla tikla. R=sifirla, ESC=iptal.")
    points = collect_4_points(frame)
    if points is None:
        print("[bilgi] Iptal edildi.")
        sys.exit(0)
    print(f"[bilgi] 4 nokta alindi:")
    for i, p in enumerate(points, 1):
        print(f"        P{i}: {p}")

    # 2) GUI'den parametreleri al
    params = prompt_params_gui(frame, points)
    if params is None:
        print("[bilgi] Iptal edildi.")
        sys.exit(0)

    lane_width, lane_period = params
    world_points = build_world_points(lane_width, lane_period)

    print(f"[bilgi] Serit genisligi: {lane_width} m")
    print(f"[bilgi] Periyot: {lane_period} m")
    print(f"[bilgi] Dunya koordinatlari:")
    for i, wp in enumerate(world_points, 1):
        print(f"        P{i}: {wp.tolist()}")

    # 3) Homografi hesapla
    pts_img = np.array(points, dtype=np.float32)
    H, _ = cv2.findHomography(pts_img, world_points)
    if H is None:
        print("[hata] Homografi hesaplanamadi.")
        sys.exit(1)

    # 4) Kaydet
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.save(OUTPUT_PATH, H)

    backup_jpg = os.path.join(REPO_ROOT, "configs", "calibration_frame.jpg")
    cv2.imwrite(backup_jpg, frame)

    print("\n─── Sonuc ───")
    print(f"Homografi matrisi:\n{H}")
    print(f"\nKaydedildi: {OUTPUT_PATH}")
    print(f"Yedek frame: {backup_jpg}")

    # 5) Dogrulama
    print("\n─── Dogrulama ───")
    for i, (u, v) in enumerate(points):
        p = np.array([u, v, 1.0])
        w = H @ p
        X, Y = w[0] / w[2], w[1] / w[2]
        expected = world_points[i]
        err = float(np.linalg.norm([X - expected[0], Y - expected[1]]))
        print(f"P{i+1}: piksel({u},{v}) -> dunya({X:6.2f}, {Y:6.2f})  "
              f"beklenen({expected[0]:.2f}, {expected[1]:.2f})  hata={err:.3f} m")

    print("\n[bilgi] Artik 'python3 main.py' ile pipeline'i baslatabilirsin.")


if __name__ == "__main__":
    main()