#!/usr/bin/env python3
"""
Mock MQTT Publisher — Normal Telemetri + Anomali Senaryoları

Dashboard geliştirme için Jetson olmadan test verisi üretir.
Normal araç telemetrisi + periyodik anomali senaryoları yayınlar.

Kullanım:
    # Sadece normal telemetri (mevcut davranış)
    python3 mock_anomaly_publisher.py

    # Normal telemetri + anomali senaryoları
    python3 mock_anomaly_publisher.py --mock-anomaly

    # Sadece anomaliler (telemetri olmadan)
    python3 mock_anomaly_publisher.py --mock-anomaly --anomaly-only

    # Özel MQTT broker
    python3 mock_anomaly_publisher.py --mock-anomaly --host 192.168.1.100
"""

import argparse
import json
import math
import random
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

SCHEMA_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_HOST = "100.84.29.29"
DEFAULT_PORT = 1883
DEFAULT_SENSOR_ID = "jetson01"

CLASS_NAMES = ["car", "van", "bus", "motorcycle", "truck"]
CLASS_WEIGHTS = [0.55, 0.12, 0.08, 0.10, 0.15]

# Simülasyon parametreleri
NORMAL_VEHICLE_COUNT = (4, 12)        # min-max aktif araç sayısı
NORMAL_SPEED_RANGE = (60.0, 130.0)    # km/h
PUBLISH_INTERVAL_SEC = 0.2            # 5 Hz telemetri
ANOMALY_INTERVAL_SEC = 8.0            # Her 8 saniyede bir anomali senaryosu


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def random_bbox() -> list[int]:
    x = random.randint(100, 1700)
    y = random.randint(200, 1000)
    w = random.randint(60, 200)
    h = random.randint(40, 150)
    return [x, y, w, h]


def random_world_pos() -> tuple[float, float]:
    """Dünya koordinatlarında rastgele pozisyon (metre)."""
    x = random.uniform(-2.0, 6.0)    # ~2 şerit genişliği
    y = random.uniform(0.0, 50.0)    # kamera görüş mesafesi
    return round(x, 2), round(y, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Vehicle simulator
# ─────────────────────────────────────────────────────────────────────────────

class MockVehicle:
    """Tek bir simüle edilmiş araç."""
    _next_id = 1

    def __init__(self, anomaly_type: str | None = None):
        self.track_id = MockVehicle._next_id
        MockVehicle._next_id += 1
        self.class_name = random.choices(CLASS_NAMES, CLASS_WEIGHTS, k=1)[0]
        self.confidence = round(random.uniform(0.75, 0.98), 2)
        self.bbox = random_bbox()

        wx, wy = random_world_pos()
        self.world_x = wx
        self.world_y = wy

        self.anomaly_type = anomaly_type

        if anomaly_type == "STOPPED_VEHICLE":
            self.speed_kmh = round(random.uniform(0.0, 1.5), 1)
        elif anomaly_type == "WRONG_WAY":
            self.speed_kmh = round(random.uniform(40.0, 90.0), 1)
        elif anomaly_type == "LANE_VIOLATION":
            self.speed_kmh = round(random.uniform(60.0, 120.0), 1)
            # Şerit dışı pozisyon
            self.world_x = round(random.uniform(-5.0, -2.0), 2)
        else:
            self.speed_kmh = round(random.uniform(*NORMAL_SPEED_RANGE), 1)

        self.created_at = time.time()
        self.ttl = random.uniform(3.0, 15.0) if anomaly_type is None else random.uniform(5.0, 20.0)

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl

    def update(self):
        """Her tick'te küçük rastgele değişimler."""
        if self.anomaly_type == "STOPPED_VEHICLE":
            self.speed_kmh = round(random.uniform(0.0, 1.8), 1)
        elif self.anomaly_type == "WRONG_WAY":
            self.speed_kmh = round(self.speed_kmh + random.uniform(-3.0, 3.0), 1)
            self.world_y = round(self.world_y - random.uniform(0.1, 0.5), 2)  # ters yön
        else:
            self.speed_kmh = round(
                max(0, self.speed_kmh + random.uniform(-5.0, 5.0)), 1
            )

        # Bbox küçük jitter
        self.bbox[0] += random.randint(-3, 3)
        self.bbox[1] += random.randint(-2, 2)

        # Dünya pozisyonu güncelle
        if self.anomaly_type != "STOPPED_VEHICLE":
            self.world_x = round(self.world_x + random.uniform(-0.05, 0.05), 2)
            self.world_y = round(self.world_y + random.uniform(0.1, 0.5), 2)

    def to_detection_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "class": self.class_name,
            "confidence": self.confidence,
            "bbox": list(self.bbox),
            "track_state": "active",
            "speed_kmh": self.speed_kmh,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly scenario generator
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyScenarioGenerator:
    """Döngüsel anomali senaryoları üretir."""

    def __init__(self, sensor_id: str):
        self.sensor_id = sensor_id
        self._scenario_index = 0
        self._active_stopped: list[MockVehicle] = []
        self._active_stopped_start: float = 0.0

    def next_scenario(self, frame_id: int) -> list[dict] | None:
        """Bir sonraki anomali senaryosunu döndürür."""
        scenarios = [
            self._stopped_vehicle,
            self._wrong_way,
            self._lane_violation,
            self._possible_accident,
            self._sudden_brake,
            self._overspeed,
            self._underspeed,
        ]

        scenario_fn = scenarios[self._scenario_index % len(scenarios)]
        self._scenario_index += 1

        return scenario_fn(frame_id)

    def _stopped_vehicle(self, frame_id: int) -> list[dict]:
        v = MockVehicle(anomaly_type="STOPPED_VEHICLE")
        duration = round(random.uniform(3.5, 12.0), 1)
        return [{
            "anomaly_id": f"STOPPED_VEHICLE_{v.track_id}_{frame_id}",
            "type": "STOPPED_VEHICLE",
            "severity": "medium",
            "track_id": v.track_id,
            "class_name": v.class_name,
            "speed_kmh": v.speed_kmh,
            "duration_sec": duration,
            "world_x": v.world_x,
            "world_y": v.world_y,
            "bbox": v.bbox,
            "message": f"Araç (track_id={v.track_id}, {v.class_name}) ana yolda {duration}s duruyor.",
        }]

    def _wrong_way(self, frame_id: int) -> list[dict]:
        v = MockVehicle(anomaly_type="WRONG_WAY")
        duration = round(random.uniform(2.5, 8.0), 1)
        return [{
            "anomaly_id": f"WRONG_WAY_{v.track_id}_{frame_id}",
            "type": "WRONG_WAY",
            "severity": "high",
            "track_id": v.track_id,
            "class_name": v.class_name,
            "speed_kmh": v.speed_kmh,
            "duration_sec": duration,
            "world_x": v.world_x,
            "world_y": v.world_y,
            "bbox": v.bbox,
            "message": f"Araç (track_id={v.track_id}, {v.class_name}) ters yönde {duration}s ilerliyor.",
        }]

    def _lane_violation(self, frame_id: int) -> list[dict]:
        v = MockVehicle(anomaly_type="LANE_VIOLATION")
        return [{
            "anomaly_id": f"LANE_VIOLATION_{v.track_id}_{frame_id}",
            "type": "LANE_VIOLATION",
            "severity": "medium",
            "track_id": v.track_id,
            "class_name": v.class_name,
            "speed_kmh": v.speed_kmh,
            "duration_sec": 0.0,
            "world_x": v.world_x,
            "world_y": v.world_y,
            "bbox": v.bbox,
            "message": f"Araç (track_id={v.track_id}, {v.class_name}) şerit dışında.",
        }]

    def _possible_accident(self, frame_id: int) -> list[dict]:
        v1 = MockVehicle(anomaly_type="STOPPED_VEHICLE")
        v2 = MockVehicle(anomaly_type="STOPPED_VEHICLE")
        # Yakın konumda olsunlar
        v2.world_x = round(v1.world_x + random.uniform(1.0, 4.0), 2)
        v2.world_y = round(v1.world_y + random.uniform(-2.0, 2.0), 2)

        duration = round(random.uniform(4.0, 15.0), 1)
        proximity = round(math.hypot(v1.world_x - v2.world_x, v1.world_y - v2.world_y), 1)
        ids = sorted([v1.track_id, v2.track_id])

        cx = round((v1.world_x + v2.world_x) / 2, 2)
        cy = round((v1.world_y + v2.world_y) / 2, 2)

        return [{
            "anomaly_id": f"POSSIBLE_ACCIDENT_{ids[0]}_{ids[1]}_{frame_id}",
            "type": "POSSIBLE_ACCIDENT",
            "severity": "critical",
            "involved_track_ids": ids,
            "class_name": v1.class_name,
            "speed_kmh": 0.0,
            "duration_sec": duration,
            "world_x": cx,
            "world_y": cy,
            "bbox": v1.bbox,
            "proximity_m": proximity,
            "message": f"{len(ids)} araç yakın konumda ({proximity}m) {duration}s duruyor. Olası kaza.",
        }]

    def _sudden_brake(self, frame_id: int) -> list[dict]:
        v = MockVehicle()
        speed_before = round(random.uniform(90.0, 130.0), 1)
        speed_after = round(random.uniform(10.0, 40.0), 1)
        decel = round(speed_before - speed_after, 1)
        v.speed_kmh = speed_after
        return [{
            "anomaly_id": f"SUDDEN_BRAKE_{v.track_id}_{frame_id}",
            "type": "SUDDEN_BRAKE",
            "severity": "high",
            "track_id": v.track_id,
            "class_name": v.class_name,
            "speed_kmh": speed_after,
            "duration_sec": 0.0,
            "world_x": v.world_x,
            "world_y": v.world_y,
            "bbox": v.bbox,
            "decel_kmh": decel,
            "message": f"Araç (track_id={v.track_id}, {v.class_name}) ani yavaşlama: {speed_before} → {speed_after} km/h ({decel} km/h düşüş).",
        }]

    def _overspeed(self, frame_id: int) -> list[dict]:
        v = MockVehicle()
        v.speed_kmh = round(random.uniform(142.0, 190.0), 1)
        over_by = round(v.speed_kmh - 140.0, 1)
        return [{
            "anomaly_id": f"OVERSPEED_{v.track_id}_{frame_id}",
            "type": "OVERSPEED",
            "severity": "high",
            "track_id": v.track_id,
            "class_name": v.class_name,
            "speed_kmh": v.speed_kmh,
            "duration_sec": 0.0,
            "world_x": v.world_x,
            "world_y": v.world_y,
            "bbox": v.bbox,
            "message": f"Araç (track_id={v.track_id}, {v.class_name}) hız limiti aşımı: {v.speed_kmh} km/h (limit: 140, +{over_by} km/h).",
        }]

    def _underspeed(self, frame_id: int) -> list[dict]:
        v = MockVehicle()
        v.speed_kmh = round(random.uniform(15.0, 38.0), 1)
        duration = round(random.uniform(3.5, 10.0), 1)
        return [{
            "anomaly_id": f"UNDERSPEED_{v.track_id}_{frame_id}",
            "type": "UNDERSPEED",
            "severity": "medium",
            "track_id": v.track_id,
            "class_name": v.class_name,
            "speed_kmh": v.speed_kmh,
            "duration_sec": duration,
            "world_x": v.world_x,
            "world_y": v.world_y,
            "bbox": v.bbox,
            "message": f"Araç (track_id={v.track_id}, {v.class_name}) asgari hız ihlali: {v.speed_kmh} km/h, {duration}s boyunca limit altında (40 km/h).",
        }]


# ─────────────────────────────────────────────────────────────────────────────
# Main publisher loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Mock MQTT publisher — telemetri + anomali senaryoları",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--sensor-id", default=DEFAULT_SENSOR_ID)
    parser.add_argument("--mock-anomaly", action="store_true",
                        help="Anomali senaryolarını da yayınla")
    parser.add_argument("--anomaly-only", action="store_true",
                        help="Sadece anomali yayınla (telemetri olmadan)")
    parser.add_argument("--anomaly-interval", type=float, default=ANOMALY_INTERVAL_SEC,
                        help="Anomali senaryoları arası süre (saniye)")
    args = parser.parse_args()

    if args.anomaly_only and not args.mock_anomaly:
        print("[hata] --anomaly-only, --mock-anomaly ile birlikte kullanılmalı")
        return

    # MQTT bağlantısı
    client = mqtt.Client(
        client_id=f"mock-{args.sensor_id}",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )

    def on_connect(c, userdata, flags, reason_code, properties):
        if reason_code == 0:
            print(f"[mqtt] Bağlandı: {args.host}:{args.port}")
        else:
            print(f"[mqtt] Bağlantı hatası: {reason_code}")

    client.on_connect = on_connect

    try:
        client.connect(args.host, args.port)
    except Exception as e:
        print(f"[hata] MQTT bağlantısı başarısız: {e}")
        return

    client.loop_start()

    # Topic'ler
    det_topic = f"highway/telemetry/{args.sensor_id}/detections"
    stats_topic = f"highway/telemetry/{args.sensor_id}/stats"
    status_topic = f"highway/sensors/{args.sensor_id}/status"
    anomaly_topic = f"highway/anomalies/{args.sensor_id}/detections"

    # Online status
    time.sleep(0.5)
    client.publish(status_topic, json.dumps({
        "schema_version": SCHEMA_VERSION,
        "sensor_id": args.sensor_id,
        "status": "online",
        "ts_utc": utcnow_iso(),
    }), qos=1, retain=True)

    # Simülasyon state
    vehicles: list[MockVehicle] = []
    anomaly_gen = AnomalyScenarioGenerator(args.sensor_id)
    frame_id = 0
    last_anomaly_time = time.time()
    det_count = 0
    anomaly_count = 0

    mode = "telemetri + anomali" if args.mock_anomaly else "sadece telemetri"
    if args.anomaly_only:
        mode = "sadece anomali"

    print(f"\n[mock] Mod: {mode}")
    print(f"[mock] Telemetri topic: {det_topic}")
    if args.mock_anomaly:
        print(f"[mock] Anomali topic:   {anomaly_topic}")
        print(f"[mock] Anomali aralığı: {args.anomaly_interval}s")
    print(f"[mock] Ctrl+C ile durdur\n")

    try:
        while True:
            frame_id += 1
            now = time.time()

            # ── Normal telemetri ──
            if not args.anomaly_only:
                # Araç sayısını dalgalandır
                target = random.randint(*NORMAL_VEHICLE_COUNT)
                while len(vehicles) < target:
                    vehicles.append(MockVehicle())

                # Süresi dolan araçları sil, kalanları güncelle
                vehicles = [v for v in vehicles if not v.is_expired()]
                for v in vehicles:
                    v.update()

                detections = [v.to_detection_dict() for v in vehicles]

                payload = {
                    "schema_version": SCHEMA_VERSION,
                    "sensor_id": args.sensor_id,
                    "ts_utc": utcnow_iso(),
                    "frame_id": frame_id,
                    "fps": round(random.uniform(27.0, 30.0), 1),
                    "objects": detections,
                }

                client.publish(det_topic, json.dumps(payload, separators=(",", ":")))
                det_count += 1

                # Stats (1Hz)
                if frame_id % 5 == 0:
                    stats_payload = {
                        "schema_version": SCHEMA_VERSION,
                        "sensor_id": args.sensor_id,
                        "ts_utc": utcnow_iso(),
                        "fps": round(random.uniform(28.0, 30.0), 2),
                        "queue_size": 0,
                        "mqtt_connected": True,
                        "published": det_count,
                        "dropped": 0,
                        "events_published": anomaly_count,
                        "tracks_confirmed": len(vehicles),
                        "tracks_lost": 0,
                        "tracks_tentative": random.randint(0, 2),
                        "tracks_active_total": len(vehicles) + random.randint(0, 2),
                    }
                    client.publish(stats_topic, json.dumps(stats_payload, separators=(",", ":")))

            # ── Anomali senaryoları ──
            if args.mock_anomaly and (now - last_anomaly_time) >= args.anomaly_interval:
                anomalies = anomaly_gen.next_scenario(frame_id)
                if anomalies:
                    anomaly_payload = {
                        "schema_version": SCHEMA_VERSION,
                        "sensor_id": args.sensor_id,
                        "ts_utc": utcnow_iso(),
                        "frame_id": frame_id,
                        "anomalies": anomalies,
                    }
                    client.publish(
                        anomaly_topic,
                        json.dumps(anomaly_payload, separators=(",", ":")),
                        qos=1,
                    )
                    anomaly_count += len(anomalies)

                    a = anomalies[0]
                    print(
                        f"[{time.strftime('%H:%M:%S')}] "
                        f"⚠ {a['type']} "
                        f"severity={a['severity']} "
                        f"{'track=' + str(a.get('track_id', '-')):12s} "
                        f"dur={a['duration_sec']}s "
                        f"| total: det={det_count} anomaly={anomaly_count}"
                    )

                last_anomaly_time = now

            # Normal log (10 saniyede bir)
            if frame_id % 50 == 0 and not args.anomaly_only:
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"frame={frame_id:6d} vehicles={len(vehicles):2d} "
                    f"det_published={det_count} anomaly_published={anomaly_count}"
                )

            time.sleep(PUBLISH_INTERVAL_SEC)

    except KeyboardInterrupt:
        print(f"\n[mock] Durduruluyor...")
        print(f"[mock] Toplam: frame={frame_id} det={det_count} anomaly={anomaly_count}")

    finally:
        client.publish(status_topic, json.dumps({
            "schema_version": SCHEMA_VERSION,
            "sensor_id": args.sensor_id,
            "status": "offline",
            "ts_utc": utcnow_iso(),
        }), qos=1, retain=True)
        time.sleep(0.3)
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
