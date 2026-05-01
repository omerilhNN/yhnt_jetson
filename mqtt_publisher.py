#!/usr/bin/env python3
"""
MQTT Publisher — Highway detection telemetri yayıncısı

Pad probe içinden çağrılan publish() fonksiyonu non-blocking — ana pipeline'ı
yavaşlatmaz. Network thread arkada queue'yu okuyup broker'a gönderir.

Bağlantı koparsa:
- Pipeline çalışmaya devam eder, mesajlar local queue'da birikir
- Queue dolarsa eski mesajlar düşer (drop-oldest, LIFO değil — geçmiş daha az değerli)
- Bağlantı dönünce birikmiş mesajları gönderir

Tailscale senaryosu:
- Jetson (100.74.245.10) → RPi5 broker (100.84.29.29:1883)
- Bağlantıyı her zaman Jetson başlatır (outbound TCP)
- Tailscale tünelinde port açmaya gerek yok

Topic yapısı:
- highway/detections/<sensor_id>   → telemetri (QoS 0, yüksek frekans)
- highway/status/<sensor_id>       → online/offline (QoS 1, retained, LWT)
- highway/events/<sensor_id>       → exit/anomaly eventleri (QoS 1)

Kullanım:
    publisher = MqttPublisher(
        host="100.84.29.29",
        port=1883,
        sensor_id="yhnt-jetson-01",
    )
    publisher.start()
    publisher.publish_detections(frame_id, fps, detections)
    # ...
    publisher.stop()
"""

import json
import threading
import time
from collections import deque
from datetime import datetime, timezone

import paho.mqtt.client as mqtt


SCHEMA_VERSION = 1


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class MqttPublisher:
    """
    Thread-safe MQTT publisher with bounded local queue.

    Args:
        host: Broker hostname/IP. Tailscale senaryosunda RPi5'in tailnet IP'si.
              Örn: "100.84.29.29" veya MagicDNS ile "rpi5-server".
        port: Broker portu. Default 1883.
        sensor_id: Bu Jetson'ın benzersiz kimliği. Topic'te kullanılır.
        topic_prefix: Telemetri topic prefix'i. Sonuç: <prefix>/<sensor_id>
        status_prefix: Status (online/offline) topic prefix'i.
        events_prefix: Event (exit/anomaly) topic prefix'i.
        max_queue_size: Bağlantı koptuğunda biriktirilecek max mesaj sayısı.
                        Aşılırsa eski mesajlar düşer.
        client_id: MQTT client ID. None ise sensor_id kullanılır.
    """

    def __init__(
        self,
        host: str = "100.84.29.29",
        port: int = 1883,
        sensor_id: str = "yhnt-jetson-01",
        topic_prefix: str = "highway/telemetry",
        status_prefix: str = "highway/status",
        events_prefix: str = "highway/events",
        max_queue_size: int = 1000,
        client_id: str | None = None,
    ):
        self.host = host
        self.port = port
        self.sensor_id = sensor_id
        self.topic = f"{topic_prefix}/{sensor_id}/detections"
        self.status_topic = f"{status_prefix}/{sensor_id}"
        self.events_topic = f"{events_prefix}/{sensor_id}"
        self.client_id = client_id or sensor_id

        # Thread-safe deque (drop-oldest semantik için maxlen)
        self._queue: deque = deque(maxlen=max_queue_size)
        self._queue_lock = threading.Lock()

        # MQTT istemcisi — paho-mqtt kendi network thread'ini açar
        self._client = mqtt.Client(
            client_id=self.client_id,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            clean_session=True,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        # Tailscale tüneli koparsa hızlı geri bağlan (1s..30s arası exponential)
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)

        # LWT — Jetson aniden ölürse broker bunu yayınlar
        offline_payload = json.dumps({
            "schema_version": SCHEMA_VERSION,
            "sensor_id": self.sensor_id,
            "status": "offline",
            "ts_utc": _utcnow_iso(),
        })
        self._client.will_set(
            self.status_topic,
            offline_payload,
            qos=1,
            retain=True,
        )

        # Worker thread — queue'yu boşaltıp publish eder
        self._stop_flag = threading.Event()
        self._worker: threading.Thread | None = None

        # İstatistik
        self._connected = False
        self._stats = {
            "published": 0,
            "queued": 0,
            "dropped": 0,
            "events_published": 0,
            "connect_count": 0,
            "disconnect_count": 0,
        }

    # ─── Public API ───────────────────────────────────────────────────────────

    def start(self):
        """Broker'a bağlan, worker thread'i başlat."""
        try:
            self._client.connect_async(self.host, self.port, keepalive=60)
        except Exception as e:
            # Hostname çözülemese bile başlatmaya devam — reconnect denemesi
            # yapacak. Bu, broker daha gelmediyse pipeline'ı durdurmuyor.
            print(f"[mqtt] Bağlantı hatası: {e} (yeniden denenecek)")

        # paho-mqtt'nin kendi network thread'i — non-blocking
        self._client.loop_start()

        # Bizim worker thread'imiz — queue → publish
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="mqtt-worker")
        self._worker.start()

        print(f"[mqtt] Publisher başlatıldı: {self.host}:{self.port} → {self.topic}")

    def stop(self):
        """
        Worker'ı durdur, broker bağlantısını kapat.
        Düzgün kapanışta retained 'offline' mesajı yayınlar (LWT'nin manuel karşılığı).
        """
        # Düzgün kapanış: retained offline yayınla
        if self._connected:
            try:
                offline_payload = json.dumps({
                    "schema_version": SCHEMA_VERSION,
                    "sensor_id": self.sensor_id,
                    "status": "offline",
                    "ts_utc": _utcnow_iso(),
                })
                self._client.publish(self.status_topic, offline_payload, qos=1, retain=True)
                # Mesajın gönderilmesi için kısa bekle
                time.sleep(0.2)
            except Exception as e:
                print(f"[mqtt] Offline status yayınlanamadı: {e}")

        self._stop_flag.set()
        if self._worker:
            self._worker.join(timeout=2.0)
        self._client.loop_stop()
        self._client.disconnect()

        s = self._stats
        print(
            f"[mqtt] Publisher kapatıldı. "
            f"Yayınlanan: {s['published']}, Event: {s['events_published']}, "
            f"Düşen: {s['dropped']}, Bağlanma sayısı: {s['connect_count']}"
        )

    def publish_detections(self, frame_id: int, fps: float, detections: list[dict]):
        """
        Pad probe'tan çağrılır. Non-blocking — sadece queue'ya yazıp döner.

        Exit event'leri varsa onları ayrı bir QoS=1 mesaj olarak da queue'lar
        (events topic'ine), kalan kısmı normal telemetri olarak gider.

        Args:
            frame_id: Frame numarası (DeepStream frame_meta.frame_num)
            fps: Anlık FPS (loglama için)
            detections: list of dict, her biri:
                {
                    "track_id": 123,
                    "class": "car",
                    "confidence": 0.91,
                    "bbox": [x, y, w, h],
                    "track_state": "active" | "exit"
                }
        """
        ts = _utcnow_iso()

        # Exit event'leri ayrıştır → kayıpsız (QoS=1) ayrı topic'e
        exits = [d for d in detections if d.get("track_state") == "exit"]
        if exits:
            event_payload = {
                "schema_version": SCHEMA_VERSION,
                "sensor_id": self.sensor_id,
                "ts_utc": ts,
                "frame_id": frame_id,
                "exits": exits,
            }
            self._enqueue({
                "_topic": self.events_topic,
                "_qos": 1,
                "_payload": event_payload,
            })

        # Tüm objeler (active + exit) telemetri stream'ine de gider (QoS=0)
        telemetry_payload = {
            "schema_version": SCHEMA_VERSION,
            "sensor_id": self.sensor_id,
            "ts_utc": ts,
            "frame_id": frame_id,
            "fps": round(fps, 1),
            "objects": detections,
        }
        self._enqueue({
            "_topic": self.topic,
            "_qos": 0,
            "_payload": telemetry_payload,
        })

    def _enqueue(self, item: dict):
        with self._queue_lock:
            if len(self._queue) == self._queue.maxlen:
                self._stats["dropped"] += 1
            self._queue.append(item)
            self._stats["queued"] += 1

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def queue_size(self) -> int:
        with self._queue_lock:
            return len(self._queue)

    def get_stats(self) -> dict:
        return dict(self._stats)

    # ─── Internal callbacks ────────────────────────────────────────────────────

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self._connected = True
            self._stats["connect_count"] += 1
            print(f"[mqtt] Bağlandı: {self.host}:{self.port}")

            # Online status (retained) — dashboard anlık görsün
            try:
                online_payload = json.dumps({
                    "schema_version": SCHEMA_VERSION,
                    "sensor_id": self.sensor_id,
                    "status": "online",
                    "ts_utc": _utcnow_iso(),
                })
                client.publish(self.status_topic, online_payload, qos=1, retain=True)
            except Exception as e:
                print(f"[mqtt] Online status yayınlanamadı: {e}")
        else:
            print(f"[mqtt] Bağlantı reddedildi: {reason_code}")

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        self._connected = False
        self._stats["disconnect_count"] += 1
        if reason_code != 0:
            print(f"[mqtt] Bağlantı koptu (reason={reason_code}), yeniden denenecek")

    # ─── Worker loop ──────────────────────────────────────────────────────────

    def _worker_loop(self):
        """
        Queue'yu boşaltır. Bağlı değilse mesajları queue'da tutar (deque
        maxlen sayesinde otomatik drop-oldest olur).

        Her item: {"_topic": str, "_qos": int, "_payload": dict}
        """
        while not self._stop_flag.is_set():
            if not self._connected:
                time.sleep(0.1)
                continue

            # Queue'dan al
            item = None
            with self._queue_lock:
                if self._queue:
                    item = self._queue.popleft()

            if item is None:
                time.sleep(0.01)  # Boş queue, kısa bekle
                continue

            # Publish
            try:
                topic = item["_topic"]
                qos = item["_qos"]
                payload = item["_payload"]
                msg = json.dumps(payload, separators=(",", ":"))

                result = self._client.publish(topic, msg, qos=qos)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    if topic == self.events_topic:
                        self._stats["events_published"] += 1
                    else:
                        self._stats["published"] += 1
                else:
                    # Geri queue'ya at, sonra dene
                    with self._queue_lock:
                        self._queue.appendleft(item)
                    time.sleep(0.1)
            except Exception as e:
                print(f"[mqtt] Publish hatası: {e}")
                time.sleep(0.5)