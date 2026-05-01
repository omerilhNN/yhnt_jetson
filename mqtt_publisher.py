#!/usr/bin/env python3
"""
MQTT Publisher — Highway detection telemetri yayıncısı

Pad probe içinden çağrılan publish() fonksiyonu non-blocking — ana pipeline'ı
yavaşlatmaz. Network thread arkada queue'yu okuyup broker'a gönderir.

Bağlantı koparsa:
- Pipeline çalışmaya devam eder, mesajlar local queue'da birikir
- Queue dolarsa eski mesajlar düşer (drop-oldest, LIFO değil — geçmiş daha az değerli)
- Bağlantı dönünce birikmiş mesajları gönderir

Kullanım:
    publisher = MqttPublisher(host="localhost", port=1883, sensor_id="jetson01")
    publisher.start()
    publisher.publish_detections(frame_id, detections)
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


class MqttPublisher:
    """
    Thread-safe MQTT publisher with bounded local queue.

    Args:
        host: Broker hostname/IP. "localhost" veya "192.168.0.42" gibi.
        port: Broker portu. Default 1883.
        sensor_id: Bu Jetson'ın benzersiz kimliği. Topic'te kullanılır.
        topic_prefix: Topic prefix'i, sensor_id ile birleşir.
                      Sonuç topic: <prefix>/<sensor_id>
        max_queue_size: Bağlantı koptuğunda biriktirilecek max mesaj sayısı.
                        Aşılırsa eski mesajlar düşer.
        client_id: MQTT client ID. None ise sensor_id kullanılır.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 1883,
        sensor_id: str = "jetson01",
        topic_prefix: str = "highway/detections",
        max_queue_size: int = 1000,
        client_id: str | None = None,
    ):
        self.host = host
        self.port = port
        self.sensor_id = sensor_id
        self.topic = f"{topic_prefix}/{sensor_id}"
        self.client_id = client_id or sensor_id

        # Thread-safe deque (drop-oldest semantik için maxlen)
        self._queue: deque = deque(maxlen=max_queue_size)
        self._queue_lock = threading.Lock()

        # MQTT istemcisi — paho-mqtt kendi network thread'ini açar
        self._client = mqtt.Client(
            client_id=self.client_id,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)

        # Worker thread — queue'yu boşaltıp publish eder
        self._stop_flag = threading.Event()
        self._worker: threading.Thread | None = None

        # İstatistik
        self._connected = False
        self._stats = {
            "published": 0,
            "queued": 0,
            "dropped": 0,
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
        """Worker'ı durdur, broker bağlantısını kapat."""
        self._stop_flag.set()
        if self._worker:
            self._worker.join(timeout=2.0)
        self._client.loop_stop()
        self._client.disconnect()

        s = self._stats
        print(
            f"[mqtt] Publisher kapatıldı. "
            f"Yayınlanan: {s['published']}, Düşen: {s['dropped']}, "
            f"Bağlanma sayısı: {s['connect_count']}"
        )

    def publish_detections(self, frame_id: int, fps: float, detections: list[dict]):
        """
        Pad probe'tan çağrılır. Non-blocking — sadece queue'ya yazıp döner.

        Args:
            frame_id: Frame numarası (DeepStream frame_meta.frame_num)
            fps: Anlık FPS (loglama için)
            detections: list of dict, her biri:
                {
                    "track_id": 123,
                    "class": "car",
                    "confidence": 0.91,
                    "bbox": [x, y, w, h]
                }
        """
        payload = {
            "schema_version": SCHEMA_VERSION,
            "sensor_id": self.sensor_id,
            "ts_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "frame_id": frame_id,
            "fps": round(fps, 1),
            "objects": detections,
        }

        with self._queue_lock:
            # deque maxlen'e ulaştıysa otomatik en eskisi düşer
            if len(self._queue) == self._queue.maxlen:
                self._stats["dropped"] += 1
            self._queue.append(payload)
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
        """
        while not self._stop_flag.is_set():
            if not self._connected:
                time.sleep(0.1)
                continue

            # Queue'dan al
            payload = None
            with self._queue_lock:
                if self._queue:
                    payload = self._queue.popleft()

            if payload is None:
                time.sleep(0.01)  # Boş queue, kısa bekle
                continue

            # Publish
            try:
                msg = json.dumps(payload, separators=(",", ":"))
                result = self._client.publish(self.topic, msg, qos=0)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    self._stats["published"] += 1
                else:
                    # Geri queue'ya at, sonra dene
                    with self._queue_lock:
                        self._queue.appendleft(payload)
                    time.sleep(0.1)
            except Exception as e:
                print(f"[mqtt] Publish hatası: {e}")
                time.sleep(0.5)
