#!/usr/bin/env python3
import json
import platform
import threading
import time
from collections import deque
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

SCHEMA_VERSION = 1


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class MqttPublisher:
    def __init__(
        self,
        host: str = "100.84.29.29",
        port: int = 1883,
        sensor_id: str = "jetson01",
        detections_prefix: str = "highway/telemetry",
        max_queue_size: int = 1000,
        client_id: str | None = None,
        heartbeat_interval_sec: float = 5.0,
        detections_hz: float = 5.0,
    ):
        self.host = host
        self.port = port
        self.sensor_id = sensor_id
        self.client_id = client_id or sensor_id
        self.heartbeat_interval_sec = heartbeat_interval_sec

        # Topicler
        self.detections_topic = f"{detections_prefix}/{sensor_id}/detections"  # korunuyor
        self.stats_topic = f"highway/telemetry/{sensor_id}/stats"

        self.sensor_status_topic = f"highway/sensors/{sensor_id}/status"
        self.sensor_meta_topic = f"highway/sensors/{sensor_id}/meta"
        self.sensor_heartbeat_topic = f"highway/sensors/{sensor_id}/heartbeat"

        self.event_vehicle_enter_topic = f"highway/events/{sensor_id}/vehicle/enter"
        self.event_vehicle_exit_topic = f"highway/events/{sensor_id}/vehicle/exit"

        self.cmd_request_topic = f"highway/commands/{sensor_id}/request"
        self.cmd_response_topic = f"highway/commands/{sensor_id}/response"

        # Queue
        self._queue: deque = deque(maxlen=max_queue_size)
        self._queue_lock = threading.Lock()

        # MQTT client
        self._client = mqtt.Client(
            client_id=self.client_id,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            clean_session=True,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.reconnect_delay_set(min_delay=1, max_delay=30)

        # LWT
        offline_payload = json.dumps({
            "schema_version": SCHEMA_VERSION,
            "sensor_id": self.sensor_id,
            "status": "offline",
            "ts_utc": _utcnow_iso(),
        })
        self._client.will_set(self.sensor_status_topic, offline_payload, qos=1, retain=True)

        self._stop_flag = threading.Event()
        self._worker: threading.Thread | None = None
        self._heartbeat_worker: threading.Thread | None = None

        self._connected = False
        self._stats = {
            "published": 0,
            "queued": 0,
            "dropped": 0,
            "events_published": 0,
            "connect_count": 0,
            "disconnect_count": 0,
            "commands_received": 0,
        }

        # rate-limit + delta
        self._min_publish_interval = 1.0 / max(1e-6, detections_hz)
        self._last_det_publish_ts = 0.0
        self._last_det_signature = None

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def start(self):
        try:
            self._client.connect_async(self.host, self.port, keepalive=60)
        except Exception as e:
            print(f"[mqtt] Bağlantı hatası: {e} (yeniden denenecek)")

        self._client.loop_start()

        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="mqtt-worker")
        self._worker.start()

        self._heartbeat_worker = threading.Thread(target=self._heartbeat_loop, daemon=True, name="mqtt-heartbeat")
        self._heartbeat_worker.start()

        print(f"[mqtt] Publisher başlatıldı: {self.host}:{self.port} → {self.detections_topic}")

    def stop(self):
        if self._connected:
            try:
                self._client.publish(
                    self.sensor_status_topic,
                    json.dumps({
                        "schema_version": SCHEMA_VERSION,
                        "sensor_id": self.sensor_id,
                        "status": "offline",
                        "ts_utc": _utcnow_iso(),
                    }),
                    qos=1,
                    retain=True,
                )
                time.sleep(0.2)
            except Exception:
                pass

        self._stop_flag.set()

        if self._worker:
            self._worker.join(timeout=2.0)
        if self._heartbeat_worker:
            self._heartbeat_worker.join(timeout=2.0)

        self._client.loop_stop()
        self._client.disconnect()

    def publish_detections(
        self,
        frame_id: int,
        fps: float,
        detections: list[dict],             # ACTIVE-ONLY
        event_objects: list[dict] | None = None,
    ):
        ts = _utcnow_iso()
        event_objects = event_objects or []

        enters = [d for d in event_objects if d.get("track_state") == "enter"]
        exits = [d for d in event_objects if d.get("track_state") == "exit"]

        # Eventleri QoS1 gönder
        for e in enters:
            self._enqueue({
                "_topic": self.event_vehicle_enter_topic,
                "_qos": 1,
                "_payload": {
                    "schema_version": SCHEMA_VERSION,
                    "sensor_id": self.sensor_id,
                    "ts_utc": ts,
                    "frame_id": frame_id,
                    "object": e,
                },
            })

        for e in exits:
            self._enqueue({
                "_topic": self.event_vehicle_exit_topic,
                "_qos": 1,
                "_payload": {
                    "schema_version": SCHEMA_VERSION,
                    "sensor_id": self.sensor_id,
                    "ts_utc": ts,
                    "frame_id": frame_id,
                    "object": e,
                },
            })

        # detections için rate-limit + delta
        now = time.time()
        sig = tuple(sorted(
            (int(o.get("track_id", -1)), o.get("class"), tuple(o.get("bbox", [0, 0, 0, 0])))
            for o in detections
        ))
        has_events = bool(enters or exits)
        interval_ok = (now - self._last_det_publish_ts) >= self._min_publish_interval
        changed = (sig != self._last_det_signature)

        if (not has_events) and ((not interval_ok) or (not changed)):
            return

        telemetry_payload = {
            "schema_version": SCHEMA_VERSION,
            "sensor_id": self.sensor_id,
            "ts_utc": ts,
            "frame_id": frame_id,
            "fps": round(float(fps), 1),
            "objects": detections,
        }
        self._enqueue({
            "_topic": self.detections_topic,
            "_qos": 0,
            "_payload": telemetry_payload,
        })

        self._last_det_publish_ts = now
        self._last_det_signature = sig

    def publish_stats(self, fps: float, queue_size: int, extra: dict | None = None):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "sensor_id": self.sensor_id,
            "ts_utc": _utcnow_iso(),
            "fps": round(float(fps), 2),
            "queue_size": int(queue_size),
            "mqtt_connected": self._connected,
            "published": self._stats["published"],
            "dropped": self._stats["dropped"],
            "events_published": self._stats["events_published"],
        }
        if extra:
            payload.update(extra)

        self._enqueue({
            "_topic": self.stats_topic,
            "_qos": 0,
            "_payload": payload,
        })

    def publish_meta(self):
        payload = {
            "schema_version": SCHEMA_VERSION,
            "sensor_id": self.sensor_id,
            "ts_utc": _utcnow_iso(),
            "device": {
                "hostname": platform.node(),
                "platform": platform.platform(),
            },
            "capabilities": {
                "detections": True,
                "events_enter_exit": True,
                "commands": True,
                "stats": True,
                "heartbeat": True,
            },
        }
        self._enqueue({
            "_topic": self.sensor_meta_topic,
            "_qos": 1,
            "_retain": True,
            "_payload": payload,
        })

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def queue_size(self) -> int:
        with self._queue_lock:
            return len(self._queue)

    def get_stats(self) -> dict:
        return dict(self._stats)

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    def _heartbeat_loop(self):
        while not self._stop_flag.is_set():
            if self._connected:
                self._enqueue({
                    "_topic": self.sensor_heartbeat_topic,
                    "_qos": 0,
                    "_payload": {
                        "schema_version": SCHEMA_VERSION,
                        "sensor_id": self.sensor_id,
                        "ts_utc": _utcnow_iso(),
                        "alive": True,
                    },
                })
            time.sleep(self.heartbeat_interval_sec)

    def _enqueue(self, item: dict):
        with self._queue_lock:
            if len(self._queue) == self._queue.maxlen:
                self._stats["dropped"] += 1
            self._queue.append(item)
            self._stats["queued"] += 1

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            self._connected = True
            self._stats["connect_count"] += 1
            print(f"[mqtt] Bağlandı: {self.host}:{self.port}")

            # online status
            client.publish(
                self.sensor_status_topic,
                json.dumps({
                    "schema_version": SCHEMA_VERSION,
                    "sensor_id": self.sensor_id,
                    "status": "online",
                    "ts_utc": _utcnow_iso(),
                }),
                qos=1,
                retain=True,
            )

            # meta retained
            self.publish_meta()

            # commands request subscribe
            client.subscribe(self.cmd_request_topic, qos=1)
        else:
            print(f"[mqtt] Bağlantı reddedildi: {reason_code}")

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        self._connected = False
        self._stats["disconnect_count"] += 1
        if reason_code != 0:
            print(f"[mqtt] Bağlantı koptu (reason={reason_code}), yeniden denenecek")

    def _on_message(self, client, userdata, msg):
        try:
            self._stats["commands_received"] += 1
            req = json.loads(msg.payload.decode("utf-8")) if msg.payload else {}
            resp = {
                "schema_version": SCHEMA_VERSION,
                "sensor_id": self.sensor_id,
                "ts_utc": _utcnow_iso(),
                "correlation_id": req.get("correlation_id"),
                "command": req.get("command", "unknown"),
                "ok": True,
                "message": "accepted",
            }
            self._enqueue({
                "_topic": self.cmd_response_topic,
                "_qos": 1,
                "_payload": resp,
            })
        except Exception as e:
            print(f"[mqtt] Command parse hatası: {e}")

    def _worker_loop(self):
        while not self._stop_flag.is_set():
            if not self._connected:
                time.sleep(0.1)
                continue

            item = None
            with self._queue_lock:
                if self._queue:
                    item = self._queue.popleft()

            if item is None:
                time.sleep(0.01)
                continue

            try:
                topic = item["_topic"]
                qos = item.get("_qos", 0)
                retain = item.get("_retain", False)
                payload = json.dumps(item["_payload"], separators=(",", ":"))

                result = self._client.publish(topic, payload, qos=qos, retain=retain)
                if result.rc == mqtt.MQTT_ERR_SUCCESS:
                    if topic.startswith(f"highway/events/{self.sensor_id}/"):
                        self._stats["events_published"] += 1
                    else:
                        self._stats["published"] += 1
                else:
                    with self._queue_lock:
                        self._queue.appendleft(item)
                    time.sleep(0.1)
            except Exception as e:
                print(f"[mqtt] Publish hatası: {e}")
                time.sleep(0.5)