#!/usr/bin/env python3
"""
Highway Anomaly Detection Module

Bağımsız, hafif anomali tespit motoru.
GStreamer/DeepStream bilmez — sadece track verisi alır, anomali döndürür.

Anomali tipleri:
  - STOPPED_VEHICLE    : Ana yolda belirli süre duran araç
  - WRONG_WAY          : Ters yöne giden araç
  - LANE_VIOLATION     : Şerit ihlali (tanımlı şerit poligonları varsa)
  - POSSIBLE_ACCIDENT  : Yakın konumda birden fazla duran araç

Kullanım:
    from anomaly_detector import AnomalyDetector, AnomalyConfig

    config = AnomalyConfig()
    detector = AnomalyDetector(config, sensor_id="jetson01")

    # Her frame'de:
    anomalies = detector.update(frame_id, timestamp, tracks)
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyType(str, Enum):
    STOPPED_VEHICLE = "STOPPED_VEHICLE"
    WRONG_WAY = "WRONG_WAY"
    LANE_VIOLATION = "LANE_VIOLATION"
    POSSIBLE_ACCIDENT = "POSSIBLE_ACCIDENT"
    SUDDEN_BRAKE = "SUDDEN_BRAKE"
    OVERSPEED = "OVERSPEED"
    UNDERSPEED = "UNDERSPEED"


class AnomalySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


ANOMALY_SEVERITY_MAP = {
    AnomalyType.STOPPED_VEHICLE: AnomalySeverity.MEDIUM,
    AnomalyType.WRONG_WAY: AnomalySeverity.HIGH,
    AnomalyType.LANE_VIOLATION: AnomalySeverity.MEDIUM,
    AnomalyType.POSSIBLE_ACCIDENT: AnomalySeverity.CRITICAL,
    AnomalyType.SUDDEN_BRAKE: AnomalySeverity.HIGH,
    AnomalyType.OVERSPEED: AnomalySeverity.HIGH,
    AnomalyType.UNDERSPEED: AnomalySeverity.MEDIUM,
}


class TrackInfo(NamedTuple):
    """Probe'dan anomaly detector'a aktarılan per-object verisi."""
    track_id: int
    class_name: str
    speed_kmh: float
    world_x: float
    world_y: float
    bbox: list  # [x, y, w, h]


class TrackSample(NamedTuple):
    """Track geçmişinde tutulan tek bir ölçüm."""
    timestamp: float
    world_x: float
    world_y: float
    speed_kmh: float


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnomalyConfig:
    """Tüm eşik değerleri burada. Hiçbir magic number yok."""

    # Stopped vehicle
    stopped_speed_threshold_kmh: float = 2.0
    stopped_duration_sec: float = 3.0

    # Wrong-way
    wrong_way_duration_sec: float = 2.0
    wrong_way_angle_threshold_deg: float = 135.0

    # Direction learning: sistem başlangıcında ilk N aracın hareket yönünü
    # gözlemleyerek geçerli trafik akış yönünü öğrenir.
    # Öğrenme tamamlanana kadar WRONG_WAY anomalisi üretilmez.
    direction_learning_vehicle_count: int = 10
    # Bir aracın yön oyu sayılabilmesi için gerekli minimum hareket (metre)
    direction_learning_min_displacement_m: float = 2.0

    # Fallback: öğrenme başarısız olursa veya devre dışıysa kullanılacak yön.
    # Default: +Y yönü (kameradan uzaklaşan trafik, homography P1→P4 yönü).
    fallback_direction_x: float = 0.0
    fallback_direction_y: float = 1.0

    # Lane violation
    # Her poligon: list of (world_x, world_y) tuples.
    # None ise lane violation kontrolü devre dışı.
    lane_polygons: list | None = None

    # Possible accident
    accident_min_stopped_vehicles: int = 2
    accident_proximity_m: float = 15.0
    accident_duration_sec: float = 3.0

    # Sudden brake (ani yavaşlama)
    # N saniye içinde hız düşüşü eşiği (km/h)
    sudden_brake_decel_kmh: float = 40.0   # 40 km/h düşüş → ani fren
    sudden_brake_window_sec: float = 2.0   # bu pencere içinde ölçülür

    # Overspeed (hız limiti aşımı)
    overspeed_threshold_kmh: float = 140.0

    # Underspeed (asgari hız ihlali)
    # Duran araçları tekrar tetiklememek için stopped threshold'dan yüksek tutulur
    underspeed_threshold_kmh: float = 40.0
    underspeed_duration_sec: float = 3.0   # geçici yavaşlamaları filtrele

    # Cooldown: aynı (track_id, anomaly_type) çifti için tekrar yayın süresi
    cooldown_sec: float = 10.0

    # Track history
    track_history_maxlen: int = 90     # ~3 saniye @ 30fps
    track_timeout_sec: float = 5.0     # bu kadar süre güncellenmezse sil

    # Minimum track yaşı (saniye) — çok yeni track'lerde false positive engelle
    min_track_age_sec: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly Event
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnomalyEvent:
    """Tespit edilen tek bir anomali. MQTT payload'a dönüştürülebilir."""
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    frame_id: int
    track_id: int | None               # POSSIBLE_ACCIDENT için None olabilir
    involved_track_ids: list[int] | None
    class_name: str
    speed_kmh: float
    duration_sec: float
    world_x: float
    world_y: float
    bbox: list
    proximity_m: float | None
    decel_kmh: float | None             # SUDDEN_BRAKE: hız düşüşü miktarı
    message: str

    @property
    def anomaly_id(self) -> str:
        if self.involved_track_ids and len(self.involved_track_ids) > 1:
            ids = "_".join(str(i) for i in sorted(self.involved_track_ids))
            return f"{self.anomaly_type.value}_{ids}_{self.frame_id}"
        return f"{self.anomaly_type.value}_{self.track_id}_{self.frame_id}"

    def to_dict(self) -> dict:
        d = {
            "anomaly_id": self.anomaly_id,
            "type": self.anomaly_type.value,
            "severity": self.severity.value,
            "class_name": self.class_name,
            "speed_kmh": round(self.speed_kmh, 1),
            "duration_sec": round(self.duration_sec, 1),
            "world_x": round(self.world_x, 2),
            "world_y": round(self.world_y, 2),
            "bbox": self.bbox,
            "message": self.message,
        }
        if self.track_id is not None:
            d["track_id"] = self.track_id
        if self.involved_track_ids is not None:
            d["involved_track_ids"] = self.involved_track_ids
        if self.proximity_m is not None:
            d["proximity_m"] = round(self.proximity_m, 1)
        if self.decel_kmh is not None:
            d["decel_kmh"] = round(self.decel_kmh, 1)
        return d


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Lightweight, stateful anomaly detector.

    Her frame'de update() çağrılır, 0 veya daha fazla AnomalyEvent döner.
    Jetson'da çalışır — hot path'te alloc yok, O(N) track sayısına göre.
    """

    def __init__(self, config: AnomalyConfig | None = None, sensor_id: str = "jetson01"):
        self.cfg = config or AnomalyConfig()
        self.sensor_id = sensor_id

        # track_id -> deque[TrackSample]
        self._history: dict[int, deque[TrackSample]] = {}

        # track_id -> first seen timestamp
        self._first_seen: dict[int, float] = {}

        # track_id -> latest TrackInfo (current frame için geçici)
        self._current_info: dict[int, TrackInfo] = {}

        # cooldown: (track_id_or_frozenset, AnomalyType) -> last fire timestamp
        self._cooldowns: dict[tuple, float] = {}

        # ── Direction learning state ──
        # İlk N aracın hareket yönünü gözlemleyerek trafik akış yönünü öğren.
        # Öğrenme tamamlanana kadar _learned_dir = None → WRONG_WAY üretilmez.
        self._direction_learned: bool = False
        self._learned_dir: tuple[float, float] | None = None

        # Oy veren track'ler: track_id -> (direction_x, direction_y) unit vector
        # Bir track yeterli displacement'a ulaşınca tek bir oy kullanır.
        self._direction_votes: dict[int, tuple[float, float]] = {}
        # Oy vermiş track id'leri (birden fazla oy engelleme)
        self._direction_voted_tids: set[int] = set()
        # Track başına ilk gözlem noktası (öğrenme fazı için)
        self._direction_first_pos: dict[int, tuple[float, float]] = {}

        # Öğrenme devre dışıysa (vehicle_count <= 0) fallback kullan
        if self.cfg.direction_learning_vehicle_count <= 0:
            self._apply_fallback_direction()
            print(f"[anomaly] Direction learning devre dışı, fallback yön kullanılıyor: {self._learned_dir}")

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def update(
        self,
        frame_id: int,
        timestamp: float,
        tracks: list[TrackInfo],
    ) -> list[AnomalyEvent]:
        """
        Ana giriş noktası. Her frame'de çağır.

        Args:
            frame_id:  mevcut frame numarası
            timestamp: saniye cinsinden zaman (PTS veya wall clock)
            tracks:    bu frame'deki aktif track'ler

        Returns:
            Tespit edilen anomaliler listesi (boş olabilir).
        """
        # 1) Geçmişi güncelle
        active_tids = set()
        self._current_info.clear()

        for t in tracks:
            tid = t.track_id
            active_tids.add(tid)
            self._current_info[tid] = t

            if tid not in self._first_seen:
                self._first_seen[tid] = timestamp

            if tid not in self._history:
                self._history[tid] = deque(maxlen=self.cfg.track_history_maxlen)

            self._history[tid].append(TrackSample(
                timestamp=timestamp,
                world_x=t.world_x,
                world_y=t.world_y,
                speed_kmh=t.speed_kmh,
            ))

        # 2) Stale track'leri temizle
        self._prune_stale_tracks(timestamp, active_tids)

        # 3) Direction learning (henüz öğrenilmediyse)
        if not self._direction_learned:
            self._direction_learning_update(tracks)

        # 4) Anomali kurallarını çalıştır
        anomalies: list[AnomalyEvent] = []

        stopped_tids = self._detect_stopped_vehicles(frame_id, timestamp, anomalies)
        self._detect_wrong_way(frame_id, timestamp, anomalies)
        self._detect_lane_violation(frame_id, timestamp, anomalies)
        self._detect_possible_accident(frame_id, timestamp, stopped_tids, anomalies)
        self._detect_sudden_brake(frame_id, timestamp, anomalies)
        self._detect_overspeed(frame_id, timestamp, anomalies)
        self._detect_underspeed(frame_id, timestamp, anomalies)

        # 5) Stale cooldown'ları temizle (periyodik)
        if frame_id % 300 == 0:
            self._prune_cooldowns(timestamp)

        return anomalies

    # ──────────────────────────────────────────────────────────────────────
    # Detection rules
    # ──────────────────────────────────────────────────────────────────────

    def _detect_stopped_vehicles(
        self,
        frame_id: int,
        timestamp: float,
        out: list[AnomalyEvent],
    ) -> set[int]:
        """Duran araç tespiti. Duran track ID'lerini de döndürür (accident için)."""
        stopped_tids: set[int] = set()
        cfg = self.cfg

        for tid, history in self._history.items():
            if len(history) < 2:
                continue
            if not self._is_track_mature(tid, timestamp):
                continue

            info = self._current_info.get(tid)
            if info is None:
                continue

            # Geçmişteki tüm hız örneklerini kontrol et
            stopped_start = self._find_continuous_stop_start(
                history, cfg.stopped_speed_threshold_kmh
            )
            if stopped_start is None:
                continue

            duration = timestamp - stopped_start
            if duration < cfg.stopped_duration_sec:
                # Henüz eşiği aşmamış ama yine de "duruyor" olarak işaretle
                # (accident detection için)
                if info.speed_kmh <= cfg.stopped_speed_threshold_kmh:
                    stopped_tids.add(tid)
                continue

            stopped_tids.add(tid)

            if self._is_on_cooldown(tid, AnomalyType.STOPPED_VEHICLE, timestamp):
                continue

            self._set_cooldown(tid, AnomalyType.STOPPED_VEHICLE, timestamp)

            out.append(AnomalyEvent(
                anomaly_type=AnomalyType.STOPPED_VEHICLE,
                severity=ANOMALY_SEVERITY_MAP[AnomalyType.STOPPED_VEHICLE],
                frame_id=frame_id,
                track_id=tid,
                involved_track_ids=[tid],
                class_name=info.class_name,
                speed_kmh=info.speed_kmh,
                duration_sec=duration,
                world_x=info.world_x,
                world_y=info.world_y,
                bbox=info.bbox,
                proximity_m=None,
                decel_kmh=None,
                message=(
                    f"Araç (track_id={tid}, {info.class_name}) ana yolda "
                    f"{duration:.1f} saniyedir duruyor."
                ),
            ))

        return stopped_tids

    def _detect_wrong_way(
        self,
        frame_id: int,
        timestamp: float,
        out: list[AnomalyEvent],
    ):
        """
        Ters yön tespiti.
        Öğrenme fazı tamamlanmadan çalışmaz — false positive önleme.
        Öğrenilen trafik yönü ile hareket vektörü karşılaştırılır.
        """
        # Yön henüz öğrenilmediyse WRONG_WAY üretme
        if not self._direction_learned or self._learned_dir is None:
            return

        cfg = self.cfg

        for tid, history in self._history.items():
            if len(history) < 3:
                continue
            if not self._is_track_mature(tid, timestamp):
                continue

            info = self._current_info.get(tid)
            if info is None:
                continue

            # Çok yavaş araçlarda yön hesabı güvenilmez
            if info.speed_kmh < cfg.stopped_speed_threshold_kmh * 1.5:
                continue

            wrong_start = self._find_continuous_wrong_way_start(history)
            if wrong_start is None:
                continue

            duration = timestamp - wrong_start
            if duration < cfg.wrong_way_duration_sec:
                continue

            if self._is_on_cooldown(tid, AnomalyType.WRONG_WAY, timestamp):
                continue

            self._set_cooldown(tid, AnomalyType.WRONG_WAY, timestamp)

            out.append(AnomalyEvent(
                anomaly_type=AnomalyType.WRONG_WAY,
                severity=ANOMALY_SEVERITY_MAP[AnomalyType.WRONG_WAY],
                frame_id=frame_id,
                track_id=tid,
                involved_track_ids=[tid],
                class_name=info.class_name,
                speed_kmh=info.speed_kmh,
                duration_sec=duration,
                world_x=info.world_x,
                world_y=info.world_y,
                bbox=info.bbox,
                proximity_m=None,
                decel_kmh=None,
                message=(
                    f"Araç (track_id={tid}, {info.class_name}) ters yönde "
                    f"{duration:.1f} saniyedir ilerliyor."
                ),
            ))

    def _detect_lane_violation(
        self,
        frame_id: int,
        timestamp: float,
        out: list[AnomalyEvent],
    ):
        """Şerit ihlali: araç tanımlı poligonların dışında."""
        if not self.cfg.lane_polygons:
            return

        for tid, info in self._current_info.items():
            if not self._is_track_mature(tid, timestamp):
                continue

            inside_any = False
            for polygon in self.cfg.lane_polygons:
                if self._point_in_polygon(info.world_x, info.world_y, polygon):
                    inside_any = True
                    break

            if inside_any:
                continue

            if self._is_on_cooldown(tid, AnomalyType.LANE_VIOLATION, timestamp):
                continue

            self._set_cooldown(tid, AnomalyType.LANE_VIOLATION, timestamp)

            out.append(AnomalyEvent(
                anomaly_type=AnomalyType.LANE_VIOLATION,
                severity=ANOMALY_SEVERITY_MAP[AnomalyType.LANE_VIOLATION],
                frame_id=frame_id,
                track_id=tid,
                involved_track_ids=[tid],
                class_name=info.class_name,
                speed_kmh=info.speed_kmh,
                duration_sec=0.0,
                world_x=info.world_x,
                world_y=info.world_y,
                bbox=info.bbox,
                proximity_m=None,
                decel_kmh=None,
                message=(
                    f"Araç (track_id={tid}, {info.class_name}) şerit dışında "
                    f"(world: {info.world_x:.2f}, {info.world_y:.2f})."
                ),
            ))

    def _detect_possible_accident(
        self,
        frame_id: int,
        timestamp: float,
        stopped_tids: set[int],
        out: list[AnomalyEvent],
    ):
        """Olası kaza: yakın konumda birden fazla duran araç."""
        cfg = self.cfg

        if len(stopped_tids) < cfg.accident_min_stopped_vehicles:
            return

        # Durma süresi yeterli olan track'leri filtrele
        qualified: list[tuple[int, TrackInfo, float]] = []  # (tid, info, stop_dur)
        for tid in stopped_tids:
            info = self._current_info.get(tid)
            history = self._history.get(tid)
            if info is None or history is None:
                continue

            stop_start = self._find_continuous_stop_start(
                history, cfg.stopped_speed_threshold_kmh
            )
            if stop_start is None:
                continue

            dur = timestamp - stop_start
            if dur >= cfg.accident_duration_sec:
                qualified.append((tid, info, dur))

        if len(qualified) < cfg.accident_min_stopped_vehicles:
            return

        # Pairwise mesafe kontrolü — küçük N için O(N²) yeterli
        # Cluster bulma: basit greedy — bir aracın yakınındaki tüm araçları topla
        used = set()
        for i in range(len(qualified)):
            if qualified[i][0] in used:
                continue

            cluster_indices = [i]
            tid_i, info_i, _ = qualified[i]

            for j in range(i + 1, len(qualified)):
                if qualified[j][0] in used:
                    continue
                tid_j, info_j, _ = qualified[j]

                dist = math.hypot(
                    info_i.world_x - info_j.world_x,
                    info_i.world_y - info_j.world_y,
                )
                if dist <= cfg.accident_proximity_m:
                    cluster_indices.append(j)

            if len(cluster_indices) < cfg.accident_min_stopped_vehicles:
                continue

            cluster = [qualified[idx] for idx in cluster_indices]
            involved_ids = sorted(c[0] for c in cluster)
            involved_key = frozenset(involved_ids)

            for idx in cluster_indices:
                used.add(qualified[idx][0])

            if self._is_on_cooldown(involved_key, AnomalyType.POSSIBLE_ACCIDENT, timestamp):
                continue

            self._set_cooldown(involved_key, AnomalyType.POSSIBLE_ACCIDENT, timestamp)

            # Cluster merkezi ve ortalama mesafe
            cx = sum(c[1].world_x for c in cluster) / len(cluster)
            cy = sum(c[1].world_y for c in cluster) / len(cluster)
            max_dur = max(c[2] for c in cluster)

            # Cluster içi max mesafe
            max_dist = 0.0
            for a in range(len(cluster)):
                for b in range(a + 1, len(cluster)):
                    d = math.hypot(
                        cluster[a][1].world_x - cluster[b][1].world_x,
                        cluster[a][1].world_y - cluster[b][1].world_y,
                    )
                    max_dist = max(max_dist, d)

            # En yaygın sınıf
            classes = [c[1].class_name for c in cluster]
            dominant_class = max(set(classes), key=classes.count)

            out.append(AnomalyEvent(
                anomaly_type=AnomalyType.POSSIBLE_ACCIDENT,
                severity=ANOMALY_SEVERITY_MAP[AnomalyType.POSSIBLE_ACCIDENT],
                frame_id=frame_id,
                track_id=None,
                involved_track_ids=involved_ids,
                class_name=dominant_class,
                speed_kmh=0.0,
                duration_sec=max_dur,
                world_x=cx,
                world_y=cy,
                bbox=cluster[0][1].bbox,
                proximity_m=max_dist,
                decel_kmh=None,
                message=(
                    f"{len(cluster)} araç yakın konumda ({max_dist:.1f}m) "
                    f"{max_dur:.1f} saniyedir duruyor. "
                    f"Olası kaza. Track ID'ler: {involved_ids}"
                ),
            ))

    def _detect_sudden_brake(
        self,
        frame_id: int,
        timestamp: float,
        out: list[AnomalyEvent],
    ):
        """
        Ani yavaşlama tespiti.
        Pencere içindeki en yüksek hız ile şu anki hız arasındaki fark eşiği aşarsa tetikler.
        """
        cfg = self.cfg

        for tid, history in self._history.items():
            if len(history) < 3:
                continue
            if not self._is_track_mature(tid, timestamp):
                continue

            info = self._current_info.get(tid)
            if info is None:
                continue

            # Pencere içindeki en yüksek hızı bul
            window_start = timestamp - cfg.sudden_brake_window_sec
            max_speed_in_window = 0.0
            for sample in history:
                if sample.timestamp >= window_start:
                    if sample.speed_kmh > max_speed_in_window:
                        max_speed_in_window = sample.speed_kmh

            # Duran araçlar zaten STOPPED_VEHICLE tarafından yakalanır
            # Burada sadece hareket halindeyken ani fren yapanları yakalıyoruz
            if max_speed_in_window < cfg.underspeed_threshold_kmh:
                continue

            decel = max_speed_in_window - info.speed_kmh
            if decel < cfg.sudden_brake_decel_kmh:
                continue

            if self._is_on_cooldown(tid, AnomalyType.SUDDEN_BRAKE, timestamp):
                continue

            self._set_cooldown(tid, AnomalyType.SUDDEN_BRAKE, timestamp)

            out.append(AnomalyEvent(
                anomaly_type=AnomalyType.SUDDEN_BRAKE,
                severity=ANOMALY_SEVERITY_MAP[AnomalyType.SUDDEN_BRAKE],
                frame_id=frame_id,
                track_id=tid,
                involved_track_ids=[tid],
                class_name=info.class_name,
                speed_kmh=info.speed_kmh,
                duration_sec=0.0,
                world_x=info.world_x,
                world_y=info.world_y,
                bbox=info.bbox,
                proximity_m=None,
                decel_kmh=decel,
                message=(
                    f"Araç (track_id={tid}, {info.class_name}) ani yavaşlama: "
                    f"{max_speed_in_window:.0f} → {info.speed_kmh:.0f} km/h "
                    f"({decel:.0f} km/h düşüş, {cfg.sudden_brake_window_sec}s içinde)."
                ),
            ))

    def _detect_overspeed(
        self,
        frame_id: int,
        timestamp: float,
        out: list[AnomalyEvent],
    ):
        """Hız limiti aşımı: anlık hız eşiği aştığında tetikler."""
        cfg = self.cfg

        for tid, info in self._current_info.items():
            if not self._is_track_mature(tid, timestamp):
                continue

            if info.speed_kmh <= cfg.overspeed_threshold_kmh:
                continue

            if self._is_on_cooldown(tid, AnomalyType.OVERSPEED, timestamp):
                continue

            self._set_cooldown(tid, AnomalyType.OVERSPEED, timestamp)

            over_by = info.speed_kmh - cfg.overspeed_threshold_kmh

            out.append(AnomalyEvent(
                anomaly_type=AnomalyType.OVERSPEED,
                severity=ANOMALY_SEVERITY_MAP[AnomalyType.OVERSPEED],
                frame_id=frame_id,
                track_id=tid,
                involved_track_ids=[tid],
                class_name=info.class_name,
                speed_kmh=info.speed_kmh,
                duration_sec=0.0,
                world_x=info.world_x,
                world_y=info.world_y,
                bbox=info.bbox,
                proximity_m=None,
                decel_kmh=None,
                message=(
                    f"Araç (track_id={tid}, {info.class_name}) hız limiti aşımı: "
                    f"{info.speed_kmh:.0f} km/h (limit: {cfg.overspeed_threshold_kmh:.0f}, "
                    f"+{over_by:.0f} km/h)."
                ),
            ))

    def _detect_underspeed(
        self,
        frame_id: int,
        timestamp: float,
        out: list[AnomalyEvent],
    ):
        """
        Asgari hız ihlali: belirli süre boyunca asgari hız altında seyreden araç.
        Duran araçları (STOPPED_VEHICLE) tekrar tetiklememek için
        stopped_speed_threshold üstünde ama underspeed_threshold altında olanları yakalar.
        """
        cfg = self.cfg

        for tid, history in self._history.items():
            if len(history) < 2:
                continue
            if not self._is_track_mature(tid, timestamp):
                continue

            info = self._current_info.get(tid)
            if info is None:
                continue

            # Duran araçlar STOPPED_VEHICLE kapsamında — burada tekrar tetikleme
            if info.speed_kmh <= cfg.stopped_speed_threshold_kmh:
                continue

            # Geçmişte kesintisiz olarak underspeed altında kaldığı başlangıcı bul
            under_start = None
            for sample in reversed(history):
                if cfg.stopped_speed_threshold_kmh < sample.speed_kmh <= cfg.underspeed_threshold_kmh:
                    under_start = sample.timestamp
                else:
                    break

            if under_start is None:
                continue

            duration = timestamp - under_start
            if duration < cfg.underspeed_duration_sec:
                continue

            if self._is_on_cooldown(tid, AnomalyType.UNDERSPEED, timestamp):
                continue

            self._set_cooldown(tid, AnomalyType.UNDERSPEED, timestamp)

            out.append(AnomalyEvent(
                anomaly_type=AnomalyType.UNDERSPEED,
                severity=ANOMALY_SEVERITY_MAP[AnomalyType.UNDERSPEED],
                frame_id=frame_id,
                track_id=tid,
                involved_track_ids=[tid],
                class_name=info.class_name,
                speed_kmh=info.speed_kmh,
                duration_sec=duration,
                world_x=info.world_x,
                world_y=info.world_y,
                bbox=info.bbox,
                proximity_m=None,
                decel_kmh=None,
                message=(
                    f"Araç (track_id={tid}, {info.class_name}) asgari hız ihlali: "
                    f"{info.speed_kmh:.0f} km/h, {duration:.1f} saniyedir "
                    f"limit altında ({cfg.underspeed_threshold_kmh:.0f} km/h)."
                ),
            ))

    # ──────────────────────────────────────────────────────────────────────
    # Direction learning
    # ──────────────────────────────────────────────────────────────────────

    def _direction_learning_update(self, tracks: list[TrackInfo]):
        """
        Öğrenme fazı: her track'in ilk ve güncel pozisyonundan hareket vektörü çıkar.
        Yeterli displacement'a ulaşan track'ler oy kullanır.
        Yeterli oy toplandığında çoğunluk yönü trafik akış yönü olarak belirlenir.
        """
        if self._direction_learned:
            return

        cfg = self.cfg
        min_disp = cfg.direction_learning_min_displacement_m

        for t in tracks:
            tid = t.track_id

            # Çok yavaş araçlar güvenilir yön vermez
            if t.speed_kmh < cfg.stopped_speed_threshold_kmh * 2:
                continue

            # İlk pozisyon kaydı
            if tid not in self._direction_first_pos:
                self._direction_first_pos[tid] = (t.world_x, t.world_y)
                continue

            # Zaten oy vermiş
            if tid in self._direction_voted_tids:
                continue

            # Displacement hesapla
            fx, fy = self._direction_first_pos[tid]
            dx = t.world_x - fx
            dy = t.world_y - fy
            disp = math.hypot(dx, dy)

            if disp < min_disp:
                continue

            # Birim vektör olarak oy kaydet
            dir_x = dx / disp
            dir_y = dy / disp
            self._direction_votes[tid] = (dir_x, dir_y)
            self._direction_voted_tids.add(tid)

            print(
                f"[anomaly] Direction vote #{len(self._direction_votes)}/"
                f"{cfg.direction_learning_vehicle_count}: "
                f"track_id={tid} yön=({dir_x:.2f}, {dir_y:.2f}) "
                f"displacement={disp:.1f}m"
            )

            # Yeterli oy toplandı mı?
            if len(self._direction_votes) >= cfg.direction_learning_vehicle_count:
                self._finalize_direction_learning()
                return

    def _finalize_direction_learning(self):
        """Toplanan oylardan çoğunluk yönünü hesapla ve öğrenmeyi tamamla."""
        votes = list(self._direction_votes.values())

        if not votes:
            self._apply_fallback_direction()
            return

        # Tüm oy vektörlerini topla (ortalama yön)
        sum_x = sum(v[0] for v in votes)
        sum_y = sum(v[1] for v in votes)
        mag = math.hypot(sum_x, sum_y)

        if mag < 1e-9:
            # Oylar birbirini nötralize etti — çift yönlü yol?
            # Fallback kullan
            print("[anomaly] Yön oyları nötralize oldu — fallback yön kullanılıyor")
            self._apply_fallback_direction()
            return

        learned_x = sum_x / mag
        learned_y = sum_y / mag

        self._learned_dir = (learned_x, learned_y)
        self._direction_learned = True

        # Yön açısını insanlar için okunur hale getir
        angle_deg = math.degrees(math.atan2(learned_y, learned_x))

        # Oy dağılımını hesapla: kaç tanesi öğrenilen yöne uyumlu?
        threshold_cos = math.cos(math.radians(90))  # ±90° tolerans
        aligned = sum(
            1 for vx, vy in votes
            if (vx * learned_x + vy * learned_y) > threshold_cos
        )

        print(
            f"[anomaly] ═══ Trafik yönü öğrenildi ═══\n"
            f"  Yön vektörü: ({learned_x:.3f}, {learned_y:.3f})\n"
            f"  Açı: {angle_deg:.1f}°\n"
            f"  Uyumlu oy: {aligned}/{len(votes)}\n"
            f"  Wrong-way tespiti artık aktif."
        )

        # Öğrenme fazı state'ini temizle (bellek tasarrufu)
        self._direction_votes.clear()
        self._direction_voted_tids.clear()
        self._direction_first_pos.clear()

    def _apply_fallback_direction(self):
        """Fallback yönü uygula ve öğrenmeyi tamamlanmış say."""
        fx = self.cfg.fallback_direction_x
        fy = self.cfg.fallback_direction_y
        mag = math.hypot(fx, fy)
        if mag < 1e-9:
            self._learned_dir = (0.0, 1.0)
        else:
            self._learned_dir = (fx / mag, fy / mag)
        self._direction_learned = True

    @property
    def direction_learned(self) -> bool:
        """Trafik yönü öğrenildi mi?"""
        return self._direction_learned

    @property
    def learned_direction(self) -> tuple[float, float] | None:
        """Öğrenilen trafik yön vektörü (birim vektör). None ise henüz öğrenilmedi."""
        return self._learned_dir

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _is_track_mature(self, tid: int, timestamp: float) -> bool:
        """Track yeterince yaşlı mı (false positive engelleme)."""
        first = self._first_seen.get(tid)
        if first is None:
            return False
        return (timestamp - first) >= self.cfg.min_track_age_sec

    def _find_continuous_stop_start(
        self,
        history: deque[TrackSample],
        threshold_kmh: float,
    ) -> float | None:
        """
        Geçmişteki en eski kesintisiz durma anını bul.
        Sondan başa doğru tarar; threshold'u aşan ilk sample'da durur.
        """
        stop_start = None
        for sample in reversed(history):
            if sample.speed_kmh <= threshold_kmh:
                stop_start = sample.timestamp
            else:
                break
        return stop_start

    def _find_continuous_wrong_way_start(
        self,
        history: deque[TrackSample],
    ) -> float | None:
        """
        Geçmişteki kesintisiz ters yön başlangıcını bul.
        Ardışık sample çiftlerinden hareket vektörü hesaplar.
        Öğrenilen trafik yönünü referans alır.
        """
        if self._learned_dir is None:
            return None

        threshold_cos = math.cos(
            math.radians(self.cfg.wrong_way_angle_threshold_deg)
        )
        ax, ay = self._learned_dir
        wrong_start = None

        samples = list(history)
        for i in range(len(samples) - 1, 0, -1):
            s1 = samples[i - 1]
            s0 = samples[i]

            dx = s0.world_x - s1.world_x
            dy = s0.world_y - s1.world_y
            mag = math.hypot(dx, dy)

            if mag < 0.01:  # çok küçük hareket — yön hesaplanamaz
                continue

            # Hareket yönü ile izin verilen yön arasındaki cos(açı)
            cos_angle = (dx * ax + dy * ay) / mag

            if cos_angle <= threshold_cos:
                # Ters yönde
                wrong_start = s1.timestamp
            else:
                break

        return wrong_start

    def _prune_stale_tracks(self, timestamp: float, active_tids: set[int]):
        """Uzun süredir güncellenmeyen track'leri temizle."""
        stale = []
        for tid, history in self._history.items():
            if tid in active_tids:
                continue
            if not history:
                stale.append(tid)
                continue
            if (timestamp - history[-1].timestamp) > self.cfg.track_timeout_sec:
                stale.append(tid)

        for tid in stale:
            del self._history[tid]
            self._first_seen.pop(tid, None)

    def _is_on_cooldown(self, key, anomaly_type: AnomalyType, timestamp: float) -> bool:
        cd_key = (key, anomaly_type)
        last = self._cooldowns.get(cd_key)
        if last is None:
            return False
        return (timestamp - last) < self.cfg.cooldown_sec

    def _set_cooldown(self, key, anomaly_type: AnomalyType, timestamp: float):
        self._cooldowns[(key, anomaly_type)] = timestamp

    def _prune_cooldowns(self, timestamp: float):
        """Süresi dolmuş cooldown'ları temizle."""
        expired = [
            k for k, ts in self._cooldowns.items()
            if (timestamp - ts) > self.cfg.cooldown_sec * 2
        ]
        for k in expired:
            del self._cooldowns[k]

    @staticmethod
    def _point_in_polygon(
        px: float, py: float, polygon: list[tuple[float, float]]
    ) -> bool:
        """
        Ray casting algoritması — point-in-polygon test.
        cv2 bağımlılığı olmadan çalışır (Jetson'da import overhead'i azaltır).
        """
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > py) != (yj > py)) and (
                px < (xj - xi) * (py - yi) / (yj - yi) + xi
            ):
                inside = not inside
            j = i
        return inside
