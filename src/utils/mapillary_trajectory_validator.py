"""
Mapillary Trajectory and Sequence Validator for Geo-RAG.

Detects invalid Mapillary sequences caused by stationary cameras, multipath GPS
drift, impossible speed teleportations, and chaotic random-walk jitter.
Maintains a local SQLite cache so sequence and photo lookups are never repeated.
"""

import json
import logging
import math
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests

from src.utils.credentials import get_mapillary_token
from src.utils.mapillary_trajectory_sql import (
    INIT_SCHEMA_STATEMENTS,
    INSERT_PHOTO_SEQUENCES_QUERY,
    SELECT_PHOTO_SEQUENCES_TEMPLATE,
    SELECT_SEQUENCE_BY_ID_QUERY,
    UPSERT_SEQUENCE_VERDICT_QUERY,
)

logger = logging.getLogger("mapillary_trajectory_validator")


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Computes great-circle distance between two (lat, lon) coordinates in meters."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


@dataclass
class TrajectoryMetrics:
    frame_count: int
    total_distance_m: float
    net_displacement_m: float
    tortuosity: float
    max_speed_kmh: float
    p95_speed_kmh: float
    bounding_radius_m: float
    duration_seconds: float


@dataclass
class ValidationResult:
    sequence_id: str
    is_valid: bool
    reason: str
    metrics: Optional[TrajectoryMetrics] = None
    geojson_track: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


class MapillaryTrajectoryValidator:
    """
    Validates Mapillary sequence tracks against realistic motion kinematics.
    Caches verdicts and photo-to-sequence mappings in SQLite.
    """

    DEFAULT_CACHE_PATH = "data/cache/mapillary_sequences.db"

    def __init__(
        self,
        token: Optional[str] = None,
        db_path: str = DEFAULT_CACHE_PATH,
        timeout: int = 15,
        max_speed_threshold_kmh: float = 160.0,
        max_tortuosity_threshold: float = 20.0,
        min_stationary_frames: int = 15,
        max_stationary_radius_m: float = 35.0,
    ):
        self.token = token or get_mapillary_token() or ""
        self.db_path = db_path
        self.timeout = timeout
        self.max_speed_threshold_kmh = max_speed_threshold_kmh
        self.max_tortuosity_threshold = max_tortuosity_threshold
        self.min_stationary_frames = min_stationary_frames
        self.max_stationary_radius_m = max_stationary_radius_m

        self._mem_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:")

        self._init_db()

    @contextmanager
    def _get_db(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding the active SQLite connection."""
        conn = (
            self._mem_conn
            if self._mem_conn is not None
            else sqlite3.connect(self.db_path)
        )
        try:
            with conn:
                yield conn
        finally:
            if self._mem_conn is None:
                conn.close()

    def _init_db(self) -> None:
        """Initializes the SQLite cache schema."""
        if self.db_path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)

        with self._get_db() as conn:
            cursor = conn.cursor()
            for stmt in INIT_SCHEMA_STATEMENTS:
                cursor.execute(stmt)

    # --------------------------------------------------------------------------
    # 1. Photo ID to Sequence ID Resolution
    # --------------------------------------------------------------------------

    def resolve_photo_sequences(
        self, photo_ids: List[str], batch_size: int = 100
    ) -> Dict[str, str]:
        """
        Resolves photo IDs to sequence IDs.
        Uses SQLite cache first, then batch queries Mapillary Graph API v4 for missing IDs.
        """
        if not photo_ids:
            return {}

        clean_ids = [str(p).strip().removesuffix(".0") for p in photo_ids if p]
        unique_ids = list(dict.fromkeys(clean_ids))
        seq_map: Dict[str, str] = {}
        missing_ids: List[str] = []

        # 1. Check local cache
        with self._get_db() as conn:
            cursor = conn.cursor()
            # Batch select in chunks of 500 to avoid SQLite variable limits
            for i in range(0, len(unique_ids), 500):
                chunk = unique_ids[i : i + 500]
                placeholders = ",".join("?" for _ in chunk)
                cursor.execute(
                    SELECT_PHOTO_SEQUENCES_TEMPLATE.format(placeholders=placeholders),
                    chunk,
                )
                for pid, sid in cursor.fetchall():
                    seq_map[str(pid)] = str(sid)

        for pid in unique_ids:
            if pid not in seq_map:
                missing_ids.append(pid)

        if not missing_ids:
            return {pid: seq_map[pid] for pid in clean_ids if pid in seq_map}

        # 2. Batch query API for missing IDs
        if not self.token:
            logger.warning(
                "No Mapillary token configured. Cannot resolve missing sequences."
            )
            return seq_map

        headers = {"Authorization": f"OAuth {self.token}"}
        new_mappings: List[Tuple[str, str]] = []

        for i in range(0, len(missing_ids), batch_size):
            chunk = missing_ids[i : i + batch_size]
            url = (
                f"https://graph.mapillary.com/?ids={','.join(chunk)}&fields=id,sequence"
            )
            for attempt in range(3):
                try:
                    r = requests.get(url, headers=headers, timeout=self.timeout)
                    if r.status_code == 200:
                        data = r.json()
                        for pid, info in data.items():
                            if isinstance(info, dict) and "sequence" in info:
                                sid = str(info["sequence"])
                                seq_map[str(pid)] = sid
                                new_mappings.append((str(pid), sid))
                        break
                    elif r.status_code == 429:
                        time.sleep(1.5 * (attempt + 1))
                    else:
                        break
                except Exception as e:
                    if attempt == 2:
                        logger.warning("Failed to fetch photo sequences batch: %s", e)
                    time.sleep(1.0)

        # 3. Cache new mappings in SQLite
        if new_mappings:
            with self._get_db() as conn:
                cursor = conn.cursor()
                cursor.executemany(
                    INSERT_PHOTO_SEQUENCES_QUERY,
                    new_mappings,
                )
                conn.commit()

        return {pid: seq_map[pid] for pid in clean_ids if pid in seq_map}

    # --------------------------------------------------------------------------
    # 2. Sequence Trajectory Retrieval
    # --------------------------------------------------------------------------

    def fetch_sequence_trajectory(
        self, sequence_id: str, max_points: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Fetches the chronological points of a Mapillary sequence track.
        Returns a sorted list of dicts:
        [{'photo_id': ..., 'captured_at': ms, 'lat': ..., 'lon': ..., 'compass_angle': ...}]
        """
        if not self.token or not sequence_id:
            return []

        headers = {"Authorization": f"OAuth {self.token}"}

        # Step A: Get all image IDs in sequence
        image_ids: List[str] = []
        url = f"https://graph.mapillary.com/image_ids?sequence_id={sequence_id}&limit=2000"
        while url and len(image_ids) < max_points:
            success = False
            for attempt in range(3):
                try:
                    r = requests.get(url, headers=headers, timeout=self.timeout)
                    if r.status_code == 200:
                        data = r.json()
                        for item in data.get("data", []):
                            if "id" in item:
                                image_ids.append(
                                    str(item["id"]).strip().removesuffix(".0")
                                )
                        url = data.get("paging", {}).get("next")
                        success = True
                        break
                    elif r.status_code == 429:
                        time.sleep(1.5 * (attempt + 1))
                    else:
                        break
                except Exception:
                    time.sleep(1.0)
            if not success:
                break

        if not image_ids:
            return []

        # If sequence is exceptionally long, subsample evenly to max_points
        if len(image_ids) > max_points:
            step = len(image_ids) / float(max_points)
            sampled_ids = [image_ids[int(i * step)] for i in range(max_points)]
            if image_ids[-1] not in sampled_ids:
                sampled_ids[-1] = image_ids[-1]
            image_ids = sampled_ids

        # Step B: Batch fetch metadata for image IDs
        points: List[Dict[str, Any]] = []
        batch_size = 100
        for i in range(0, len(image_ids), batch_size):
            chunk = image_ids[i : i + batch_size]
            url = f"https://graph.mapillary.com/?ids={','.join(chunk)}&fields=id,geometry,computed_geometry,captured_at,compass_angle"
            for attempt in range(3):
                try:
                    r = requests.get(url, headers=headers, timeout=self.timeout)
                    if r.status_code == 200:
                        data = r.json()
                        for pid, info in data.items():
                            if isinstance(info, dict) and "geometry" in info:
                                coords = info["geometry"].get("coordinates", [])
                                if len(coords) == 2:
                                    lon, lat = coords
                                    ts = info.get("captured_at") or 0
                                    angle = info.get("compass_angle")
                                    points.append(
                                        {
                                            "photo_id": str(pid),
                                            "captured_at": ts,
                                            "lat": float(lat),
                                            "lon": float(lon),
                                            "compass_angle": angle,
                                        }
                                    )
                        break
                    elif r.status_code == 429:
                        time.sleep(1.5 * (attempt + 1))
                    else:
                        break
                except Exception:
                    time.sleep(1.0)

        # Sort points chronologically
        points.sort(key=lambda p: p["captured_at"])
        return points

    # --------------------------------------------------------------------------
    # 3. Kinematic & Geometric Trajectory Evaluation
    # --------------------------------------------------------------------------

    def compute_trajectory_metrics(
        self, points: List[Dict[str, Any]]
    ) -> Optional[TrajectoryMetrics]:
        """Computes physical and kinematic metrics across a sorted list of trajectory points."""
        n = len(points)
        if n < 2:
            return None

        # 1. Total distance and step speeds
        total_dist = 0.0
        speeds_kmh: List[float] = []

        for i in range(n - 1):
            p1 = points[i]
            p2 = points[i + 1]
            dist_step = haversine_distance(p1["lat"], p1["lon"], p2["lat"], p2["lon"])
            total_dist += dist_step

            dt_ms = p2["captured_at"] - p1["captured_at"]
            if (
                dt_ms >= 500
            ):  # At least 0.5s to avoid sub-second timestamp quantization noise
                dt_sec = dt_ms / 1000.0
                v_kmh = (dist_step / dt_sec) * 3.6
                speeds_kmh.append(v_kmh)

        # 2. Net displacement
        first_pt = points[0]
        last_pt = points[-1]
        net_disp = haversine_distance(
            first_pt["lat"], first_pt["lon"], last_pt["lat"], last_pt["lon"]
        )

        # 3. Tortuosity ratio
        tortuosity = total_dist / max(net_disp, 1.0)

        # 4. Speeds
        max_speed = max(speeds_kmh) if speeds_kmh else 0.0
        if speeds_kmh:
            sorted_speeds = sorted(speeds_kmh)
            p95_idx = int(0.95 * len(sorted_speeds))
            p95_speed = sorted_speeds[min(p95_idx, len(sorted_speeds) - 1)]
        else:
            p95_speed = 0.0

        # 5. Bounding radius from centroid
        avg_lat = sum(p["lat"] for p in points) / float(n)
        avg_lon = sum(p["lon"] for p in points) / float(n)
        bounding_radius = max(
            haversine_distance(avg_lat, avg_lon, p["lat"], p["lon"]) for p in points
        )

        # 6. Duration
        duration_sec = max(
            0.0, (last_pt["captured_at"] - first_pt["captured_at"]) / 1000.0
        )

        return TrajectoryMetrics(
            frame_count=n,
            total_distance_m=round(total_dist, 2),
            net_displacement_m=round(net_disp, 2),
            tortuosity=round(tortuosity, 2),
            max_speed_kmh=round(max_speed, 1),
            p95_speed_kmh=round(p95_speed, 1),
            bounding_radius_m=round(bounding_radius, 2),
            duration_seconds=round(duration_sec, 1),
        )

    def evaluate_trajectory_metrics(
        self, metrics: TrajectoryMetrics
    ) -> Tuple[bool, str]:
        """
        Evaluates computed trajectory metrics against kinematic rules.
        Returns (is_valid: bool, reason: str).
        """
        # Rule 1: Stationary camera GPS drift / multipath jitter
        # Many frames, small bounding radius, but high cumulative path length and tortuosity
        if (
            metrics.frame_count >= self.min_stationary_frames
            and metrics.bounding_radius_m <= self.max_stationary_radius_m
            and metrics.total_distance_m >= 100.0
            and metrics.tortuosity >= 15.0
        ):
            return (
                False,
                f"Stationary camera GPS jitter: {metrics.frame_count} frames trapped in {metrics.bounding_radius_m:.1f}m radius with {metrics.total_distance_m:.1f}m jitter (Tortuosity: {metrics.tortuosity:.1f})",
            )

        # Rule 2: Random-walk ping-pong tortuosity
        if (
            metrics.total_distance_m >= 250.0
            and metrics.net_displacement_m <= 25.0
            and metrics.tortuosity >= self.max_tortuosity_threshold
        ):
            return (
                False,
                f"Severe random-walk tortuosity: {metrics.total_distance_m:.1f}m traveled with only {metrics.net_displacement_m:.1f}m net progression (Tortuosity: {metrics.tortuosity:.1f})",
            )

        # Rule 3: Unrealistic speed teleportation jumps
        if metrics.max_speed_kmh >= self.max_speed_threshold_kmh:
            return (
                False,
                f"Impossible speed teleportation: peak instantaneous speed {metrics.max_speed_kmh:.1f} km/h exceeds threshold {self.max_speed_threshold_kmh:.1f} km/h",
            )

        return (True, "Valid moving trajectory")

    # --------------------------------------------------------------------------
    # 4. Sequence Validation with SQLite Cache
    # --------------------------------------------------------------------------

    def validate_sequence(
        self,
        sequence_id: str,
        trajectory_points: Optional[List[Dict[str, Any]]] = None,
        force_refresh: bool = False,
    ) -> ValidationResult:
        """
        Validates a sequence ID.
        Checks SQLite cache first. If not cached, fetches trajectory, calculates metrics,
        evaluates kinematics, and caches the result.
        """
        if not sequence_id:
            return ValidationResult(
                sequence_id="", is_valid=False, reason="Empty sequence ID"
            )

        # 1. Check SQLite cache
        if not force_refresh:
            with self._get_db() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    SELECT_SEQUENCE_BY_ID_QUERY,
                    (sequence_id,),
                )
                row = cursor.fetchone()
                if row:
                    (
                        is_valid,
                        reason,
                        frames,
                        tot_dist,
                        net_disp,
                        tort,
                        max_spd,
                        p95_spd,
                        b_rad,
                        dur,
                        geojson_str,
                    ) = row
                    metrics = TrajectoryMetrics(
                        frame_count=frames,
                        total_distance_m=tot_dist,
                        net_displacement_m=net_disp,
                        tortuosity=tort,
                        max_speed_kmh=max_spd,
                        p95_speed_kmh=p95_spd,
                        bounding_radius_m=b_rad,
                        duration_seconds=dur,
                    )
                    track_geojson = json.loads(geojson_str) if geojson_str else None
                    return ValidationResult(
                        sequence_id=sequence_id,
                        is_valid=bool(is_valid),
                        reason=reason,
                        metrics=metrics,
                        geojson_track=track_geojson,
                    )

        # 2. Fetch trajectory points if not provided
        points = trajectory_points
        if points is None:
            points = self.fetch_sequence_trajectory(sequence_id)

        # If too few points exist to disprove validity, consider it valid
        if not points or len(points) < 2:
            verdict = ValidationResult(
                sequence_id=sequence_id,
                is_valid=True,
                reason="Single frame or trajectory unavailable",
                metrics=None,
                geojson_track=None,
            )
            self._cache_verdict(verdict)
            return verdict

        # 3. Compute metrics & evaluate
        metrics = self.compute_trajectory_metrics(points)
        if not metrics:
            verdict = ValidationResult(
                sequence_id=sequence_id,
                is_valid=True,
                reason="Insufficient coordinate data",
            )
            self._cache_verdict(verdict)
            return verdict

        is_valid, reason = self.evaluate_trajectory_metrics(metrics)

        # 4. Build GeoJSON LineString track
        coordinates = [[p["lon"], p["lat"]] for p in points]
        geojson_track = {
            "type": "Feature",
            "properties": {
                "sequence_id": sequence_id,
                "is_valid": is_valid,
                "reason": reason,
                **asdict(metrics),
            },
            "geometry": {"type": "LineString", "coordinates": coordinates},
        }

        verdict = ValidationResult(
            sequence_id=sequence_id,
            is_valid=is_valid,
            reason=reason,
            metrics=metrics,
            geojson_track=geojson_track,
        )

        # 5. Save to cache
        self._cache_verdict(verdict)
        return verdict

    def _cache_verdict(self, verdict: ValidationResult) -> None:
        """Stores a validation result in the SQLite sequences table."""
        m = verdict.metrics
        frames = m.frame_count if m else 0
        tot_dist = m.total_distance_m if m else 0.0
        net_disp = m.net_displacement_m if m else 0.0
        tort = m.tortuosity if m else 0.0
        max_spd = m.max_speed_kmh if m else 0.0
        p95_spd = m.p95_speed_kmh if m else 0.0
        b_rad = m.bounding_radius_m if m else 0.0
        dur = m.duration_seconds if m else 0.0
        geojson_str = json.dumps(verdict.geojson_track) if verdict.geojson_track else ""

        with self._get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                UPSERT_SEQUENCE_VERDICT_QUERY,
                (
                    verdict.sequence_id,
                    1 if verdict.is_valid else 0,
                    verdict.reason,
                    frames,
                    tot_dist,
                    net_disp,
                    tort,
                    max_spd,
                    p95_spd,
                    b_rad,
                    dur,
                    geojson_str,
                    time.time(),
                ),
            )
            conn.commit()

    def validate_photo_id(self, photo_id: str) -> ValidationResult:
        """Convenience method: resolves photo ID to sequence ID and validates the sequence."""
        seq_map = self.resolve_photo_sequences([photo_id])
        seq_id = seq_map.get(str(photo_id).strip().removesuffix(".0"))
        if not seq_id:
            return ValidationResult(
                sequence_id="",
                is_valid=True,
                reason="Could not resolve sequence ID for photo",
            )
        return self.validate_sequence(seq_id)
