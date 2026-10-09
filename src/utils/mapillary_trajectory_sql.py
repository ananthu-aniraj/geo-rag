"""
SQL Queries and Schema Definitions for Mapillary Trajectory & Sequence Caching.

Decouples raw SQLite DDL and DML queries from MapillaryTrajectoryValidator business logic.
"""

# Schema DDL statements
CREATE_PHOTO_SEQUENCES_TABLE = """
CREATE TABLE IF NOT EXISTS photo_sequences (
    photo_id TEXT PRIMARY KEY,
    sequence_id TEXT NOT NULL
);
"""

CREATE_SEQUENCES_TABLE = """
CREATE TABLE IF NOT EXISTS sequences (
    sequence_id TEXT PRIMARY KEY,
    is_valid INTEGER NOT NULL,
    reason TEXT NOT NULL,
    frame_count INTEGER NOT NULL,
    total_distance_m REAL NOT NULL,
    net_displacement_m REAL NOT NULL,
    tortuosity REAL NOT NULL,
    max_speed_kmh REAL NOT NULL,
    p95_speed_kmh REAL NOT NULL,
    bounding_radius_m REAL NOT NULL,
    duration_seconds REAL NOT NULL,
    track_geojson TEXT,
    checked_at REAL NOT NULL
);
"""

CREATE_PHOTO_SEQ_INDEX = """
CREATE INDEX IF NOT EXISTS idx_photo_seq ON photo_sequences(sequence_id);
"""

INIT_SCHEMA_STATEMENTS = [
    CREATE_PHOTO_SEQUENCES_TABLE,
    CREATE_SEQUENCES_TABLE,
    CREATE_PHOTO_SEQ_INDEX,
]

# Photo sequence queries
SELECT_PHOTO_SEQUENCES_TEMPLATE = "SELECT photo_id, sequence_id FROM photo_sequences WHERE photo_id IN ({placeholders})"

INSERT_PHOTO_SEQUENCES_QUERY = (
    "INSERT OR REPLACE INTO photo_sequences (photo_id, sequence_id) VALUES (?, ?)"
)


def get_photo_sequences_query(num_ids: int) -> str:
    """Returns the parameterized SQL query for selecting photo sequence IDs."""
    placeholders = ",".join("?" for _ in range(num_ids))
    return SELECT_PHOTO_SEQUENCES_TEMPLATE.format(placeholders=placeholders)


# Sequence verdict queries
SELECT_SEQUENCE_BY_ID_QUERY = """
SELECT is_valid, reason, frame_count, total_distance_m, net_displacement_m,
       tortuosity, max_speed_kmh, p95_speed_kmh, bounding_radius_m,
       duration_seconds, track_geojson
FROM sequences
WHERE sequence_id = ?
"""

UPSERT_SEQUENCE_VERDICT_QUERY = """
INSERT OR REPLACE INTO sequences (
    sequence_id, is_valid, reason, frame_count, total_distance_m,
    net_displacement_m, tortuosity, max_speed_kmh, p95_speed_kmh,
    bounding_radius_m, duration_seconds, track_geojson, checked_at
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""
