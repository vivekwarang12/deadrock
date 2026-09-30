"""
types.py  –  SIH Edge Engine Data Types
=========================================
Standardized, device-agnostic data structures for inertial and GNSS telemetry.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Tuple


class AxisConvention(str, Enum):
    """IMU mounting / body frame axis convention."""
    ENU = "East-North-Up"               # X=East, Y=North, Z=Up
    NED = "North-East-Down"             # X=North, Y=East, Z=Down
    PHONE_PORTRAIT = "Phone-Portrait"   # X=Right, Y=Up, Z=Out of screen
    VEHICLE_CAN = "Vehicle-Standard"    # X=Forward, Y=Left, Z=Up


@dataclass
class IMUSample:
    """
    Standardized IMU reading from any physical sensor (smartphone, MEMS, FOG).
    timestamp_sec: Monotonic or POSIX epoch timestamp in seconds.
    accel_ms2: (ax, ay, az) in m/s^2.
    gyro_rads: (gx, gy, gz) in rad/s.
    mag_ut: Optional (mx, my, mz) in microTesla.
    """
    timestamp_sec: float
    accel_ms2: Tuple[float, float, float]
    gyro_rads: Tuple[float, float, float]
    mag_ut: Optional[Tuple[float, float, float]] = None


@dataclass
class GNSSSample:
    """
    GNSS position fix from receiver / Android Location API.
    """
    timestamp_sec: float
    latitude_deg: float
    longitude_deg: float
    altitude_m: Optional[float] = None
    speed_ms: Optional[float] = None
    bearing_deg: Optional[float] = None
    accuracy_m: float = 3.0
    satellites: Optional[int] = None
    hdop: Optional[float] = None


@dataclass
class NavigationState:
    """
    Real-time navigation state output from SIHEdgeEngine.
    """
    timestamp_sec: float
    latitude_deg: float
    longitude_deg: float
    speed_ms: float
    heading_deg: float
    mode: str                         # GNSS-AIDED-INS | DEGRADED-GNSS | PURE-INERTIAL-DR
    drift_percent: float              # Estimated drift % of distance
    confidence_score: float           # 0.0 to 1.0 confidence estimate
    map_matched_lat: Optional[float] = None
    map_matched_lon: Optional[float] = None
    matched_road_name: Optional[str] = None
