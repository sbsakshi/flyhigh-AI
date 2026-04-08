"""
anomaly.py -- Anomaly injection system for mid-flight disruptions.

Three anomaly types:
  BATTERY_DROP    -- sudden battery loss (25-40%)
  NO_FLY_ZONE     -- circular restricted area that blocks waypoints inside it
  WAYPOINT_FAILURE -- a specific waypoint becomes inaccessible

After every injection, the feasibility checker is re-run and the result
is returned so the main loop knows whether to trigger replanning.
"""

import os
import sys
import math
import random
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from drone_replanner.sim.drone import Position, Drone
from drone_replanner.sim.mission import Mission, WaypointStatus
from drone_replanner.sim.feasibility import (
    FeasibilityResult,
    check_feasibility,
    SAFETY_MARGIN,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ANOMALY] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AnomalyType(str, Enum):
    """Supported anomaly categories."""
    BATTERY_DROP     = "BATTERY_DROP"
    NO_FLY_ZONE      = "NO_FLY_ZONE"
    WAYPOINT_FAILURE = "WAYPOINT_FAILURE"


class AnomalySeverity(str, Enum):
    """Severity of the anomaly event."""
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


# ---------------------------------------------------------------------------
# Anomaly dataclass
# ---------------------------------------------------------------------------

@dataclass
class Anomaly:
    """
    Record of a single injected anomaly event.

    Attributes:
        type:        Category of anomaly.
        severity:    LOW / MEDIUM / HIGH.
        timestamp:   ISO-8601 UTC string when the anomaly was injected.
        description: Human-readable explanation (also passed to LLM prompt).
        params:      Raw parameters used to create the anomaly (for replay).
    """
    type:        AnomalyType
    severity:    AnomalySeverity
    timestamp:   str
    description: str
    params:      dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"Anomaly({self.type.value}/{self.severity.value} "
            f"@ {self.timestamp[:19]}: {self.description})"
        )


# ---------------------------------------------------------------------------
# No-fly zone helper
# ---------------------------------------------------------------------------

@dataclass
class NoFlyZone:
    """
    Circular no-fly zone defined by a centre position and radius.

    Attributes:
        center: Centre of the restricted circle (metres).
        radius: Radius of the restricted circle (metres).
    """
    center: Position
    radius: float

    def contains(self, pos: Position) -> bool:
        """Return True if pos falls inside (or on the boundary of) this zone."""
        return pos.distance_to(self.center) <= self.radius

    def __repr__(self) -> str:
        return f"NFZ(center={self.center}, r={self.radius}m)"


# ---------------------------------------------------------------------------
# Injection result
# ---------------------------------------------------------------------------

@dataclass
class InjectionResult:
    """
    Outcome of a single anomaly injection.

    Attributes:
        anomaly:           The anomaly that was injected.
        feasibility:       Post-injection feasibility check result.
        replanning_needed: True if the mission is no longer feasible.
        affected_waypoints: IDs of waypoints newly BLOCKED by this anomaly.
        battery_before:    Battery % before the anomaly (BATTERY_DROP only).
        battery_after:     Battery % after the anomaly (BATTERY_DROP only).
    """
    anomaly:            Anomaly
    feasibility:        FeasibilityResult
    replanning_needed:  bool
    affected_waypoints: list[int]    = field(default_factory=list)
    battery_before:     Optional[float] = None
    battery_after:      Optional[float] = None


# ---------------------------------------------------------------------------
# Anomaly Injector
# ---------------------------------------------------------------------------

class AnomalyInjector:
    """
    Injects anomalies into a live drone mission and evaluates the impact.

    Maintains a stack of all active anomalies and no-fly zones so the
    visualiser and prompt builder can render them.

    Usage::

        injector = AnomalyInjector(drone, mission, base_position)
        result = injector.inject_random_anomaly()
        if result.replanning_needed:
            # trigger AI replanner
    """

    # Battery drop bounds
    BATTERY_DROP_MIN: float = 25.0
    BATTERY_DROP_MAX: float = 40.0

    def __init__(
        self,
        drone:         Drone,
        mission:       Mission,
        base_position: Position,
    ) -> None:
        """
        Initialise the injector.

        Args:
            drone:         Live Drone instance (battery will be mutated).
            mission:       Live Mission instance (waypoint statuses mutated).
            base_position: Home base used for feasibility checks.
        """
        self.drone:          Drone        = drone
        self.mission:        Mission      = mission
        self.base_position:  Position     = base_position
        self._anomalies:     list[Anomaly]    = []
        self._no_fly_zones:  list[NoFlyZone]  = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inject_random_anomaly(self) -> InjectionResult:
        """
        Inject a randomly chosen anomaly type with random parameters.

        BATTERY_DROP:     random drop between BATTERY_DROP_MIN and MAX.
        NO_FLY_ZONE:      random centre near the remaining waypoints,
                          radius between 8 and 20 m.
        WAYPOINT_FAILURE: random PENDING waypoint.

        Returns:
            InjectionResult with feasibility check and replanning flag.
        """
        anomaly_type = random.choice(list(AnomalyType))

        if anomaly_type == AnomalyType.BATTERY_DROP:
            drop = round(random.uniform(self.BATTERY_DROP_MIN, self.BATTERY_DROP_MAX), 1)
            return self.inject_specific_anomaly(
                AnomalyType.BATTERY_DROP,
                params={"drop_amount": drop},
            )

        elif anomaly_type == AnomalyType.NO_FLY_ZONE:
            # Place zone near a random remaining waypoint
            remaining = self.mission.get_remaining_waypoints()
            if not remaining:
                # Fallback: battery drop if nothing to block
                drop = round(random.uniform(self.BATTERY_DROP_MIN, self.BATTERY_DROP_MAX), 1)
                return self.inject_specific_anomaly(
                    AnomalyType.BATTERY_DROP,
                    params={"drop_amount": drop},
                )
            anchor = random.choice(remaining).position
            cx = round(anchor.x + random.uniform(-5, 5), 1)
            cy = round(anchor.y + random.uniform(-5, 5), 1)
            radius = round(random.uniform(8.0, 20.0), 1)
            return self.inject_specific_anomaly(
                AnomalyType.NO_FLY_ZONE,
                params={"center_x": cx, "center_y": cy, "radius": radius},
            )

        else:  # WAYPOINT_FAILURE
            remaining = self.mission.get_remaining_waypoints()
            if not remaining:
                drop = round(random.uniform(self.BATTERY_DROP_MIN, self.BATTERY_DROP_MAX), 1)
                return self.inject_specific_anomaly(
                    AnomalyType.BATTERY_DROP,
                    params={"drop_amount": drop},
                )
            target_wp = random.choice(remaining)
            return self.inject_specific_anomaly(
                AnomalyType.WAYPOINT_FAILURE,
                params={"waypoint_id": target_wp.id},
            )

    def inject_specific_anomaly(
        self,
        anomaly_type: AnomalyType,
        params:       dict[str, Any],
    ) -> InjectionResult:
        """
        Inject a specific anomaly with explicit parameters.

        Args:
            anomaly_type: One of AnomalyType enum values.
            params:       Dict of parameters for the anomaly type:
                          BATTERY_DROP     -> {"drop_amount": float}
                          NO_FLY_ZONE      -> {"center_x": float,
                                               "center_y": float,
                                               "radius": float}
                          WAYPOINT_FAILURE -> {"waypoint_id": int}

        Returns:
            InjectionResult with anomaly record, feasibility, and replanning flag.

        Raises:
            ValueError: On unknown anomaly type or missing params.
        """
        if anomaly_type == AnomalyType.BATTERY_DROP:
            return self._inject_battery_drop(params)
        elif anomaly_type == AnomalyType.NO_FLY_ZONE:
            return self._inject_no_fly_zone(params)
        elif anomaly_type == AnomalyType.WAYPOINT_FAILURE:
            return self._inject_waypoint_failure(params)
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    def get_active_anomalies(self) -> list[Anomaly]:
        """Return all anomalies injected so far (immutable copy)."""
        return list(self._anomalies)

    def get_no_fly_zones(self) -> list[NoFlyZone]:
        """Return all active no-fly zones (immutable copy)."""
        return list(self._no_fly_zones)

    # ------------------------------------------------------------------
    # Private injection handlers
    # ------------------------------------------------------------------

    def _inject_battery_drop(self, params: dict[str, Any]) -> InjectionResult:
        """Handle BATTERY_DROP anomaly."""
        drop_amount: float = float(params.get("drop_amount", 30.0))
        drop_amount = max(0.0, min(drop_amount, self.drone.battery))

        battery_before = round(self.drone.battery, 2)
        self.drone.battery = max(0.0, self.drone.battery - drop_amount)
        battery_after  = round(self.drone.battery, 2)

        severity = self._battery_drop_severity(drop_amount)
        description = (
            f"Sudden battery drop of {drop_amount:.1f}%: "
            f"{battery_before:.1f}% -> {battery_after:.1f}%"
        )

        anomaly = self._record_anomaly(
            AnomalyType.BATTERY_DROP, severity, description, params,
        )
        logger.warning(description)

        feasibility = self._run_feasibility()
        return InjectionResult(
            anomaly           = anomaly,
            feasibility       = feasibility,
            replanning_needed = not feasibility.is_feasible,
            battery_before    = battery_before,
            battery_after     = battery_after,
        )

    def _inject_no_fly_zone(self, params: dict[str, Any]) -> InjectionResult:
        """Handle NO_FLY_ZONE anomaly."""
        cx:     float = float(params["center_x"])
        cy:     float = float(params["center_y"])
        radius: float = float(params["radius"])

        nfz = NoFlyZone(center=Position(cx, cy), radius=radius)
        self._no_fly_zones.append(nfz)

        # Block all PENDING waypoints inside the zone
        affected: list[int] = []
        for wp in self.mission.get_remaining_waypoints():
            if nfz.contains(wp.position):
                self.mission.mark_blocked(
                    wp.id,
                    reason=f"Inside no-fly zone at ({cx},{cy}) r={radius}m",
                )
                affected.append(wp.id)

        severity    = self._nfz_severity(len(affected))
        description = (
            f"No-fly zone activated at ({cx},{cy}) radius={radius}m. "
            f"Blocked waypoints: {affected if affected else 'none'}"
        )

        anomaly = self._record_anomaly(
            AnomalyType.NO_FLY_ZONE, severity, description, params,
        )
        logger.warning(description)

        feasibility = self._run_feasibility()
        return InjectionResult(
            anomaly            = anomaly,
            feasibility        = feasibility,
            replanning_needed  = not feasibility.is_feasible or len(affected) > 0,
            affected_waypoints = affected,
        )

    def _inject_waypoint_failure(self, params: dict[str, Any]) -> InjectionResult:
        """Handle WAYPOINT_FAILURE anomaly."""
        wp_id: int = int(params["waypoint_id"])

        wp = self.mission.get_waypoint(wp_id)
        if wp.status == WaypointStatus.PENDING:
            self.mission.mark_blocked(wp_id, reason="Waypoint failure: site inaccessible")

        severity    = self._wp_failure_severity(wp.priority.value)
        description = (
            f"Waypoint {wp_id} ({wp.type.value}/{wp.priority.value}) "
            f"failed: site inaccessible"
        )

        anomaly = self._record_anomaly(
            AnomalyType.WAYPOINT_FAILURE, severity, description, params,
        )
        logger.warning(description)

        feasibility = self._run_feasibility()
        return InjectionResult(
            anomaly            = anomaly,
            feasibility        = feasibility,
            replanning_needed  = not feasibility.is_feasible or True,
            affected_waypoints = [wp_id],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_feasibility(self) -> FeasibilityResult:
        """Re-run feasibility check with current drone + mission state."""
        return check_feasibility(
            current_position    = self.drone.position,
            current_battery     = self.drone.battery,
            remaining_waypoints = self.mission.get_remaining_waypoints(),
            base_position       = self.base_position,
            speed               = self.drone.speed,
        )

    def _record_anomaly(
        self,
        anomaly_type: AnomalyType,
        severity:     AnomalySeverity,
        description:  str,
        params:       dict[str, Any],
    ) -> Anomaly:
        """Create, store, and return a new Anomaly record."""
        anomaly = Anomaly(
            type        = anomaly_type,
            severity    = severity,
            timestamp   = datetime.now(timezone.utc).isoformat(),
            description = description,
            params      = dict(params),
        )
        self._anomalies.append(anomaly)
        return anomaly

    @staticmethod
    def _battery_drop_severity(drop: float) -> AnomalySeverity:
        """Map drop amount to severity."""
        if drop >= 35:
            return AnomalySeverity.HIGH
        elif drop >= 28:
            return AnomalySeverity.MEDIUM
        return AnomalySeverity.LOW

    @staticmethod
    def _nfz_severity(blocked_count: int) -> AnomalySeverity:
        """Map number of blocked waypoints to severity."""
        if blocked_count >= 3:
            return AnomalySeverity.HIGH
        elif blocked_count >= 1:
            return AnomalySeverity.MEDIUM
        return AnomalySeverity.LOW

    @staticmethod
    def _wp_failure_severity(priority: str) -> AnomalySeverity:
        """Map waypoint priority to anomaly severity."""
        mapping = {
            "CRITICAL": AnomalySeverity.HIGH,
            "HIGH":     AnomalySeverity.MEDIUM,
            "LOW":      AnomalySeverity.LOW,
        }
        return mapping.get(priority, AnomalySeverity.MEDIUM)

    def __repr__(self) -> str:
        return (
            f"AnomalyInjector("
            f"anomalies={len(self._anomalies)}, "
            f"nfz={len(self._no_fly_zones)})"
        )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from drone_replanner.sim.mission import make_mission_medium, make_mission_hard

    print("=" * 60)
    print("COMPONENT 4 -- Anomaly Injector Test")
    print("=" * 60)

    BASE  = Position(0.0, 0.0)

    # ------------------------------------------------------------------
    # Test 1: BATTERY_DROP
    # ------------------------------------------------------------------
    print("\n[Test 1] BATTERY_DROP anomaly")
    from drone_replanner.sim.mission import make_mission_easy
    m1      = make_mission_easy()
    drone1  = Drone(Position(0, 0), battery=80.0, speed=5.0)
    inj1    = AnomalyInjector(drone1, m1, BASE)

    result1 = inj1.inject_specific_anomaly(
        AnomalyType.BATTERY_DROP, {"drop_amount": 30.0}
    )
    print(f"  Battery before: {result1.battery_before}%")
    print(f"  Battery after:  {result1.battery_after}%")
    print(f"  Replanning needed: {result1.replanning_needed}")
    print(f"  Feasibility: {result1.feasibility.summary_str()}")
    assert result1.battery_after == 50.0, f"Expected 50.0, got {result1.battery_after}"
    assert result1.anomaly.type == AnomalyType.BATTERY_DROP
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 2: NO_FLY_ZONE blocks waypoints
    # ------------------------------------------------------------------
    print("\n[Test 2] NO_FLY_ZONE anomaly")
    m2     = make_mission_medium()
    drone2 = Drone(Position(0, 0), battery=100.0, speed=5.0)
    inj2   = AnomalyInjector(drone2, m2, BASE)

    # WP3 is at (45,10), WP4 at (40,25) -- zone centred at (43,15) r=15 should hit both
    result2 = inj2.inject_specific_anomaly(
        AnomalyType.NO_FLY_ZONE,
        {"center_x": 43.0, "center_y": 15.0, "radius": 15.0},
    )
    print(f"  Affected waypoints: {result2.affected_waypoints}")
    print(f"  Blocked IDs in mission: {m2.get_blocked_ids()}")
    print(f"  Replanning needed: {result2.replanning_needed}")
    print(f"  NFZ recorded: {inj2.get_no_fly_zones()}")
    assert len(result2.affected_waypoints) >= 1, "Expected at least 1 WP blocked"
    assert result2.replanning_needed
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 3: WAYPOINT_FAILURE
    # ------------------------------------------------------------------
    print("\n[Test 3] WAYPOINT_FAILURE anomaly")
    m3     = make_mission_medium()
    drone3 = Drone(Position(0, 0), battery=100.0, speed=5.0)
    inj3   = AnomalyInjector(drone3, m3, BASE)

    result3 = inj3.inject_specific_anomaly(
        AnomalyType.WAYPOINT_FAILURE, {"waypoint_id": 3}
    )
    print(f"  Affected WP: {result3.affected_waypoints}")
    print(f"  Severity: {result3.anomaly.severity.value}")
    print(f"  Replanning needed: {result3.replanning_needed}")
    assert 3 in m3.get_blocked_ids()
    assert result3.anomaly.severity == AnomalySeverity.HIGH  # WP3 is CRITICAL
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 4: Stacked anomalies
    # ------------------------------------------------------------------
    print("\n[Test 4] Stacked anomalies (battery drop + no-fly zone)")
    m4     = make_mission_hard()
    drone4 = Drone(Position(0, 0), battery=70.0, speed=5.0)
    inj4   = AnomalyInjector(drone4, m4, BASE)

    inj4.inject_specific_anomaly(AnomalyType.BATTERY_DROP, {"drop_amount": 35.0})
    inj4.inject_specific_anomaly(
        AnomalyType.NO_FLY_ZONE,
        {"center_x": 35.0, "center_y": 10.0, "radius": 10.0},
    )
    active = inj4.get_active_anomalies()
    print(f"  Active anomalies: {len(active)}")
    for a in active:
        print(f"    {a}")
    assert len(active) == 2
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 5: inject_random_anomaly
    # ------------------------------------------------------------------
    print("\n[Test 5] inject_random_anomaly x5")
    random.seed(42)
    m5     = make_mission_hard()
    drone5 = Drone(Position(0, 0), battery=100.0, speed=5.0)
    inj5   = AnomalyInjector(drone5, m5, BASE)

    for i in range(5):
        r = inj5.inject_random_anomaly()
        print(f"  [{i+1}] {r.anomaly.type.value} | replan={r.replanning_needed}")
    assert len(inj5.get_active_anomalies()) == 5
    print("  PASS")

    print("\nAll tests passed.")
