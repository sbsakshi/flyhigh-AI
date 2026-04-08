"""
drone.py — Drone state machine and movement simulation.

Simulates a 2D drone flying between waypoints with battery drain,
discrete 1-second time steps, and status tracking.
"""

import math
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [DRONE] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DroneStatus(str, Enum):
    """Possible operational states of the drone."""
    FLYING      = "flying"
    HOVERING    = "hovering"
    REPLANNING  = "replanning"
    ABORTED     = "aborted"
    COMPLETE    = "complete"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Position:
    """2D position in meters."""
    x: float
    y: float

    def distance_to(self, other: "Position") -> float:
        """Return Euclidean distance in meters to another position."""
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def __repr__(self) -> str:
        return f"({self.x:.1f}, {self.y:.1f})"


@dataclass
class DroneStateSnapshot:
    """
    Immutable snapshot of drone state at a given simulation tick.
    Passed to the AI replanner and feasibility checker.
    """
    timestamp: str
    tick: int
    position: Position
    battery: float          # 0.0 – 100.0 %
    speed: float            # m/s
    status: DroneStatus
    current_waypoint_index: int
    total_distance_flown: float   # metres
    battery_consumed: float       # % since mission start


# ---------------------------------------------------------------------------
# Drone
# ---------------------------------------------------------------------------

class Drone:
    """
    2D drone simulation with discrete 1-second time steps.

    Battery drain formula:
        drain_per_step (%) = (distance_this_step / speed) * BATTERY_DRAIN_RATE
    where BATTERY_DRAIN_RATE = 0.8 % per second of flight.

    The drone moves in straight lines between waypoints.
    """

    BATTERY_DRAIN_RATE: float = 0.8   # % per second of flight
    CRITICAL_BATTERY:   float = 20.0  # % threshold
    TIME_STEP:          int   = 1     # seconds per simulation tick

    def __init__(
        self,
        start_position: Position,
        battery: float = 100.0,
        speed: float = 5.0,            # m/s
    ) -> None:
        """
        Initialise drone at start_position with full battery.

        Args:
            start_position: Initial (x, y) coordinates in metres.
            battery:        Starting battery percentage (0–100).
            speed:          Cruise speed in m/s.
        """
        self.position:               Position    = start_position
        self.battery:                float       = battery
        self.speed:                  float       = speed
        self.status:                 DroneStatus = DroneStatus.HOVERING
        self.current_waypoint_index: int         = 0
        self.tick:                   int         = 0
        self.total_distance_flown:   float       = 0.0
        self.battery_consumed:       float       = 0.0

        # Path history for visualiser: list of Position objects
        self.path_history: list[Position] = [Position(start_position.x, start_position.y)]

        logger.info(
            "Drone initialised at %s | battery=%.1f%% | speed=%.1f m/s",
            self.position, self.battery, self.speed,
        )

    # ------------------------------------------------------------------
    # Core movement
    # ------------------------------------------------------------------

    def move_to_next_waypoint(
        self,
        waypoints: list[Position],
    ) -> bool:
        """
        Advance drone one TIME_STEP (1 second) toward the next waypoint.

        Movement is along a straight line. If the drone reaches (or passes)
        the waypoint within this tick it snaps to the waypoint position and
        returns True (waypoint reached). Otherwise it moves speed * TIME_STEP
        metres along the heading and returns False.

        Args:
            waypoints: Ordered list of remaining Position targets.

        Returns:
            True if the current waypoint was reached this tick, else False.
        """
        if not waypoints:
            self._set_status(DroneStatus.HOVERING)
            logger.warning("move_to_next_waypoint called with empty waypoint list.")
            return False

        target = waypoints[0]
        dist_to_target = self.position.distance_to(target)

        # Max distance coverable this tick
        step_distance = self.speed * self.TIME_STEP

        self._set_status(DroneStatus.FLYING)

        if step_distance >= dist_to_target:
            # Reach (or exactly hit) the waypoint this tick
            actual_distance = dist_to_target
            self.position = Position(target.x, target.y)
            self.apply_battery_drain(actual_distance)
            self.path_history.append(Position(self.position.x, self.position.y))
            self.tick += 1
            logger.info(
                "Tick %03d | Reached waypoint at %s | battery=%.1f%%",
                self.tick, self.position, self.battery,
            )
            return True
        else:
            # Move step_distance towards target
            ratio = step_distance / dist_to_target
            new_x = self.position.x + ratio * (target.x - self.position.x)
            new_y = self.position.y + ratio * (target.y - self.position.y)
            self.position = Position(new_x, new_y)
            self.apply_battery_drain(step_distance)
            self.path_history.append(Position(self.position.x, self.position.y))
            self.tick += 1
            logger.debug(
                "Tick %03d | Flying to %s | now at %s | battery=%.1f%%",
                self.tick, target, self.position, self.battery,
            )
            return False

    # ------------------------------------------------------------------
    # Battery
    # ------------------------------------------------------------------

    def apply_battery_drain(self, distance_meters: float) -> float:
        """
        Drain battery based on distance flown.

        Formula:
            drain (%) = (distance_meters / speed) * BATTERY_DRAIN_RATE

        Args:
            distance_meters: Metres flown this tick.

        Returns:
            Battery percentage drained (positive float).
        """
        drain = (distance_meters / self.speed) * self.BATTERY_DRAIN_RATE
        self.battery = max(0.0, self.battery - drain)
        self.battery_consumed += drain
        self.total_distance_flown += distance_meters

        if self.is_battery_critical():
            logger.warning(
                "Tick %03d | CRITICAL BATTERY: %.1f%% remaining!",
                self.tick, self.battery,
            )

        return drain

    def is_battery_critical(self) -> bool:
        """Return True if battery is below the critical threshold (< 20%)."""
        return self.battery < self.CRITICAL_BATTERY

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def _set_status(self, new_status: DroneStatus) -> None:
        """Update drone status and log transitions."""
        if self.status != new_status:
            logger.info(
                "Tick %03d | Status: %s -> %s",
                self.tick, self.status.value, new_status.value,
            )
            self.status = new_status

    def set_replanning(self) -> None:
        """Freeze the drone in REPLANNING state (called by main loop)."""
        self._set_status(DroneStatus.REPLANNING)

    def set_complete(self) -> None:
        """Mark mission as complete."""
        self._set_status(DroneStatus.COMPLETE)

    def set_aborted(self) -> None:
        """Mark mission as aborted (e.g. RTB due to critical battery)."""
        self._set_status(DroneStatus.ABORTED)

    # ------------------------------------------------------------------
    # State snapshot
    # ------------------------------------------------------------------

    def get_state_snapshot(self) -> DroneStateSnapshot:
        """
        Return an immutable snapshot of the current drone state.

        Used by the feasibility checker and AI replanner.
        """
        return DroneStateSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            tick=self.tick,
            position=Position(self.position.x, self.position.y),
            battery=round(self.battery, 2),
            speed=self.speed,
            status=self.status,
            current_waypoint_index=self.current_waypoint_index,
            total_distance_flown=round(self.total_distance_flown, 2),
            battery_consumed=round(self.battery_consumed, 2),
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Drone(pos={self.position}, battery={self.battery:.1f}%, "
            f"status={self.status.value}, tick={self.tick})"
        )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("COMPONENT 1 — Drone Simulation Test")
    print("=" * 60)

    # Create drone at origin
    drone = Drone(start_position=Position(0.0, 0.0), battery=100.0, speed=5.0)
    print(f"\nInitial state: {drone}")

    # Define a simple triangle route
    waypoints = [
        Position(20.0, 0.0),   # 20 m east
        Position(20.0, 15.0),  # 15 m north
        Position(0.0,  0.0),   # back to base
    ]

    print(f"\nRoute: {waypoints}")
    print(f"Total straight-line distance: "
          f"{sum(waypoints[i].distance_to(waypoints[i+1]) for i in range(len(waypoints)-1)) + Position(0,0).distance_to(waypoints[0]):.1f} m")
    print()

    wp_index = 0
    max_ticks = 200

    for _ in range(max_ticks):
        if wp_index >= len(waypoints):
            drone.set_complete()
            break

        reached = drone.move_to_next_waypoint(waypoints[wp_index:])
        if reached:
            wp_index += 1
            drone.current_waypoint_index = wp_index

        if drone.is_battery_critical():
            print(f"\n!! Battery critical at tick {drone.tick}: {drone.battery:.1f}%")
            drone.set_aborted()
            break

    snapshot = drone.get_state_snapshot()
    print("\n--- Final Snapshot ---")
    print(f"  Tick:              {snapshot.tick}")
    print(f"  Position:          {snapshot.position}")
    print(f"  Battery:           {snapshot.battery}%")
    print(f"  Status:            {snapshot.status.value}")
    print(f"  Distance flown:    {snapshot.total_distance_flown} m")
    print(f"  Battery consumed:  {snapshot.battery_consumed}%")
    print(f"  Path points:       {len(drone.path_history)}")
    print()

    # Quick drain test
    print("--- Battery drain sanity check ---")
    d2 = Drone(Position(0, 0), battery=100.0, speed=5.0)
    # 25 m at 5 m/s = 5 seconds flight → 5 * 0.8 = 4.0% drain
    drained = d2.apply_battery_drain(25.0)
    print(f"  25 m at 5 m/s -> drain={drained:.2f}%  (expected 4.00%)")
    print(f"  Battery after:  {d2.battery:.2f}%  (expected 96.00%)")
    assert abs(drained - 4.0) < 1e-6, "Drain formula incorrect!"
    assert abs(d2.battery - 96.0) < 1e-6, "Battery after drain incorrect!"
    print("  PASS")

    print()
    print("  is_battery_critical() at 96%:", d2.is_battery_critical(), "(expected False)")
    d2.battery = 19.9
    print("  is_battery_critical() at 19.9%:", d2.is_battery_critical(), "(expected True)")
