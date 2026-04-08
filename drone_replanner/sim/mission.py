"""
mission.py — Waypoint definitions and mission management engine.

Manages an ordered list of waypoints, tracks completion/skip/block state,
and provides three hardcoded test missions of increasing complexity.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

# Allow both `python drone_replanner/sim/mission.py` and package imports.
if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from drone_replanner.sim.drone import Position

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MISSION] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class WaypointPriority(str, Enum):
    """Mission criticality of a waypoint."""
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    LOW      = "LOW"


class WaypointType(str, Enum):
    """Operational type of a waypoint."""
    DELIVER = "DELIVER"
    SURVEY  = "SURVEY"
    INSPECT = "INSPECT"
    RTB     = "RTB"       # Return to Base


class WaypointStatus(str, Enum):
    """Execution state of a waypoint."""
    PENDING   = "PENDING"
    COMPLETED = "COMPLETED"
    SKIPPED   = "SKIPPED"
    BLOCKED   = "BLOCKED"


# ---------------------------------------------------------------------------
# Waypoint
# ---------------------------------------------------------------------------

@dataclass
class Waypoint:
    """
    A single mission waypoint with position, priority, type, and status.

    Attributes:
        id:       Unique integer identifier.
        position: 2D position in metres.
        priority: CRITICAL / HIGH / LOW.
        type:     DELIVER / SURVEY / INSPECT / RTB.
        status:   PENDING / COMPLETED / SKIPPED / BLOCKED.
        skip_reason:  Populated when status is SKIPPED.
        block_reason: Populated when status is BLOCKED.
    """
    id:           int
    position:     Position
    priority:     WaypointPriority
    type:         WaypointType
    status:       WaypointStatus       = field(default=WaypointStatus.PENDING)
    skip_reason:  Optional[str]        = field(default=None)
    block_reason: Optional[str]        = field(default=None)

    def is_actionable(self) -> bool:
        """Return True if the waypoint can still be visited (status PENDING)."""
        return self.status == WaypointStatus.PENDING

    def __repr__(self) -> str:
        return (
            f"WP(id={self.id}, pos={self.position}, "
            f"pri={self.priority.value}, type={self.type.value}, "
            f"status={self.status.value})"
        )


# ---------------------------------------------------------------------------
# Mission
# ---------------------------------------------------------------------------

@dataclass
class MissionSummary:
    """
    Snapshot of mission progress returned by get_mission_summary().

    Attributes:
        total_waypoints:     Total number of waypoints in the mission.
        completed:           IDs of completed waypoints.
        skipped:             IDs of skipped waypoints.
        blocked:             IDs of blocked waypoints.
        pending:             IDs of still-pending waypoints.
        completion_rate:     Fraction of non-RTB waypoints completed.
        critical_completed:  Count of CRITICAL waypoints completed.
        critical_skipped:    Count of CRITICAL waypoints skipped.
        replanning_count:    Number of times the plan was updated.
    """
    total_waypoints:    int
    completed:          list[int]
    skipped:            list[int]
    blocked:            list[int]
    pending:            list[int]
    completion_rate:    float
    critical_completed: int
    critical_skipped:   int
    replanning_count:   int


class Mission:
    """
    Manages an ordered list of waypoints for a drone mission.

    Responsibilities:
    - Track current execution order (mutable by replanner).
    - Record completed, skipped, and blocked waypoints.
    - Provide remaining-waypoint views to the feasibility checker and AI.
    """

    def __init__(self, waypoints: list[Waypoint], name: str = "Mission") -> None:
        """
        Initialise the mission with an ordered list of waypoints.

        Args:
            waypoints: Ordered list of Waypoint objects.
            name:      Human-readable mission name for logging.
        """
        if not waypoints:
            raise ValueError("Mission must have at least one waypoint.")

        self.name:             str           = name
        self._waypoints:       dict[int, Waypoint] = {wp.id: wp for wp in waypoints}
        self._order:           list[int]     = [wp.id for wp in waypoints]
        self._completed:       list[int]     = []
        self._skipped:         list[int]     = []
        self._replanning_count: int          = 0

        logger.info(
            "Mission '%s' loaded with %d waypoints: %s",
            self.name, len(waypoints), self._order,
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_waypoint(self, wp_id: int) -> Waypoint:
        """Return the Waypoint object for the given ID.

        Raises:
            KeyError: If wp_id is not in this mission.
        """
        if wp_id not in self._waypoints:
            raise KeyError(f"Waypoint ID {wp_id} not found in mission '{self.name}'.")
        return self._waypoints[wp_id]

    def get_all_waypoints(self) -> list[Waypoint]:
        """Return all waypoints in current execution order."""
        return [self._waypoints[wid] for wid in self._order]

    def get_remaining_waypoints(self) -> list[Waypoint]:
        """
        Return waypoints that are still PENDING, in current execution order.

        These are the candidates passed to the feasibility checker.
        """
        return [
            self._waypoints[wid]
            for wid in self._order
            if self._waypoints[wid].status == WaypointStatus.PENDING
        ]

    def get_blocked_ids(self) -> list[int]:
        """Return IDs of all BLOCKED waypoints."""
        return [
            wid for wid, wp in self._waypoints.items()
            if wp.status == WaypointStatus.BLOCKED
        ]

    # ------------------------------------------------------------------
    # State mutators
    # ------------------------------------------------------------------

    def mark_completed(self, wp_id: int) -> None:
        """
        Mark a waypoint as COMPLETED.

        Args:
            wp_id: ID of the waypoint to mark.

        Raises:
            KeyError: If wp_id not found.
            ValueError: If waypoint is not in PENDING state.
        """
        wp = self.get_waypoint(wp_id)
        if wp.status != WaypointStatus.PENDING:
            raise ValueError(
                f"Cannot complete WP {wp_id}: status is {wp.status.value}, not PENDING."
            )
        wp.status = WaypointStatus.COMPLETED
        self._completed.append(wp_id)
        logger.info("WP %d (%s/%s) marked COMPLETED.", wp_id, wp.type.value, wp.priority.value)

    def mark_skipped(self, wp_id: int, reason: str) -> None:
        """
        Mark a waypoint as SKIPPED with a reason string.

        Args:
            wp_id:  ID of the waypoint to skip.
            reason: Plain-English explanation for the skip.
        """
        wp = self.get_waypoint(wp_id)
        wp.status = WaypointStatus.SKIPPED
        wp.skip_reason = reason
        if wp_id not in self._skipped:
            self._skipped.append(wp_id)
        logger.info(
            "WP %d (%s/%s) marked SKIPPED. Reason: %s",
            wp_id, wp.type.value, wp.priority.value, reason,
        )

    def mark_blocked(self, wp_id: int, reason: str) -> None:
        """
        Mark a waypoint as BLOCKED with a reason string.

        Blocked waypoints are removed from active planning but kept in
        the waypoint registry so the AI replanner can inspect them.

        Args:
            wp_id:  ID of the waypoint to block.
            reason: Plain-English explanation (e.g. 'inside no-fly zone').
        """
        wp = self.get_waypoint(wp_id)
        wp.status = WaypointStatus.BLOCKED
        wp.block_reason = reason
        logger.warning(
            "WP %d (%s/%s) marked BLOCKED. Reason: %s",
            wp_id, wp.type.value, wp.priority.value, reason,
        )

    def update_order(self, new_order: list[int]) -> None:
        """
        Replace the current execution order with a new list from the replanner.

        Only PENDING waypoint IDs are accepted in new_order; completed,
        skipped, and blocked IDs are silently filtered out if present.

        Args:
            new_order: Ordered list of waypoint IDs.

        Raises:
            ValueError: If new_order contains an unknown waypoint ID.
        """
        for wid in new_order:
            if wid not in self._waypoints:
                raise ValueError(f"update_order: unknown waypoint ID {wid}.")

        # Filter to only actionable waypoints
        filtered = [
            wid for wid in new_order
            if self._waypoints[wid].status == WaypointStatus.PENDING
        ]

        old_order = [
            wid for wid in self._order
            if self._waypoints[wid].status == WaypointStatus.PENDING
        ]

        self._order = filtered
        self._replanning_count += 1
        logger.info(
            "Replanning #%d | Order updated: %s -> %s",
            self._replanning_count, old_order, filtered,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_mission_summary(self) -> MissionSummary:
        """
        Return a MissionSummary snapshot of current mission progress.
        """
        all_ids  = list(self._waypoints.keys())
        blocked  = self.get_blocked_ids()
        pending  = [wp.id for wp in self.get_remaining_waypoints()]

        # Exclude RTB from completion-rate calculation
        non_rtb = [
            wid for wid, wp in self._waypoints.items()
            if wp.type != WaypointType.RTB
        ]
        completed_non_rtb = [wid for wid in self._completed if wid in non_rtb]
        completion_rate = (
            len(completed_non_rtb) / len(non_rtb) if non_rtb else 0.0
        )

        critical_completed = sum(
            1 for wid in self._completed
            if self._waypoints[wid].priority == WaypointPriority.CRITICAL
        )
        critical_skipped = sum(
            1 for wid in self._skipped
            if self._waypoints[wid].priority == WaypointPriority.CRITICAL
        )

        return MissionSummary(
            total_waypoints=len(all_ids),
            completed=list(self._completed),
            skipped=list(self._skipped),
            blocked=blocked,
            pending=pending,
            completion_rate=round(completion_rate, 3),
            critical_completed=critical_completed,
            critical_skipped=critical_skipped,
            replanning_count=self._replanning_count,
        )

    def __repr__(self) -> str:
        remaining = len(self.get_remaining_waypoints())
        return (
            f"Mission('{self.name}', total={len(self._waypoints)}, "
            f"remaining={remaining}, replannings={self._replanning_count})"
        )


# ---------------------------------------------------------------------------
# Hardcoded test missions
# ---------------------------------------------------------------------------

def make_mission_easy() -> Mission:
    """
    Mission A — 5 waypoints, simple linear route.
    Suitable for baseline testing with no anomalies.
    """
    waypoints = [
        Waypoint(id=1, position=Position(10, 0),  priority=WaypointPriority.HIGH,     type=WaypointType.DELIVER),
        Waypoint(id=2, position=Position(20, 10), priority=WaypointPriority.LOW,      type=WaypointType.SURVEY),
        Waypoint(id=3, position=Position(30, 5),  priority=WaypointPriority.CRITICAL, type=WaypointType.INSPECT),
        Waypoint(id=4, position=Position(25, 20), priority=WaypointPriority.LOW,      type=WaypointType.SURVEY),
        Waypoint(id=5, position=Position(0, 0),   priority=WaypointPriority.HIGH,     type=WaypointType.RTB),
    ]
    return Mission(waypoints, name="Mission-A-Easy")


def make_mission_medium() -> Mission:
    """
    Mission B — 8 waypoints, mixed priorities, spread across a wider area.
    Tests replanning when a mid-mission anomaly blocks 1-2 waypoints.
    """
    waypoints = [
        Waypoint(id=1,  position=Position(15, 5),   priority=WaypointPriority.HIGH,     type=WaypointType.DELIVER),
        Waypoint(id=2,  position=Position(30, 0),   priority=WaypointPriority.LOW,      type=WaypointType.SURVEY),
        Waypoint(id=3,  position=Position(45, 10),  priority=WaypointPriority.CRITICAL, type=WaypointType.INSPECT),
        Waypoint(id=4,  position=Position(40, 25),  priority=WaypointPriority.HIGH,     type=WaypointType.DELIVER),
        Waypoint(id=5,  position=Position(50, 40),  priority=WaypointPriority.LOW,      type=WaypointType.SURVEY),
        Waypoint(id=6,  position=Position(25, 35),  priority=WaypointPriority.CRITICAL, type=WaypointType.INSPECT),
        Waypoint(id=7,  position=Position(10, 20),  priority=WaypointPriority.LOW,      type=WaypointType.SURVEY),
        Waypoint(id=8,  position=Position(0, 0),    priority=WaypointPriority.HIGH,     type=WaypointType.RTB),
    ]
    return Mission(waypoints, name="Mission-B-Medium")


def make_mission_hard() -> Mission:
    """
    Mission C — 12 waypoints, dense grid, multiple CRITICAL tasks.
    Tests replanner under battery-drop + no-fly-zone stacked anomalies.
    """
    waypoints = [
        Waypoint(id=1,  position=Position(10, 10),  priority=WaypointPriority.HIGH,     type=WaypointType.DELIVER),
        Waypoint(id=2,  position=Position(20, 5),   priority=WaypointPriority.LOW,      type=WaypointType.SURVEY),
        Waypoint(id=3,  position=Position(35, 10),  priority=WaypointPriority.CRITICAL, type=WaypointType.INSPECT),
        Waypoint(id=4,  position=Position(50, 5),   priority=WaypointPriority.HIGH,     type=WaypointType.DELIVER),
        Waypoint(id=5,  position=Position(60, 20),  priority=WaypointPriority.CRITICAL, type=WaypointType.INSPECT),
        Waypoint(id=6,  position=Position(55, 35),  priority=WaypointPriority.LOW,      type=WaypointType.SURVEY),
        Waypoint(id=7,  position=Position(40, 40),  priority=WaypointPriority.HIGH,     type=WaypointType.DELIVER),
        Waypoint(id=8,  position=Position(25, 45),  priority=WaypointPriority.CRITICAL, type=WaypointType.INSPECT),
        Waypoint(id=9,  position=Position(10, 40),  priority=WaypointPriority.LOW,      type=WaypointType.SURVEY),
        Waypoint(id=10, position=Position(5, 25),   priority=WaypointPriority.HIGH,     type=WaypointType.DELIVER),
        Waypoint(id=11, position=Position(15, 30),  priority=WaypointPriority.LOW,      type=WaypointType.SURVEY),
        Waypoint(id=12, position=Position(0, 0),    priority=WaypointPriority.HIGH,     type=WaypointType.RTB),
    ]
    return Mission(waypoints, name="Mission-C-Hard")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("COMPONENT 2 -- Mission Engine Test")
    print("=" * 60)

    # --- Test 1: Easy mission basic operations ---
    print("\n[Test 1] Easy mission (5 waypoints)")
    m = make_mission_easy()
    print(f"  {m}")

    remaining = m.get_remaining_waypoints()
    print(f"  Remaining: {[wp.id for wp in remaining]}")
    assert len(remaining) == 5

    m.mark_completed(1)
    m.mark_skipped(2, reason="Low priority, battery conserved")
    m.mark_blocked(4, reason="Inside no-fly zone")

    remaining = m.get_remaining_waypoints()
    print(f"  Remaining after events: {[wp.id for wp in remaining]}")
    assert [wp.id for wp in remaining] == [3, 5], f"Got {[wp.id for wp in remaining]}"

    summary = m.get_mission_summary()
    print(f"  Completed:        {summary.completed}")
    print(f"  Skipped:          {summary.skipped}")
    print(f"  Blocked:          {summary.blocked}")
    print(f"  Pending:          {summary.pending}")
    print(f"  Completion rate:  {summary.completion_rate}")
    assert summary.completed == [1]
    assert summary.skipped   == [2]
    assert summary.blocked   == [4]
    assert summary.pending   == [3, 5]
    print("  PASS")

    # --- Test 2: update_order (replanner reorders) ---
    print("\n[Test 2] Replanner reorders remaining waypoints")
    m2 = make_mission_medium()
    m2.mark_completed(1)
    m2.mark_completed(2)
    # Replanner wants: 6, 3, 4, 5, 7, 8
    m2.update_order([6, 3, 4, 5, 7, 8])
    remaining2 = m2.get_remaining_waypoints()
    assert [wp.id for wp in remaining2] == [6, 3, 4, 5, 7, 8], \
        f"Got {[wp.id for wp in remaining2]}"
    print(f"  New order: {[wp.id for wp in remaining2]}")
    print(f"  Replanning count: {m2._replanning_count}")
    assert m2._replanning_count == 1
    print("  PASS")

    # --- Test 3: Hard mission summary ---
    print("\n[Test 3] Hard mission (12 waypoints) summary")
    m3 = make_mission_hard()
    for wp_id in [1, 3, 5, 7]:
        m3.mark_completed(wp_id)
    m3.mark_skipped(2, "Low priority")
    m3.mark_skipped(6, "Low priority")
    m3.mark_blocked(9, "No-fly zone")
    s = m3.get_mission_summary()
    print(f"  Total:              {s.total_waypoints}")
    print(f"  Completed:          {s.completed}")
    print(f"  Skipped:            {s.skipped}")
    print(f"  Blocked:            {s.blocked}")
    print(f"  Pending:            {s.pending}")
    print(f"  Completion rate:    {s.completion_rate}")
    print(f"  Critical completed: {s.critical_completed}")
    print(f"  Critical skipped:   {s.critical_skipped}")
    assert s.critical_completed == 2   # WP 3 and 5 are CRITICAL
    assert s.critical_skipped   == 0
    print("  PASS")

    # --- Test 4: All three factory functions load without error ---
    print("\n[Test 4] All mission factories load cleanly")
    for factory in [make_mission_easy, make_mission_medium, make_mission_hard]:
        mx = factory()
        assert len(mx.get_remaining_waypoints()) > 0
        print(f"  {mx}")
    print("  PASS")

    print("\nAll tests passed.")
