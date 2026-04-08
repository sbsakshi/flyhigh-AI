"""
feasibility.py -- Battery feasibility checker for drone mission planning.

Given the drone's current position, battery, speed, and an ordered list of
remaining waypoints, computes whether the mission can be completed and
returns a detailed breakdown used by both the main loop and the AI replanner.

Called BEFORE every LLM replanning call (to provide context) and AFTER
(to validate the LLM's proposed plan before it is applied).
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from drone_replanner.sim.drone import Position
from drone_replanner.sim.mission import Waypoint

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [FEASIBILITY] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATTERY_DRAIN_RATE: float = 0.8   # % per second of flight
DEFAULT_SPEED:      float = 5.0   # m/s
SAFETY_MARGIN:      float = 5.0   # % battery buffer kept in reserve


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeasibilityResult:
    """
    Full output of a feasibility check.

    Attributes:
        is_feasible:               True if the drone can complete all remaining
                                   waypoints AND return to base with >= safety margin.
        battery_needed:            Total % battery required (waypoints + RTB).
        battery_available:         Current battery minus the safety margin reserve.
        battery_remaining_after:   Estimated battery left after full mission.
        first_infeasible_waypoint: ID of the first waypoint that would make the
                                   mission infeasible, or None if fully feasible.
        max_reachable_waypoints:   Ordered list of waypoint IDs reachable before
                                   the drone must turn back to base.
        waypoint_costs:            Per-waypoint battery cost breakdown (id -> %).
        rtb_cost:                  Battery cost of the final return-to-base leg.
        total_distance:            Total route distance in metres.
        safety_margin:             The % buffer held in reserve (default 5%).
    """
    is_feasible:               bool
    battery_needed:            float
    battery_available:         float
    battery_remaining_after:   float
    first_infeasible_waypoint: Optional[int]
    max_reachable_waypoints:   list[int]
    waypoint_costs:            dict[int, float]
    rtb_cost:                  float
    total_distance:            float
    safety_margin:             float = SAFETY_MARGIN

    def summary_str(self) -> str:
        """Return a compact one-line summary for prompt injection."""
        feasible_tag = "FEASIBLE" if self.is_feasible else "INFEASIBLE"
        return (
            f"[{feasible_tag}] need={self.battery_needed:.1f}% "
            f"available={self.battery_available:.1f}% "
            f"after={self.battery_remaining_after:.1f}% "
            f"rtb_cost={self.rtb_cost:.1f}% "
            f"reachable={self.max_reachable_waypoints}"
        )


# ---------------------------------------------------------------------------
# Core checker
# ---------------------------------------------------------------------------

def check_feasibility(
    current_position: Position,
    current_battery:  float,
    remaining_waypoints: list[Waypoint],
    base_position:    Position,
    speed:            float = DEFAULT_SPEED,
    safety_margin:    float = SAFETY_MARGIN,
) -> FeasibilityResult:
    """
    Compute whether the drone can complete the remaining mission and RTB.

    Algorithm:
        1. Walk the ordered waypoint list, accumulating battery cost leg by leg.
        2. After each waypoint, compute the hypothetical RTB cost from that point.
        3. The waypoint is reachable if (cumulative_cost + rtb_from_here) fits
           within available battery.
        4. Stop accumulating at the first waypoint that pushes total cost over
           available battery — that waypoint is first_infeasible_waypoint.

    Battery cost formula (per leg):
        cost (%) = (distance_m / speed) * BATTERY_DRAIN_RATE

    Args:
        current_position:    Drone's current (x, y) in metres.
        current_battery:     Drone's current battery percentage.
        remaining_waypoints: Ordered list of PENDING Waypoint objects.
        base_position:       Home/base (x, y) the drone must return to.
        speed:               Drone cruise speed in m/s.
        safety_margin:       Battery % to hold in reserve (not consumable).

    Returns:
        FeasibilityResult with full breakdown.
    """
    drain_per_meter: float = BATTERY_DRAIN_RATE / speed
    available:       float = current_battery - safety_margin  # usable battery

    waypoint_costs:          dict[int, float] = {}
    max_reachable:           list[int]        = []
    first_infeasible:        Optional[int]    = None

    cumulative_cost: float = 0.0
    total_distance:  float = 0.0
    prev_pos:        Position = current_position

    for wp in remaining_waypoints:
        # Cost to fly from previous position to this waypoint
        leg_dist = prev_pos.distance_to(wp.position)
        leg_cost = leg_dist * drain_per_meter

        # Hypothetical RTB cost from THIS waypoint
        rtb_dist = wp.position.distance_to(base_position)
        rtb_cost = rtb_dist * drain_per_meter

        projected_total = cumulative_cost + leg_cost + rtb_cost

        waypoint_costs[wp.id] = round(leg_cost, 3)

        if projected_total <= available:
            # Still reachable
            cumulative_cost += leg_cost
            total_distance  += leg_dist
            max_reachable.append(wp.id)
            prev_pos = wp.position
        else:
            # First waypoint that breaks feasibility
            if first_infeasible is None:
                first_infeasible = wp.id
            logger.debug(
                "WP %d infeasible: need %.1f%% but only %.1f%% available",
                wp.id, projected_total, available,
            )

    # Final RTB leg cost from last reachable waypoint (or current pos if none)
    final_rtb_dist = prev_pos.distance_to(base_position)
    final_rtb_cost = final_rtb_dist * drain_per_meter
    total_distance += final_rtb_dist

    battery_needed        = round(cumulative_cost + final_rtb_cost, 3)
    battery_remaining     = round(current_battery - battery_needed, 3)
    is_feasible           = (first_infeasible is None) and (battery_needed <= available)

    result = FeasibilityResult(
        is_feasible               = is_feasible,
        battery_needed            = battery_needed,
        battery_available         = round(available, 3),
        battery_remaining_after   = battery_remaining,
        first_infeasible_waypoint = first_infeasible,
        max_reachable_waypoints   = max_reachable,
        waypoint_costs            = waypoint_costs,
        rtb_cost                  = round(final_rtb_cost, 3),
        total_distance            = round(total_distance, 3),
        safety_margin             = safety_margin,
    )

    logger.info(
        "Feasibility check: %s | need=%.1f%% avail=%.1f%% reachable=%s",
        "OK" if is_feasible else "FAIL",
        battery_needed, available, max_reachable,
    )

    return result


# ---------------------------------------------------------------------------
# Convenience wrapper: validate an LLM-proposed plan
# ---------------------------------------------------------------------------

def validate_replan(
    proposed_order:   list[int],
    all_waypoints:    dict[int, Waypoint],
    current_position: Position,
    current_battery:  float,
    base_position:    Position,
    speed:            float = DEFAULT_SPEED,
) -> FeasibilityResult:
    """
    Validate a replanner-proposed waypoint order against the feasibility checker.

    Builds an ordered Waypoint list from the proposed IDs and runs the full
    feasibility check. Used after every LLM response to catch invalid plans.

    Args:
        proposed_order:   Ordered list of waypoint IDs from the LLM.
        all_waypoints:    Dict mapping id -> Waypoint for the full mission.
        current_position: Drone's current position.
        current_battery:  Drone's current battery %.
        base_position:    Home base position.
        speed:            Drone cruise speed in m/s.

    Returns:
        FeasibilityResult for the proposed plan.

    Raises:
        KeyError: If a proposed ID is not in all_waypoints.
    """
    ordered_wps = [all_waypoints[wid] for wid in proposed_order]
    return check_feasibility(
        current_position    = current_position,
        current_battery     = current_battery,
        remaining_waypoints = ordered_wps,
        base_position       = base_position,
        speed               = speed,
    )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from drone_replanner.sim.mission import (
        Waypoint, WaypointPriority, WaypointType, WaypointStatus,
        make_mission_easy, make_mission_medium,
    )

    print("=" * 60)
    print("COMPONENT 3 -- Feasibility Checker Test")
    print("=" * 60)

    BASE = Position(0.0, 0.0)
    SPEED = 5.0

    # ------------------------------------------------------------------
    # Test 1: Fully feasible mission (plenty of battery)
    # ------------------------------------------------------------------
    print("\n[Test 1] Fully feasible mission")
    mission = make_mission_easy()
    remaining = mission.get_remaining_waypoints()
    result = check_feasibility(
        current_position    = Position(0.0, 0.0),
        current_battery     = 100.0,
        remaining_waypoints = remaining,
        base_position       = BASE,
        speed               = SPEED,
    )
    print(f"  {result.summary_str()}")
    print(f"  Waypoint costs: {result.waypoint_costs}")
    print(f"  RTB cost:       {result.rtb_cost}%")
    print(f"  Total distance: {result.total_distance} m")
    assert result.is_feasible, "Expected feasible with 100% battery"
    assert result.first_infeasible_waypoint is None
    assert len(result.max_reachable_waypoints) == 5
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 2: Low battery — some waypoints unreachable
    # ------------------------------------------------------------------
    print("\n[Test 2] Low battery -- partial route only")
    mission2 = make_mission_medium()
    remaining2 = mission2.get_remaining_waypoints()
    result2 = check_feasibility(
        current_position    = Position(0.0, 0.0),
        current_battery     = 25.0,
        remaining_waypoints = remaining2,
        base_position       = BASE,
        speed               = SPEED,
    )
    print(f"  {result2.summary_str()}")
    print(f"  Reachable:              {result2.max_reachable_waypoints}")
    print(f"  First infeasible WP:    {result2.first_infeasible_waypoint}")
    assert not result2.is_feasible, "Expected infeasible at 25% battery for 8-WP mission"
    assert result2.first_infeasible_waypoint is not None
    assert len(result2.max_reachable_waypoints) < 8
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 3: Critical battery (< 20%) — RTB only
    # ------------------------------------------------------------------
    print("\n[Test 3] Critical battery (15%) -- near-RTB only")
    result3 = check_feasibility(
        current_position    = Position(10.0, 10.0),
        current_battery     = 15.0,
        remaining_waypoints = mission2.get_remaining_waypoints(),
        base_position       = BASE,
        speed               = SPEED,
    )
    print(f"  {result3.summary_str()}")
    print(f"  Reachable: {result3.max_reachable_waypoints}")
    assert not result3.is_feasible
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 4: Battery cost sanity check
    # ------------------------------------------------------------------
    print("\n[Test 4] Battery cost sanity check")
    # Single waypoint 25 m away, base is at origin, drone at origin
    # Leg cost:  25 m / 5 m/s * 0.8 = 4.0%
    # RTB cost:  0 m (waypoint IS the base effectively — but let's use
    #            a waypoint at (25, 0) and base at (0, 0))
    # RTB from (25,0) to (0,0) = 25 m -> 4.0%
    # Total needed = 4.0 + 4.0 = 8.0%
    single_wp = [
        Waypoint(
            id=99, position=Position(25.0, 0.0),
            priority=WaypointPriority.HIGH,
            type=WaypointType.DELIVER,
            status=WaypointStatus.PENDING,
        )
    ]
    result4 = check_feasibility(
        current_position    = Position(0.0, 0.0),
        current_battery     = 100.0,
        remaining_waypoints = single_wp,
        base_position       = BASE,
        speed               = SPEED,
    )
    print(f"  battery_needed = {result4.battery_needed}%  (expected 8.0%)")
    print(f"  rtb_cost       = {result4.rtb_cost}%        (expected 4.0%)")
    assert abs(result4.battery_needed - 8.0) < 1e-4, f"Got {result4.battery_needed}"
    assert abs(result4.rtb_cost - 4.0) < 1e-4,       f"Got {result4.rtb_cost}"
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 5: validate_replan — proposed order feasibility
    # ------------------------------------------------------------------
    print("\n[Test 5] validate_replan with proposed LLM order")
    mission5 = make_mission_medium()
    all_wps = {wp.id: wp for wp in mission5.get_all_waypoints()}
    # Propose skipping WP 5 and 7 (LOW priority) to save battery
    proposed = [1, 2, 3, 4, 6, 8]
    result5 = validate_replan(
        proposed_order   = proposed,
        all_waypoints    = all_wps,
        current_position = Position(0.0, 0.0),
        current_battery  = 40.0,
        base_position    = BASE,
        speed            = SPEED,
    )
    print(f"  Proposed order: {proposed}")
    print(f"  {result5.summary_str()}")
    print(f"  PASS (feasible={result5.is_feasible})")

    print("\nAll tests passed.")
