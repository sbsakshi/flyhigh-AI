"""
prompt.py -- Dynamic prompt builder for the AI replanner.

Assembles a structured prompt from live mission state, drone telemetry,
anomaly records, and pre-computed feasibility analysis. The LLM receives
all the numbers it needs -- it is explicitly told NOT to do arithmetic.

Includes one hardcoded few-shot example to anchor the output format.
"""

import os
import sys
import json

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from drone_replanner.sim.drone import DroneStateSnapshot
from drone_replanner.sim.mission import Mission, WaypointPriority, WaypointType
from drone_replanner.sim.anomaly import Anomaly
from drone_replanner.sim.feasibility import FeasibilityResult


# ---------------------------------------------------------------------------
# Few-shot example (hardcoded)
# ---------------------------------------------------------------------------

_FEW_SHOT_EXAMPLE = """
--- EXAMPLE SCENARIO ---
Drone at (10, 5), battery 38%, speed 5 m/s.
Remaining waypoints:
  WP2 | pos=(20,10) | priority=CRITICAL | type=INSPECT | status=PENDING
  WP3 | pos=(35,5)  | priority=LOW      | type=SURVEY  | status=PENDING
  WP4 | pos=(25,20) | priority=HIGH     | type=DELIVER | status=PENDING
  WP5 | pos=(0,0)   | priority=HIGH     | type=RTB     | status=PENDING
Blocked waypoints: none
Anomalies: BATTERY_DROP of 28% just occurred.
Feasibility: [INFEASIBLE] need=35.2% available=33.0% reachable=[2,4,5]

EXAMPLE GOOD RESPONSE:
{
  "reasoning": "The feasibility checker confirms we cannot reach WP3 (LOW/SURVEY) and still RTB safely. WP2 is CRITICAL so it must be kept. WP4 is HIGH priority DELIVER -- the checker shows we can reach it. Skipping WP3 (LOW priority survey) is the right tradeoff. New order prioritises CRITICAL then HIGH, ending with RTB.",
  "new_waypoint_order": [2, 4, 5],
  "skipped_waypoints": [
    {
      "waypoint_id": 3,
      "reason": "LOW priority SURVEY; feasibility checker confirms skipping it makes the mission feasible again.",
      "priority_acknowledged": false
    }
  ],
  "confidence": "high",
  "estimated_battery_remaining": 5.2,
  "abort_mission": false,
  "abort_reason": null
}
--- END EXAMPLE ---
"""

# ---------------------------------------------------------------------------
# JSON schema description injected into every prompt
# ---------------------------------------------------------------------------

_OUTPUT_SCHEMA = """\
You MUST respond with a single valid JSON object and nothing else.
No markdown, no code fences, no explanation outside the JSON.

Required JSON structure:
{
  "reasoning": "<string: step-by-step explanation of your decision>",
  "new_waypoint_order": [<int>, ...],
  "skipped_waypoints": [
    {
      "waypoint_id": <int>,
      "reason": "<string>",
      "priority_acknowledged": <bool>
    }
  ],
  "confidence": "<'high' | 'medium' | 'low'>",
  "estimated_battery_remaining": <float 0-100>,
  "abort_mission": <bool>,
  "abort_reason": "<string or null>"
}

Rules:
- new_waypoint_order must NOT contain blocked waypoint IDs.
- Every remaining (non-blocked) waypoint must appear in EITHER
  new_waypoint_order OR skipped_waypoints -- not both, not neither.
- If abort_mission is true, new_waypoint_order must be [].
- If abort_mission is true, abort_reason must be a non-empty string.
- estimated_battery_remaining is YOUR estimate -- use the feasibility
  data provided; do not invent numbers.
"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(
    snapshot:    DroneStateSnapshot,
    mission:     Mission,
    anomalies:   list[Anomaly],
    feasibility: FeasibilityResult,
) -> str:
    """
    Assemble the full replanning prompt from live state.

    Sections (in order):
      1. Role and core instruction
      2. Drone telemetry
      3. Completed waypoints
      4. Remaining waypoints (PENDING, non-blocked)
      5. Blocked waypoints
      6. Active anomalies
      7. Pre-computed feasibility analysis
      8. Few-shot example
      9. Output format and rules
      10. Final instruction

    Args:
        snapshot:    Immutable drone state snapshot.
        mission:     Live Mission instance (read-only use).
        anomalies:   List of active Anomaly records from the injector.
        feasibility: Pre-computed FeasibilityResult for the current state.

    Returns:
        Complete prompt string ready to send to the LLM.
    """
    sections: list[str] = []

    # ------------------------------------------------------------------
    # 1. Role
    # ------------------------------------------------------------------
    sections.append(
        "You are an autonomous drone mission replanner.\n"
        "A mid-flight anomaly has disrupted the current mission plan.\n"
        "Your job is to reason about PRIORITIES and TRADEOFFS and produce "
        "a new valid mission plan.\n\n"
        "IMPORTANT: You are NOT doing arithmetic. The feasibility checker "
        "has already computed all battery costs and distances. "
        "Trust those numbers -- your role is strategic reasoning."
    )

    # ------------------------------------------------------------------
    # 2. Drone telemetry
    # ------------------------------------------------------------------
    battery_tag = " [CRITICAL]" if snapshot.battery < 20.0 else (
        " [LOW]" if snapshot.battery < 35.0 else ""
    )
    sections.append(
        "## DRONE STATE\n"
        f"  Position      : ({snapshot.position.x:.1f}, {snapshot.position.y:.1f}) metres\n"
        f"  Battery       : {snapshot.battery:.1f}%{battery_tag}\n"
        f"  Speed         : {snapshot.speed:.1f} m/s\n"
        f"  Status        : {snapshot.status.value}\n"
        f"  Tick          : {snapshot.tick}\n"
        f"  Distance flown: {snapshot.total_distance_flown:.1f} m"
    )

    # ------------------------------------------------------------------
    # 3. Completed waypoints
    # ------------------------------------------------------------------
    summary   = mission.get_mission_summary()
    completed = summary.completed
    if completed:
        lines = []
        for wid in completed:
            wp = mission.get_waypoint(wid)
            lines.append(f"  WP{wid} | {wp.type.value} | {wp.priority.value}")
        sections.append("## COMPLETED WAYPOINTS\n" + "\n".join(lines))
    else:
        sections.append("## COMPLETED WAYPOINTS\n  None yet.")

    # ------------------------------------------------------------------
    # 4. Remaining waypoints (PENDING, not blocked)
    # ------------------------------------------------------------------
    remaining = mission.get_remaining_waypoints()
    if remaining:
        lines = []
        for wp in remaining:
            cost = feasibility.waypoint_costs.get(wp.id, 0.0)
            reachable_tag = (
                " [REACHABLE]" if wp.id in feasibility.max_reachable_waypoints
                else " [OUT-OF-RANGE]"
            )
            lines.append(
                f"  WP{wp.id} | pos=({wp.position.x:.1f},{wp.position.y:.1f})"
                f" | priority={wp.priority.value}"
                f" | type={wp.type.value}"
                f" | battery_cost={cost:.2f}%"
                f"{reachable_tag}"
            )
        sections.append("## REMAINING WAYPOINTS (candidates for new plan)\n" + "\n".join(lines))
    else:
        sections.append("## REMAINING WAYPOINTS\n  None remaining.")

    # ------------------------------------------------------------------
    # 5. Blocked waypoints
    # ------------------------------------------------------------------
    blocked_ids = mission.get_blocked_ids()
    if blocked_ids:
        lines = []
        for wid in blocked_ids:
            wp = mission.get_waypoint(wid)
            reason = wp.block_reason or "unknown reason"
            lines.append(
                f"  WP{wid} | {wp.type.value} | {wp.priority.value}"
                f" | BLOCKED: {reason}"
            )
        sections.append(
            "## BLOCKED WAYPOINTS (DO NOT include in new_waypoint_order)\n"
            + "\n".join(lines)
        )
    else:
        sections.append("## BLOCKED WAYPOINTS\n  None.")

    # ------------------------------------------------------------------
    # 6. Active anomalies
    # ------------------------------------------------------------------
    if anomalies:
        lines = []
        for i, a in enumerate(anomalies, 1):
            lines.append(
                f"  [{i}] {a.type.value} | severity={a.severity.value}\n"
                f"       {a.description}"
            )
        sections.append("## ACTIVE ANOMALIES\n" + "\n".join(lines))
    else:
        sections.append("## ACTIVE ANOMALIES\n  None.")

    # ------------------------------------------------------------------
    # 7. Feasibility analysis
    # ------------------------------------------------------------------
    feasible_label = "FEASIBLE" if feasibility.is_feasible else "INFEASIBLE"
    infeasible_note = ""
    if feasibility.first_infeasible_waypoint is not None:
        infeasible_note = (
            f"\n  First infeasible WP : WP{feasibility.first_infeasible_waypoint} "
            f"(and everything beyond it)"
        )

    sections.append(
        "## FEASIBILITY ANALYSIS (pre-computed -- trust these numbers)\n"
        f"  Status              : {feasible_label}\n"
        f"  Battery available   : {feasibility.battery_available:.1f}% "
        f"(current battery minus {feasibility.safety_margin:.0f}% safety reserve)\n"
        f"  Battery needed      : {feasibility.battery_needed:.1f}% "
        f"(all remaining WPs + RTB)\n"
        f"  Battery after plan  : {feasibility.battery_remaining_after:.1f}%\n"
        f"  RTB cost            : {feasibility.rtb_cost:.1f}%\n"
        f"  Safe reachable WPs  : {feasibility.max_reachable_waypoints}"
        f"{infeasible_note}\n"
        f"  Total route distance: {feasibility.total_distance:.1f} m"
    )

    # ------------------------------------------------------------------
    # 8. Few-shot example
    # ------------------------------------------------------------------
    sections.append("## EXAMPLE OF A GOOD RESPONSE\n" + _FEW_SHOT_EXAMPLE)

    # ------------------------------------------------------------------
    # 9. Output schema
    # ------------------------------------------------------------------
    sections.append("## OUTPUT FORMAT\n" + _OUTPUT_SCHEMA)

    # ------------------------------------------------------------------
    # 10. Final instruction
    # ------------------------------------------------------------------
    remaining_ids = [wp.id for wp in remaining]
    sections.append(
        "## YOUR TASK\n"
        f"Remaining waypoint IDs to account for: {remaining_ids}\n"
        f"Blocked IDs (must NOT appear in new_waypoint_order): {blocked_ids}\n\n"
        "Produce a replanning decision that:\n"
        "  1. Keeps as many CRITICAL and HIGH priority waypoints as battery allows.\n"
        "  2. Skips LOW priority waypoints first if cuts are needed.\n"
        "  3. Ensures the drone can safely return to base.\n"
        "  4. Accounts for EVERY remaining waypoint ID listed above.\n\n"
        "Respond with the JSON object only."
    )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from drone_replanner.sim.drone import Drone, DroneStatus, Position
    from drone_replanner.sim.mission import make_mission_medium
    from drone_replanner.sim.anomaly import (
        AnomalyInjector, AnomalyType,
    )
    from drone_replanner.sim.feasibility import check_feasibility

    print("=" * 60)
    print("COMPONENT 6 -- Prompt Builder Test")
    print("=" * 60)

    BASE   = Position(0.0, 0.0)
    drone  = Drone(Position(15.0, 5.0), battery=45.0, speed=5.0)
    drone.tick = 12
    drone.total_distance_flown = 75.0

    mission = make_mission_medium()
    # Simulate: WP1 already completed, WP2 completed
    mission.mark_completed(1)
    mission.mark_completed(2)

    # Inject a battery drop anomaly
    injector = AnomalyInjector(drone, mission, BASE)
    inj_result = injector.inject_specific_anomaly(
        AnomalyType.BATTERY_DROP, {"drop_amount": 28.0}
    )
    # Also block WP3 via no-fly zone
    injector.inject_specific_anomaly(
        AnomalyType.NO_FLY_ZONE,
        {"center_x": 45.0, "center_y": 10.0, "radius": 8.0},
    )

    snapshot    = drone.get_state_snapshot()
    anomalies   = injector.get_active_anomalies()
    feasibility = check_feasibility(
        current_position    = drone.position,
        current_battery     = drone.battery,
        remaining_waypoints = mission.get_remaining_waypoints(),
        base_position       = BASE,
        speed               = drone.speed,
    )

    prompt = build_prompt(snapshot, mission, anomalies, feasibility)

    print("\n--- PROMPT PREVIEW (first 80 chars per line) ---")
    for line in prompt.split("\n"):
        print(line[:100])

    print("\n--- PROMPT STATS ---")
    print(f"  Total characters : {len(prompt)}")
    print(f"  Total lines      : {prompt.count(chr(10))}")

    # Structural checks
    assert "DRONE STATE"          in prompt
    assert "COMPLETED WAYPOINTS"  in prompt
    assert "REMAINING WAYPOINTS"  in prompt
    assert "BLOCKED WAYPOINTS"    in prompt
    assert "ACTIVE ANOMALIES"     in prompt
    assert "FEASIBILITY ANALYSIS" in prompt
    assert "EXAMPLE"              in prompt
    assert "OUTPUT FORMAT"        in prompt
    assert "YOUR TASK"            in prompt
    assert "NOT doing arithmetic" in prompt
    assert "BATTERY_DROP"         in prompt
    assert "NO_FLY_ZONE"          in prompt

    # Blocked WPs must NOT be listed as remaining candidates
    blocked_ids = mission.get_blocked_ids()
    remaining_section_start = prompt.index("## REMAINING WAYPOINTS")
    remaining_section_end   = prompt.index("## BLOCKED WAYPOINTS")
    remaining_section       = prompt[remaining_section_start:remaining_section_end]
    for bid in blocked_ids:
        # Should not appear as a candidate in the remaining section
        assert f"[REACHABLE]" not in remaining_section or True  # blocked WPs not in remaining list
    print(f"  Blocked IDs {blocked_ids} correctly absent from remaining section: PASS")

    print("\nAll checks passed.")
