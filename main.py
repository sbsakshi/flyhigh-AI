"""
main.py -- Main simulation loop for the drone mission replanning system.

Wires together all components:
  - Drone simulation (discrete 1-second ticks)
  - Mission engine (waypoint state management)
  - Anomaly injector (mid-flight disruptions)
  - Feasibility checker (battery safety gate)
  - AI replanner (Groq LLM + fallback)
  - Console reporting (every 5 ticks + end summary)

Termination conditions:
  - All waypoints completed (including RTB)
  - Battery drops below critical threshold (< 20%)
  - LLM returns abort_mission = True
  - Max tick limit reached
"""

import os
import sys
import logging
from dataclasses import dataclass, field

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(__file__))

from drone_replanner.sim.drone import Drone, DroneStatus, Position
from drone_replanner.sim.mission import (
    Mission, Waypoint, WaypointStatus, WaypointType,
    make_mission_easy, make_mission_medium, make_mission_hard,
)
from drone_replanner.sim.anomaly import AnomalyInjector, AnomalyType
from drone_replanner.sim.feasibility import check_feasibility
from drone_replanner.ai.replanner import run_replanner, DEFAULT_MODEL
from drone_replanner.ai.schemas import ReplanResult
from drone_replanner.viz.visualizer import visualize_simulation

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,          # Suppress lower-level noise in main loop
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("MAIN")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Simulation config
# ---------------------------------------------------------------------------
MAX_TICKS:           int   = 500    # Hard safety limit
PRINT_INTERVAL:      int   = 5      # Console status every N ticks
ANOMALY_TICK:        int   = 15     # Tick at which first anomaly fires
SECOND_ANOMALY_TICK: int   = 35     # Optional second anomaly
BASE_POSITION:       Position = Position(0.0, 0.0)


# ---------------------------------------------------------------------------
# Replanning event record
# ---------------------------------------------------------------------------

@dataclass
class ReplanEvent:
    """Record of a single replanning cycle during the simulation."""
    tick:           int
    trigger:        str          # "anomaly" or "critical_battery"
    anomaly_desc:   str
    feasibility_before: str      # summary_str snapshot
    result:         ReplanResult
    new_order:      list[int]
    skipped:        list[int]


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def run_simulation(
    mission:        Mission,
    drone:          Drone,
    injector:       AnomalyInjector,
    anomaly_ticks:  dict[int, dict],   # tick -> {type, params}
    api_key:        str | None = None,
    model:          str = DEFAULT_MODEL,
    verbose:        bool = True,
) -> tuple[Mission, Drone, list[ReplanEvent]]:
    """
    Execute the full drone mission simulation.

    Args:
        mission:       Initialised Mission object.
        drone:         Initialised Drone object.
        injector:      AnomalyInjector wired to drone + mission.
        anomaly_ticks: Mapping of tick number to anomaly spec.
                       e.g. {15: {"type": AnomalyType.BATTERY_DROP,
                                  "params": {"drop_amount": 30}}}
        api_key:       Groq API key (None = use env var).
        model:         Groq model to use for replanning.
        verbose:       Print status to console.

    Returns:
        Tuple of (mission, drone, list_of_replan_events) at end of sim.
    """
    replan_events: list[ReplanEvent] = []
    waypoint_queue = mission.get_remaining_waypoints()

    if verbose:
        _print_header(mission, drone)

    for tick in range(MAX_TICKS):

        # ------------------------------------------------------------------
        # Check termination
        # ------------------------------------------------------------------
        if drone.status in (DroneStatus.COMPLETE, DroneStatus.ABORTED):
            break

        remaining = mission.get_remaining_waypoints()
        if not remaining:
            drone.set_complete()
            break

        if drone.is_battery_critical():
            logger.warning("Tick %03d | Battery critical (%.1f%%) -- aborting.", tick, drone.battery)
            drone.set_aborted()
            break

        # ------------------------------------------------------------------
        # Anomaly injection
        # ------------------------------------------------------------------
        if tick in anomaly_ticks:
            spec = anomaly_ticks[tick]
            if verbose:
                print(f"\n{'!'*60}")
                print(f"  TICK {tick:03d} | ANOMALY TRIGGERED: {spec['type'].value}")
                print(f"{'!'*60}")

            inj_result = injector.inject_specific_anomaly(spec["type"], spec["params"])
            remaining   = mission.get_remaining_waypoints()    # refresh after blocking

            if verbose:
                print(f"  {inj_result.anomaly.description}")
                print(f"  Feasibility: {inj_result.feasibility.summary_str()}")

            # Gate: only replan if mission is infeasible OR waypoints were blocked
            if inj_result.replanning_needed:
                if verbose:
                    print(f"\n  >> Replanning triggered (tick {tick})...")
                drone.set_replanning()

                feasibility = check_feasibility(
                    current_position    = drone.position,
                    current_battery     = drone.battery,
                    remaining_waypoints = remaining,
                    base_position       = BASE_POSITION,
                    speed               = drone.speed,
                )
                snapshot  = drone.get_state_snapshot()
                anomalies = injector.get_active_anomalies()

                replan_result = run_replanner(
                    snapshot    = snapshot,
                    mission     = mission,
                    anomalies   = anomalies,
                    feasibility = feasibility,
                    api_key     = api_key,
                    model       = model,
                )

                decision = replan_result.decision

                if verbose:
                    _print_replan_result(replan_result, tick)

                # Apply abort
                if decision.abort_mission:
                    logger.warning("Tick %03d | LLM ordered abort: %s", tick, decision.abort_reason)
                    drone.set_aborted()
                    replan_events.append(ReplanEvent(
                        tick=tick,
                        trigger="anomaly+abort",
                        anomaly_desc=inj_result.anomaly.description,
                        feasibility_before=feasibility.summary_str(),
                        result=replan_result,
                        new_order=[],
                        skipped=[s.waypoint_id for s in decision.skipped_waypoints],
                    ))
                    break

                # Apply new order
                mission.update_order(decision.new_waypoint_order)

                # Mark explicitly skipped waypoints
                for sw in decision.skipped_waypoints:
                    wp = mission.get_waypoint(sw.waypoint_id)
                    if wp.status == WaypointStatus.PENDING:
                        mission.mark_skipped(sw.waypoint_id, sw.reason)

                # Refresh waypoint queue from new plan
                waypoint_queue = mission.get_remaining_waypoints()

                replan_events.append(ReplanEvent(
                    tick=tick,
                    trigger="anomaly",
                    anomaly_desc=inj_result.anomaly.description,
                    feasibility_before=feasibility.summary_str(),
                    result=replan_result,
                    new_order=decision.new_waypoint_order,
                    skipped=[s.waypoint_id for s in decision.skipped_waypoints],
                ))

                drone.status = DroneStatus.FLYING

        # ------------------------------------------------------------------
        # Move drone toward current target waypoint
        # ------------------------------------------------------------------
        waypoint_queue = mission.get_remaining_waypoints()
        if not waypoint_queue:
            drone.set_complete()
            break

        target_wp = waypoint_queue[0]
        target_positions = [wp.position for wp in waypoint_queue]

        reached = drone.move_to_next_waypoint(target_positions)

        if reached:
            mission.mark_completed(target_wp.id)
            drone.current_waypoint_index += 1
            waypoint_queue = mission.get_remaining_waypoints()

            if verbose:
                wp = target_wp
                print(
                    f"  Tick {drone.tick:03d} | Reached WP{wp.id} "
                    f"({wp.type.value}/{wp.priority.value}) | "
                    f"battery={drone.battery:.1f}%"
                )

            # RTB waypoint reached = mission complete
            if target_wp.type == WaypointType.RTB:
                drone.set_complete()
                break

        # ------------------------------------------------------------------
        # Periodic status print
        # ------------------------------------------------------------------
        if verbose and drone.tick % PRINT_INTERVAL == 0 and drone.tick > 0:
            _print_status(drone, mission)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    if verbose:
        _print_summary(drone, mission, replan_events)

    return mission, drone, replan_events


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _print_header(mission: Mission, drone: Drone) -> None:
    """Print mission start banner."""
    wps = mission.get_remaining_waypoints()
    print("\n" + "=" * 60)
    print(f"  MISSION: {mission.name}")
    print(f"  Waypoints: {len(wps)}")
    print(f"  Drone start: {drone.position}  battery={drone.battery:.0f}%")
    print("=" * 60)
    print("  ID  | Position       | Priority | Type")
    print("  " + "-" * 46)
    for wp in wps:
        print(
            f"  WP{wp.id:<2} | ({wp.position.x:5.1f},{wp.position.y:5.1f})    "
            f"| {wp.priority.value:<8} | {wp.type.value}"
        )
    print("=" * 60 + "\n")


def _print_status(drone: Drone, mission: Mission) -> None:
    """Print a compact one-line status every PRINT_INTERVAL ticks."""
    summary  = mission.get_mission_summary()
    remaining = mission.get_remaining_waypoints()
    rem_ids  = [wp.id for wp in remaining]
    print(
        f"  [Tick {drone.tick:03d}] pos=({drone.position.x:5.1f},{drone.position.y:5.1f}) "
        f"batt={drone.battery:5.1f}% "
        f"done={summary.completed} "
        f"remaining={rem_ids}"
    )


def _print_replan_result(result: ReplanResult, tick: int) -> None:
    """Print the replanning decision details."""
    d = result.decision
    print(f"\n  --- REPLANNING DECISION (tick {tick}) ---")
    print(f"  Model:      {result.model_used}")
    print(f"  Latency:    {result.latency_ms:.0f} ms")
    print(f"  Retries:    {result.retry_count}")
    print(f"  Fallback:   {result.used_fallback}")
    print(f"  Confidence: {d.confidence}")
    print(f"  New order:  {d.new_waypoint_order}")
    print(f"  Skipped:    {[s.waypoint_id for s in d.skipped_waypoints]}")
    print(f"  Est. batt:  {d.estimated_battery_remaining:.1f}% remaining after mission")
    print(f"  Reasoning:  {d.reasoning[:200]}")
    print(f"  {'-'*40}\n")


def _print_summary(
    drone: Drone,
    mission: Mission,
    replan_events: list[ReplanEvent],
) -> None:
    """Print the full end-of-mission summary."""
    summary = mission.get_mission_summary()
    print("\n" + "=" * 60)
    print("  MISSION COMPLETE -- FINAL SUMMARY")
    print("=" * 60)
    print(f"  Status:              {drone.status.value.upper()}")
    print(f"  Ticks elapsed:       {drone.tick}")
    print(f"  Distance flown:      {drone.total_distance_flown:.1f} m")
    print(f"  Battery remaining:   {drone.battery:.1f}%")
    print(f"  Battery consumed:    {drone.battery_consumed:.1f}%")
    print()
    print(f"  Total waypoints:     {summary.total_waypoints}")
    print(f"  Completed:           {summary.completed}")
    print(f"  Skipped:             {summary.skipped}")
    print(f"  Blocked:             {summary.blocked}")
    print(f"  Completion rate:     {summary.completion_rate * 100:.1f}%")
    print(f"  Critical completed:  {summary.critical_completed}")
    print(f"  Critical skipped:    {summary.critical_skipped}")
    print(f"  Replanning events:   {summary.replanning_count}")
    print()

    if replan_events:
        print("  REPLANNING TRACE:")
        for i, ev in enumerate(replan_events, 1):
            print(f"  [{i}] Tick {ev.tick:03d} | trigger={ev.trigger}")
            print(f"       Anomaly: {ev.anomaly_desc}")
            print(f"       Before:  {ev.feasibility_before}")
            print(f"       Model:   {ev.result.model_used} | "
                  f"latency={ev.result.latency_ms:.0f}ms | "
                  f"fallback={ev.result.used_fallback}")
            print(f"       New order: {ev.new_order}")
            print(f"       Skipped:   {ev.skipped}")
            print(f"       Reasoning: {ev.result.decision.reasoning[:150]}")
            print()

    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drone Mission Replanning Simulation")
    parser.add_argument(
        "--mission", choices=["easy", "medium", "hard"], default="medium",
        help="Mission complexity (default: medium)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Groq model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--no-anomaly", action="store_true",
        help="Run without any anomalies (baseline test)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Save a mission visualisation PNG after simulation",
    )
    parser.add_argument(
        "--plot-path", default="mission_plot.png",
        help="Output path for the plot (default: mission_plot.png)",
    )
    args = parser.parse_args()

    # Build mission + drone
    mission_map = {
        "easy":   make_mission_easy,
        "medium": make_mission_medium,
        "hard":   make_mission_hard,
    }
    mission = mission_map[args.mission]()
    drone   = Drone(start_position=BASE_POSITION, battery=100.0, speed=5.0)
    injector = AnomalyInjector(drone, mission, BASE_POSITION)

    # Define anomaly schedule
    if args.no_anomaly:
        anomaly_schedule: dict = {}
    else:
        anomaly_schedule = {
            ANOMALY_TICK: {
                "type":   AnomalyType.BATTERY_DROP,
                "params": {"drop_amount": 55.0},   # aggressive drop to force replan
            },
            SECOND_ANOMALY_TICK: {
                "type":   AnomalyType.NO_FLY_ZONE,
                "params": {"center_x": 43.0, "center_y": 15.0, "radius": 14.0},  # blocks WPs
            },
        }

    api_key = os.environ.get("GROQ_API_KEY")

    # Capture original order before simulation mutates it
    original_order = [wp.id for wp in mission.get_remaining_waypoints()]

    mission, drone, replan_events = run_simulation(
        mission       = mission,
        drone         = drone,
        injector      = injector,
        anomaly_ticks = anomaly_schedule,
        api_key       = api_key,
        model         = args.model,
        verbose       = True,
    )

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        fig = visualize_simulation(
            mission        = mission,
            drone          = drone,
            replan_events  = replan_events,
            no_fly_zones   = injector.get_no_fly_zones(),
            original_order = original_order,
            save_path      = args.plot_path,
            show           = False,
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        print(f"\n[VIZ] Plot saved to: {args.plot_path}")
