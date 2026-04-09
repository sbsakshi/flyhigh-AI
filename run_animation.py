"""
run_animation.py -- Run a full simulation and render an animated replay.

Usage:
    python run_animation.py                       # GIF, medium mission
    python run_animation.py --mission hard        # GIF, hard mission
    python run_animation.py --live                # interactive window + slider
    python run_animation.py --no-anomaly          # baseline (no replanning)
"""

import argparse
import os

from drone_replanner.sim.drone import Drone, Position
from drone_replanner.sim.mission import (
    make_mission_easy, make_mission_medium, make_mission_hard,
)
from drone_replanner.sim.anomaly import AnomalyInjector, AnomalyType
from drone_replanner.viz.animated_visualizer import play_simulation
from main import run_simulation, BASE_POSITION, ANOMALY_TICK, SECOND_ANOMALY_TICK


MISSION_FACTORIES = {
    "easy":   make_mission_easy,
    "medium": make_mission_medium,
    "hard":   make_mission_hard,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", choices=list(MISSION_FACTORIES), default="medium")
    ap.add_argument("--model",   default="llama-3.3-70b-versatile")
    ap.add_argument("--no-anomaly", action="store_true")
    ap.add_argument("--live", action="store_true",
                    help="Interactive window with speed slider (no GIF).")
    ap.add_argument("--out", default="mission_replay.gif")
    args = ap.parse_args()

    # Build mission + drone + injector (mirrors main.py)
    mission  = MISSION_FACTORIES[args.mission]()
    drone    = Drone(start_position=BASE_POSITION, battery=100.0, speed=5.0)
    injector = AnomalyInjector(drone, mission, BASE_POSITION)

    if args.no_anomaly:
        anomaly_schedule = {}
    else:
        anomaly_schedule = {
            ANOMALY_TICK: {
                "type":   AnomalyType.BATTERY_DROP,
                "params": {"drop_amount": 55.0},
            },
            SECOND_ANOMALY_TICK: {
                "type":   AnomalyType.NO_FLY_ZONE,
                "params": {"center_x": 43.0, "center_y": 15.0, "radius": 14.0},
            },
        }

    original_order = [wp.id for wp in mission.get_remaining_waypoints()]

    print(f"[RUN] Simulating {args.mission} mission...")
    mission, drone, replan_events = run_simulation(
        mission       = mission,
        drone         = drone,
        injector      = injector,
        anomaly_ticks = anomaly_schedule,
        api_key       = os.environ.get("GROQ_API_KEY"),
        model         = args.model,
        verbose       = True,
    )

    print(f"\n[ANIM] Path samples: {len(drone.path_history)}  "
          f"Replans: {len(replan_events)}")

    play_simulation(
        mission        = mission,
        drone          = drone,
        replan_events  = replan_events,
        no_fly_zones   = injector.get_no_fly_zones(),
        original_order = original_order,
        base_position  = BASE_POSITION,
        save_gif_path  = None if args.live else args.out,
        live           = args.live,
    )

    if not args.live and os.path.exists(args.out):
        size_kb = os.path.getsize(args.out) / 1024
        print(f"[DONE] {args.out} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
