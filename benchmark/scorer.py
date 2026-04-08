"""
benchmark/scorer.py -- Multi-model benchmark scorer for the AI replanner.

Runs a fixed set of scenarios through each configured Groq model and
scores each ReplanDecision on four metrics:

  valid_json       -- Schema valid on first LLM attempt (no retries needed)
  feasible_plan    -- Plan passes feasibility checker post-hoc
  priority_respect -- Ratio of CRITICAL+HIGH waypoints kept (not skipped)
  safe_return      -- Estimated battery >= safety margin after mission

Produces:
  - A pandas DataFrame with per-scenario and per-model scores
  - A bar chart comparing models (saved to PNG)
  - A printed leaderboard summary
"""

import os
import sys
import time
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from drone_replanner.sim.drone import Drone, Position
from drone_replanner.sim.mission import (
    Mission, Waypoint, WaypointPriority, WaypointType, WaypointStatus,
    make_mission_easy, make_mission_medium, make_mission_hard,
)
from drone_replanner.sim.anomaly import AnomalyInjector, AnomalyType
from drone_replanner.sim.feasibility import (
    check_feasibility, validate_replan, SAFETY_MARGIN, FeasibilityResult,
)
from drone_replanner.ai.prompt import build_prompt
from drone_replanner.ai.replanner import run_replanner, DEFAULT_MODEL
from drone_replanner.ai.schemas import ReplanDecision, ReplanResult

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [BENCH] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BENCH")
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------
WEIGHTS: dict[str, float] = {
    "valid_json":       0.25,
    "feasible_plan":    0.35,
    "priority_respect": 0.25,
    "safe_return":      0.15,
}

# ---------------------------------------------------------------------------
# Models to benchmark
# ---------------------------------------------------------------------------
BENCHMARK_MODELS: list[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

BASE_POSITION = Position(0.0, 0.0)


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkScenario:
    """
    A single reproducible benchmark scenario.

    Attributes:
        name:            Human-readable scenario name.
        mission_factory: Callable that creates a fresh Mission.
        drone_pos:       Drone position at replanning time.
        drone_battery:   Drone battery % at replanning time.
        anomalies:       List of (AnomalyType, params) to inject in order.
    """
    name:            str
    mission_factory: callable
    drone_pos:       Position
    drone_battery:   float
    completed_ids:   list[int]       # WP IDs already completed before the anomaly
    anomalies:       list[tuple]     # [(AnomalyType, params_dict), ...]


@dataclass
class ScenarioScore:
    """
    Scores for a single (scenario, model) combination.
    """
    scenario:         str
    model:            str
    valid_json:       float    # 1.0 or 0.0
    feasible_plan:    float    # 1.0 or 0.0
    priority_respect: float    # 0.0 – 1.0
    safe_return:      float    # 1.0 or 0.0
    composite_score:  float    # weighted average
    latency_ms:       float
    retry_count:      int
    used_fallback:    bool
    error:            Optional[str] = None


# ---------------------------------------------------------------------------
# Hardcoded scenarios
# ---------------------------------------------------------------------------

def get_benchmark_scenarios() -> list[BenchmarkScenario]:
    """
    Return the five fixed benchmark scenarios.

    Scenarios are deterministic -- same drone state, same anomalies,
    same mission every run. Only the model changes.
    """
    return [
        BenchmarkScenario(
            name            = "S1-Easy-BatteryDrop",
            mission_factory = make_mission_easy,
            drone_pos       = Position(20.0, 10.0),
            drone_battery   = 40.0,
            completed_ids   = [1, 2],
            anomalies       = [
                (AnomalyType.BATTERY_DROP, {"drop_amount": 28.0}),
            ],
        ),
        BenchmarkScenario(
            name            = "S2-Easy-NoFlyZone",
            mission_factory = make_mission_easy,
            drone_pos       = Position(10.0, 0.0),
            drone_battery   = 80.0,
            completed_ids   = [1],
            anomalies       = [
                (AnomalyType.NO_FLY_ZONE, {"center_x": 25.0, "center_y": 12.0, "radius": 10.0}),
            ],
        ),
        BenchmarkScenario(
            name            = "S3-Medium-StackedAnomalies",
            mission_factory = make_mission_medium,
            drone_pos       = Position(30.0, 0.0),
            drone_battery   = 55.0,
            completed_ids   = [1, 2],
            anomalies       = [
                (AnomalyType.BATTERY_DROP, {"drop_amount": 25.0}),
                (AnomalyType.NO_FLY_ZONE, {"center_x": 45.0, "center_y": 10.0, "radius": 8.0}),
            ],
        ),
        BenchmarkScenario(
            name            = "S4-Medium-WaypointFailure",
            mission_factory = make_mission_medium,
            drone_pos       = Position(15.0, 5.0),
            drone_battery   = 60.0,
            completed_ids   = [1],
            anomalies       = [
                (AnomalyType.WAYPOINT_FAILURE, {"waypoint_id": 3}),
                (AnomalyType.WAYPOINT_FAILURE, {"waypoint_id": 6}),
            ],
        ),
        BenchmarkScenario(
            name            = "S5-Hard-CriticalBattery",
            mission_factory = make_mission_hard,
            drone_pos       = Position(35.0, 10.0),
            drone_battery   = 38.0,
            completed_ids   = [1, 2, 3],
            anomalies       = [
                (AnomalyType.BATTERY_DROP, {"drop_amount": 15.0}),
                (AnomalyType.NO_FLY_ZONE, {"center_x": 55.0, "center_y": 30.0, "radius": 12.0}),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------

def run_scenario(
    scenario: BenchmarkScenario,
    model:    str,
    api_key:  str,
) -> ScenarioScore:
    """
    Execute one (scenario, model) combination and return scores.

    Sets up mission state, injects anomalies, calls the replanner,
    then scores the result.

    Args:
        scenario: BenchmarkScenario to execute.
        model:    Groq model ID.
        api_key:  Groq API key.

    Returns:
        ScenarioScore with all four metrics + metadata.
    """
    try:
        # -- Build fresh mission + drone --
        mission = scenario.mission_factory()
        drone   = Drone(scenario.drone_pos, battery=scenario.drone_battery, speed=5.0)
        drone.position = scenario.drone_pos

        # Mark completed waypoints
        for wid in scenario.completed_ids:
            try:
                mission.mark_completed(wid)
            except ValueError:
                pass  # Already completed or wrong state -- ignore in benchmark

        # -- Inject anomalies --
        injector = AnomalyInjector(drone, mission, BASE_POSITION)
        for atype, aparams in scenario.anomalies:
            injector.inject_specific_anomaly(atype, aparams)

        # -- Compute feasibility --
        remaining   = mission.get_remaining_waypoints()
        feasibility = check_feasibility(
            current_position    = drone.position,
            current_battery     = drone.battery,
            remaining_waypoints = remaining,
            base_position       = BASE_POSITION,
            speed               = drone.speed,
        )

        snapshot  = drone.get_state_snapshot()
        anomalies = injector.get_active_anomalies()

        # -- Call replanner --
        t0 = time.perf_counter()
        result = run_replanner(
            snapshot    = snapshot,
            mission     = mission,
            anomalies   = anomalies,
            feasibility = feasibility,
            api_key     = api_key,
            model       = model,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        # -- Score --
        scores = _score_decision(
            result      = result,
            mission     = mission,
            feasibility = feasibility,
            drone       = drone,
        )

        return ScenarioScore(
            scenario         = scenario.name,
            model            = model,
            valid_json       = scores["valid_json"],
            feasible_plan    = scores["feasible_plan"],
            priority_respect = scores["priority_respect"],
            safe_return      = scores["safe_return"],
            composite_score  = scores["composite_score"],
            latency_ms       = round(latency_ms, 1),
            retry_count      = result.retry_count,
            used_fallback    = result.used_fallback,
        )

    except Exception as exc:
        logger.error("Scenario %s / model %s failed: %s", scenario.name, model, exc)
        return ScenarioScore(
            scenario         = scenario.name,
            model            = model,
            valid_json       = 0.0,
            feasible_plan    = 0.0,
            priority_respect = 0.0,
            safe_return      = 0.0,
            composite_score  = 0.0,
            latency_ms       = 0.0,
            retry_count      = 0,
            used_fallback    = True,
            error            = str(exc)[:120],
        )


# ---------------------------------------------------------------------------
# Scoring logic
# ---------------------------------------------------------------------------

def _score_decision(
    result:      ReplanResult,
    mission:     Mission,
    feasibility: FeasibilityResult,
    drone:       Drone,
) -> dict[str, float]:
    """
    Compute the four scoring metrics for a ReplanResult.

    Args:
        result:      ReplanResult from the replanner.
        mission:     Mission object (post-anomaly state).
        feasibility: Pre-computed feasibility for the remaining plan.
        drone:       Drone state at decision time.

    Returns:
        Dict with keys: valid_json, feasible_plan, priority_respect,
        safe_return, composite_score.
    """
    decision = result.decision
    remaining = mission.get_remaining_waypoints()
    blocked   = mission.get_blocked_ids()
    all_wps   = {wp.id: wp for wp in mission.get_all_waypoints()}

    # 1. valid_json: no retries and no fallback on first attempt
    valid_json = 1.0 if (result.retry_count == 0 and not result.used_fallback) else 0.0

    # 2. feasible_plan: post-hoc feasibility check on proposed order
    if decision.abort_mission:
        feasible_plan = 1.0   # Abort is always safe
    else:
        proposed_wps = [
            all_wps[wid] for wid in decision.new_waypoint_order
            if wid in all_wps
        ]
        post_feasibility = check_feasibility(
            current_position    = drone.position,
            current_battery     = drone.battery,
            remaining_waypoints = proposed_wps,
            base_position       = BASE_POSITION,
            speed               = drone.speed,
        )
        feasible_plan = 1.0 if post_feasibility.is_feasible else 0.0

    # 3. priority_respect: ratio of CRITICAL+HIGH waypoints preserved
    high_priority_remaining = [
        wp for wp in remaining
        if wp.priority in (WaypointPriority.CRITICAL, WaypointPriority.HIGH)
        and wp.id not in blocked
    ]
    if not high_priority_remaining:
        priority_respect = 1.0
    else:
        skipped_high = [
            sw for sw in decision.skipped_waypoints
            if sw.waypoint_id in {wp.id for wp in high_priority_remaining}
        ]
        kept = len(high_priority_remaining) - len(skipped_high)
        priority_respect = round(max(0.0, kept / len(high_priority_remaining)), 3)

    # 4. safe_return: estimated battery >= safety margin
    safe_return = 1.0 if decision.estimated_battery_remaining >= SAFETY_MARGIN else 0.0

    # 5. Composite (weighted average)
    composite_score = round(
        WEIGHTS["valid_json"]       * valid_json
        + WEIGHTS["feasible_plan"]    * feasible_plan
        + WEIGHTS["priority_respect"] * priority_respect
        + WEIGHTS["safe_return"]      * safe_return,
        4,
    )

    return {
        "valid_json":       valid_json,
        "feasible_plan":    feasible_plan,
        "priority_respect": priority_respect,
        "safe_return":      safe_return,
        "composite_score":  composite_score,
    }


# ---------------------------------------------------------------------------
# Full benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    models:    list[str] = BENCHMARK_MODELS,
    scenarios: list[BenchmarkScenario] | None = None,
    api_key:   str | None = None,
    save_plot: str | None = "benchmark_results.png",
    save_csv:  str | None = "benchmark_results.csv",
) -> pd.DataFrame:
    """
    Run all scenarios through all models and return a scored DataFrame.

    Args:
        models:    List of Groq model IDs to benchmark.
        scenarios: List of BenchmarkScenario (defaults to all 5).
        api_key:   Groq API key (falls back to GROQ_API_KEY env var).
        save_plot: Path to save the bar chart (None = skip).
        save_csv:  Path to save the results CSV (None = skip).

    Returns:
        pandas DataFrame with one row per (scenario, model) combination.
    """
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not scenarios:
        scenarios = get_benchmark_scenarios()

    all_scores: list[ScenarioScore] = []
    total = len(models) * len(scenarios)
    done  = 0

    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {len(scenarios)} scenarios x {len(models)} models = {total} runs")
    print(f"{'='*60}\n")

    for model in models:
        print(f"  Model: {model}")
        for scenario in scenarios:
            print(f"    [{done+1}/{total}] {scenario.name} ... ", end="", flush=True)
            score = run_scenario(scenario, model, key)
            all_scores.append(score)
            done += 1
            status = "FALLBACK" if score.used_fallback else f"{score.composite_score:.2f}"
            print(f"{status}  ({score.latency_ms:.0f}ms, retries={score.retry_count})")

    # -- Build DataFrame --
    df = pd.DataFrame([
        {
            "scenario":         s.scenario,
            "model":            s.model,
            "valid_json":       s.valid_json,
            "feasible_plan":    s.feasible_plan,
            "priority_respect": s.priority_respect,
            "safe_return":      s.safe_return,
            "composite_score":  s.composite_score,
            "latency_ms":       s.latency_ms,
            "retry_count":      s.retry_count,
            "used_fallback":    s.used_fallback,
            "error":            s.error or "",
        }
        for s in all_scores
    ])

    # -- Print summary table --
    _print_summary(df)

    # -- Save CSV --
    if save_csv:
        df.to_csv(save_csv, index=False)
        print(f"\n  Results saved to: {save_csv}")

    # -- Plot --
    if save_plot:
        _plot_results(df, save_path=save_plot)
        print(f"  Chart saved to:   {save_plot}")

    return df


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    """Print the leaderboard and per-metric averages."""
    print(f"\n{'='*60}")
    print("  BENCHMARK RESULTS")
    print(f"{'='*60}")

    # Aggregate by model
    agg = (
        df.groupby("model")
        .agg(
            avg_composite  = ("composite_score",  "mean"),
            avg_valid_json = ("valid_json",        "mean"),
            avg_feasible   = ("feasible_plan",     "mean"),
            avg_priority   = ("priority_respect",  "mean"),
            avg_safe       = ("safe_return",       "mean"),
            avg_latency    = ("latency_ms",        "mean"),
            fallback_rate  = ("used_fallback",     "mean"),
        )
        .sort_values("avg_composite", ascending=False)
        .reset_index()
    )

    print(f"\n  {'Model':<35} {'Composite':>9} {'ValidJSON':>9} "
          f"{'Feasible':>9} {'Priority':>9} {'SafeRTB':>8} {'Latency':>8} {'Fallback%':>9}")
    print("  " + "-" * 100)

    for _, row in agg.iterrows():
        print(
            f"  {row['model']:<35} "
            f"{row['avg_composite']:>9.3f} "
            f"{row['avg_valid_json']:>9.2f} "
            f"{row['avg_feasible']:>9.2f} "
            f"{row['avg_priority']:>9.2f} "
            f"{row['avg_safe']:>8.2f} "
            f"{row['avg_latency']:>7.0f}ms "
            f"{row['fallback_rate']*100:>8.0f}%"
        )

    print(f"\n  Weights: {WEIGHTS}")
    print(f"{'='*60}\n")


def _plot_results(df: pd.DataFrame, save_path: str) -> None:
    """
    Render a grouped bar chart comparing models across all four metrics.

    Args:
        df:        Results DataFrame.
        save_path: Output PNG path.
    """
    metrics = ["valid_json", "feasible_plan", "priority_respect", "safe_return", "composite_score"]
    metric_labels = ["Valid JSON", "Feasible Plan", "Priority Respect", "Safe Return", "Composite"]

    model_avgs = (
        df.groupby("model")[metrics]
        .mean()
        .reset_index()
    )
    models_list = model_avgs["model"].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Drone Replanner LLM Benchmark", fontsize=14, fontweight="bold")

    # --- Left: grouped bar chart per metric ---
    ax1 = axes[0]
    x      = range(len(metrics))
    n      = len(models_list)
    width  = 0.8 / n
    colors = plt.cm.Set2.colors  # type: ignore

    for i, model in enumerate(models_list):
        row    = model_avgs[model_avgs["model"] == model].iloc[0]
        values = [row[m] for m in metrics]
        offset = (i - n / 2 + 0.5) * width
        bars   = ax1.bar(
            [xi + offset for xi in x], values,
            width=width * 0.9,
            label=model,
            color=colors[i % len(colors)],
            alpha=0.88,
        )
        for bar, val in zip(bars, values):
            if val > 0.05:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}",
                    ha="center", va="bottom", fontsize=7.5,
                )

    ax1.set_xticks(list(x))
    ax1.set_xticklabels(metric_labels, rotation=12, ha="right", fontsize=9)
    ax1.set_ylim(0, 1.18)
    ax1.set_ylabel("Score (0–1)")
    ax1.set_title("Scores by Metric")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # --- Right: composite score per scenario ---
    ax2 = axes[1]
    scenario_pivot = (
        df.pivot_table(
            index="scenario", columns="model",
            values="composite_score", aggfunc="mean",
        )
        .reset_index()
    )

    scenario_names = scenario_pivot["scenario"].tolist()
    xs = range(len(scenario_names))
    for i, model in enumerate(models_list):
        if model not in scenario_pivot.columns:
            continue
        vals   = scenario_pivot[model].tolist()
        offset = (i - n / 2 + 0.5) * width
        ax2.bar(
            [xi + offset for xi in xs], vals,
            width=width * 0.9,
            label=model,
            color=colors[i % len(colors)],
            alpha=0.88,
        )

    ax2.set_xticks(list(xs))
    ax2.set_xticklabels(
        [s.replace("-", "\n") for s in scenario_names],
        fontsize=7.5, ha="center",
    )
    ax2.set_ylim(0, 1.18)
    ax2.set_ylabel("Composite Score (0–1)")
    ax2.set_title("Composite Score by Scenario")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Standalone test / entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drone Replanner Benchmark")
    parser.add_argument(
        "--models", nargs="+", default=BENCHMARK_MODELS,
        help="Groq model IDs to benchmark",
    )
    parser.add_argument(
        "--scenarios", nargs="+", default=None,
        help="Scenario names to include (default: all 5)",
    )
    parser.add_argument(
        "--save-plot", default="benchmark_results.png",
        help="Output path for bar chart PNG",
    )
    parser.add_argument(
        "--save-csv", default="benchmark_results.csv",
        help="Output path for results CSV",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip LLM calls, use fallback only (tests scoring pipeline)",
    )
    args = parser.parse_args()

    api_key = os.environ.get("GROQ_API_KEY", "")

    if args.dry_run or not api_key:
        print("\n[BENCHMARK] Dry-run mode (no API key / --dry-run flag)")
        print("  Testing scoring pipeline with rule-based fallback only.\n")

        from drone_replanner.ai.replanner import _fallback_replan

        scenarios = get_benchmark_scenarios()
        if args.scenarios:
            scenarios = [s for s in scenarios if s.name in args.scenarios]

        all_scores: list[ScenarioScore] = []

        for scenario in scenarios:
            for model in args.models:
                mission  = scenario.mission_factory()
                drone_   = Drone(scenario.drone_pos, battery=scenario.drone_battery, speed=5.0)

                for wid in scenario.completed_ids:
                    try:
                        mission.mark_completed(wid)
                    except ValueError:
                        pass

                injector = AnomalyInjector(drone_, mission, BASE_POSITION)
                for atype, aparams in scenario.anomalies:
                    injector.inject_specific_anomaly(atype, aparams)

                remaining   = mission.get_remaining_waypoints()
                feasibility = check_feasibility(
                    current_position    = drone_.position,
                    current_battery     = drone_.battery,
                    remaining_waypoints = remaining,
                    base_position       = BASE_POSITION,
                    speed               = drone_.speed,
                )
                snapshot = drone_.get_state_snapshot()
                fb       = _fallback_replan(snapshot, mission, feasibility, model, 50.0)

                scores = _score_decision(
                    result      = fb,
                    mission     = mission,
                    feasibility = feasibility,
                    drone       = drone_,
                )

                all_scores.append(ScenarioScore(
                    scenario         = scenario.name,
                    model            = f"{model}[dry]",
                    valid_json       = scores["valid_json"],
                    feasible_plan    = scores["feasible_plan"],
                    priority_respect = scores["priority_respect"],
                    safe_return      = scores["safe_return"],
                    composite_score  = scores["composite_score"],
                    latency_ms       = 50.0,
                    retry_count      = 0,
                    used_fallback    = True,
                ))

        df = pd.DataFrame([vars(s) for s in all_scores])
        _print_summary(df)

        if args.save_csv:
            df.to_csv(args.save_csv, index=False)
            print(f"  CSV saved to: {args.save_csv}")

        if args.save_plot:
            _plot_results(df, args.save_plot)
            print(f"  Plot saved to: {args.save_plot}")

        print("\nDry-run complete. Set GROQ_API_KEY to run live benchmark.")

    else:
        filtered_scenarios = None
        if args.scenarios:
            all_s = get_benchmark_scenarios()
            filtered_scenarios = [s for s in all_s if s.name in args.scenarios]

        run_benchmark(
            models    = args.models,
            scenarios = filtered_scenarios,
            api_key   = api_key,
            save_plot = args.save_plot,
            save_csv  = args.save_csv,
        )
