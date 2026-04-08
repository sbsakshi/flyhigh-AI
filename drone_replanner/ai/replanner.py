"""
replanner.py -- AI-powered mission replanner using the Groq API.

Flow:
  1. Build prompt from live state (via prompt.py).
  2. Call Groq LLM and parse JSON response into ReplanDecision.
  3. Validate against mission state (blocked IDs, unaccounted waypoints).
  4. On failure: retry up to MAX_RETRIES times, feeding the error back to the LLM.
  5. If all retries fail: rule-based fallback (skip LOWs, nearest-neighbour CRITICAL+HIGH).
  6. Return ReplanResult with decision + metadata.
"""

import json
import logging
import math
import os
import sys
import time

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from groq import Groq
from pydantic import ValidationError

from drone_replanner.sim.drone import DroneStateSnapshot, Position
from drone_replanner.sim.mission import Mission, WaypointPriority, Waypoint
from drone_replanner.sim.anomaly import Anomaly
from drone_replanner.sim.feasibility import FeasibilityResult
from drone_replanner.ai.prompt import build_prompt
from drone_replanner.ai.schemas import ReplanDecision, ReplanResult, SkippedWaypoint

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [REPLANNER] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = "llama-3.3-70b-versatile"
MAX_RETRIES:   int = 2
TEMPERATURE:   float = 0.2   # Low temperature for consistent structured output


# ---------------------------------------------------------------------------
# Main replanner entry point
# ---------------------------------------------------------------------------

def run_replanner(
    snapshot:    DroneStateSnapshot,
    mission:     Mission,
    anomalies:   list[Anomaly],
    feasibility: FeasibilityResult,
    api_key:     str | None = None,
    model:       str = DEFAULT_MODEL,
) -> ReplanResult:
    """
    Run the AI replanner and return a validated ReplanResult.

    Tries up to MAX_RETRIES times. On each failure the validation error
    is appended to the conversation so the LLM can self-correct.
    If all attempts fail, falls back to a deterministic rule-based plan.

    Args:
        snapshot:    Current drone state snapshot.
        mission:     Live Mission object (read-only).
        anomalies:   Active anomaly list from AnomalyInjector.
        feasibility: Pre-computed FeasibilityResult.
        api_key:     Groq API key. Falls back to GROQ_API_KEY env var.
        model:       Groq model ID to use.

    Returns:
        ReplanResult with decision, metadata, and fallback flag.
    """
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if not key:
        logger.warning("No GROQ_API_KEY found -- using fallback immediately.")
        return _fallback_replan(snapshot, mission, feasibility, model, latency_ms=0.0)

    client = Groq(api_key=key)

    system_prompt = build_prompt(snapshot, mission, anomalies, feasibility)
    remaining_ids = [wp.id for wp in mission.get_remaining_waypoints()]
    blocked_ids   = mission.get_blocked_ids()

    messages: list[dict] = [
        {"role": "user", "content": system_prompt},
    ]

    start_time  = time.perf_counter()
    last_error  = ""
    raw_response = ""

    for attempt in range(MAX_RETRIES + 1):
        if attempt > 0:
            # Feed the previous error back so the LLM can self-correct
            retry_msg = (
                f"Your previous response was invalid. Error:\n{last_error}\n\n"
                "Please fix the issue and respond with a valid JSON object only. "
                f"Remember: remaining IDs to account for = {remaining_ids}, "
                f"blocked IDs (exclude from order) = {blocked_ids}."
            )
            messages.append({"role": "user", "content": retry_msg})
            logger.info("Retry %d/%d -- feeding error back to LLM.", attempt, MAX_RETRIES)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=1024,
            )
            raw_response = response.choices[0].message.content.strip()
            logger.info("LLM raw response (attempt %d): %s", attempt + 1, raw_response[:200])

            # Append assistant reply to conversation history for next retry
            messages.append({"role": "assistant", "content": raw_response})

            # --- Parse JSON ---
            cleaned = _extract_json(raw_response)
            decision = ReplanDecision.model_validate_json(cleaned)

            # --- Validate against mission state ---
            errors = decision.validate_against_mission(remaining_ids, blocked_ids)
            if errors:
                last_error = "Mission validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
                logger.warning("Validation errors on attempt %d: %s", attempt + 1, errors)
                continue

            # --- Success ---
            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "Replanning succeeded on attempt %d | latency=%.0fms | "
                "confidence=%s | abort=%s",
                attempt + 1, latency_ms, decision.confidence, decision.abort_mission,
            )
            return ReplanResult(
                decision      = decision,
                model_used    = model,
                latency_ms    = round(latency_ms, 1),
                retry_count   = attempt,
                used_fallback = False,
                raw_response  = raw_response,
            )

        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Parse/validation error on attempt %d: %s", attempt + 1, exc)
            continue

        except Exception as exc:
            logger.error("Unexpected LLM error on attempt %d: %s", attempt + 1, exc)
            last_error = str(exc)
            continue

    # --- All retries exhausted: use rule-based fallback ---
    latency_ms = (time.perf_counter() - start_time) * 1000
    logger.warning(
        "All %d attempts failed. Using rule-based fallback. Last error: %s",
        MAX_RETRIES + 1, last_error,
    )
    result = _fallback_replan(snapshot, mission, feasibility, model, latency_ms)
    return ReplanResult(
        decision      = result.decision,
        model_used    = result.model_used,
        latency_ms    = round(latency_ms, 1),
        retry_count   = MAX_RETRIES,
        used_fallback = True,
        raw_response  = raw_response,
    )


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def _fallback_replan(
    snapshot:    DroneStateSnapshot,
    mission:     Mission,
    feasibility: FeasibilityResult,
    model:       str,
    latency_ms:  float,
) -> ReplanResult:
    """
    Deterministic fallback when LLM fails.

    Strategy:
      1. Take only waypoints in feasibility.max_reachable_waypoints.
      2. Among those, skip all LOW priority waypoints.
      3. Order remaining by nearest-neighbour from current position.
      4. Mark all others as skipped.

    Args:
        snapshot:    Current drone state (for position).
        mission:     Live Mission (for waypoint details).
        feasibility: Pre-computed feasibility result.
        model:       Model name to record in metadata.
        latency_ms:  Elapsed time so far.

    Returns:
        ReplanResult flagged as used_fallback=True.
    """
    logger.info("Running rule-based fallback planner.")

    remaining   = mission.get_remaining_waypoints()
    reachable   = set(feasibility.max_reachable_waypoints)
    wp_map      = {wp.id: wp for wp in remaining}

    # Keep CRITICAL and HIGH that are reachable; skip LOW and out-of-range
    keep_ids:    list[int] = []
    skip_ids:    list[int] = []

    for wp in remaining:
        if wp.id in reachable and wp.priority != WaypointPriority.LOW:
            keep_ids.append(wp.id)
        else:
            skip_ids.append(wp.id)

    # Nearest-neighbour ordering on keep_ids
    ordered = _nearest_neighbour(
        start    = snapshot.position,
        wp_ids   = keep_ids,
        wp_map   = wp_map,
    )

    skipped = [
        SkippedWaypoint(
            waypoint_id           = wid,
            reason                = (
                "Rule-based fallback: waypoint is LOW priority or out of battery range."
                if wid in reachable
                else "Rule-based fallback: out of battery range."
            ),
            priority_acknowledged = (
                wp_map[wid].priority in (WaypointPriority.CRITICAL, WaypointPriority.HIGH)
            ),
        )
        for wid in skip_ids
    ]

    battery_est = max(0.0, feasibility.battery_remaining_after)

    decision = ReplanDecision(
        reasoning=(
            f"Rule-based fallback activated after LLM failure. "
            f"Keeping {len(ordered)} reachable non-LOW waypoints in nearest-neighbour order. "
            f"Skipping {len(skip_ids)} waypoints (LOW priority or out of range)."
        ),
        new_waypoint_order       = ordered,
        skipped_waypoints        = skipped,
        confidence               = "low",
        estimated_battery_remaining = round(battery_est, 1),
        abort_mission            = False,
        abort_reason             = None,
    )

    logger.info("Fallback plan: order=%s  skipped=%s", ordered, skip_ids)

    return ReplanResult(
        decision      = decision,
        model_used    = f"{model}[fallback]",
        latency_ms    = round(latency_ms, 1),
        retry_count   = MAX_RETRIES,
        used_fallback = True,
        raw_response  = "",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _nearest_neighbour(
    start:  Position,
    wp_ids: list[int],
    wp_map: dict[int, Waypoint],
) -> list[int]:
    """
    Order wp_ids by nearest-neighbour greedy traversal from start.

    Args:
        start:  Starting position.
        wp_ids: IDs of waypoints to order.
        wp_map: Mapping from ID to Waypoint.

    Returns:
        Ordered list of waypoint IDs.
    """
    unvisited = list(wp_ids)
    ordered:  list[int] = []
    current = start

    while unvisited:
        nearest = min(
            unvisited,
            key=lambda wid: current.distance_to(wp_map[wid].position),
        )
        ordered.append(nearest)
        current = wp_map[nearest].position
        unvisited.remove(nearest)

    return ordered


def _extract_json(text: str) -> str:
    """
    Extract a JSON object from LLM output that may contain prose or markdown.

    Tries:
      1. Direct parse (clean response).
      2. Extract between first '{' and last '}'.
      3. Strip markdown code fences (```json ... ```).

    Args:
        text: Raw LLM response string.

    Returns:
        Cleaned JSON string.

    Raises:
        ValueError: If no JSON object can be extracted.
    """
    # Try direct parse first
    stripped = text.strip()
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    # Strip markdown fences
    if "```" in stripped:
        import re
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped)
        if match:
            candidate = match.group(1).strip()
            try:
                json.loads(candidate)
                return candidate
            except json.JSONDecodeError:
                pass

    # Extract between first { and last }
    start = stripped.find("{")
    end   = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start:end + 1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract valid JSON from LLM response: {stripped[:200]}")


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from drone_replanner.sim.drone import Drone
    from drone_replanner.sim.mission import make_mission_medium
    from drone_replanner.sim.anomaly import AnomalyInjector, AnomalyType
    from drone_replanner.sim.feasibility import check_feasibility

    print("=" * 60)
    print("COMPONENT 7 -- AI Replanner Test")
    print("=" * 60)

    BASE   = Position(0.0, 0.0)
    drone  = Drone(Position(15.0, 5.0), battery=45.0, speed=5.0)
    mission = make_mission_medium()
    mission.mark_completed(1)
    mission.mark_completed(2)

    injector = AnomalyInjector(drone, mission, BASE)
    injector.inject_specific_anomaly(AnomalyType.BATTERY_DROP, {"drop_amount": 20.0})
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

    print(f"\nDrone battery: {drone.battery:.1f}%")
    print(f"Remaining WPs: {[wp.id for wp in mission.get_remaining_waypoints()]}")
    print(f"Blocked WPs:   {mission.get_blocked_ids()}")
    print(f"Feasibility:   {feasibility.summary_str()}")
    print()

    # --- Test _extract_json helper ---
    print("[Test] _extract_json helper")
    cases = [
        ('{"reasoning": "ok", "abort_mission": false}', True),
        ('```json\n{"reasoning": "ok"}\n```', True),
        ('Here is the plan:\n{"reasoning": "ok"}', True),
        ('no json here', False),
    ]
    for text, should_pass in cases:
        try:
            result = _extract_json(text)
            assert should_pass, f"Expected failure for: {text[:40]}"
            print(f"  PASS (extracted {len(result)} chars)")
        except ValueError:
            assert not should_pass, f"Expected success for: {text[:40]}"
            print(f"  PASS (correctly rejected non-JSON)")

    # --- Test fallback planner directly ---
    print("\n[Test] Rule-based fallback")
    fb = _fallback_replan(snapshot, mission, feasibility, DEFAULT_MODEL, 0.0)
    print(f"  Order:        {fb.decision.new_waypoint_order}")
    print(f"  Skipped:      {[s.waypoint_id for s in fb.decision.skipped_waypoints]}")
    print(f"  Confidence:   {fb.decision.confidence}")
    print(f"  Used fallback:{fb.used_fallback}")
    assert fb.used_fallback
    assert fb.decision.confidence == "low"
    # Validate fallback decision against mission
    errors = fb.decision.validate_against_mission(
        remaining_ids = [wp.id for wp in mission.get_remaining_waypoints()],
        blocked_ids   = mission.get_blocked_ids(),
    )
    print(f"  Validation errors: {errors}")
    assert errors == [], f"Fallback produced invalid plan: {errors}"
    print("  PASS")

    # --- Live LLM test (requires GROQ_API_KEY) ---
    print("\n[Test] Live LLM call (skipped if no GROQ_API_KEY)")
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("  SKIPPED -- set GROQ_API_KEY env var to run live test.")
    else:
        print("  Calling Groq API...")
        result = run_replanner(
            snapshot    = snapshot,
            mission     = mission,
            anomalies   = anomalies,
            feasibility = feasibility,
            model       = DEFAULT_MODEL,
        )
        print(f"  Model:        {result.model_used}")
        print(f"  Latency:      {result.latency_ms:.0f} ms")
        print(f"  Retries:      {result.retry_count}")
        print(f"  Fallback:     {result.used_fallback}")
        print(f"  Confidence:   {result.decision.confidence}")
        print(f"  Abort:        {result.decision.abort_mission}")
        print(f"  New order:    {result.decision.new_waypoint_order}")
        print(f"  Skipped:      {[s.waypoint_id for s in result.decision.skipped_waypoints]}")
        print(f"  Reasoning:    {result.decision.reasoning[:120]}...")
        # Validate
        errors = result.decision.validate_against_mission(
            remaining_ids = [wp.id for wp in mission.get_remaining_waypoints()],
            blocked_ids   = mission.get_blocked_ids(),
        )
        print(f"  Validation errors: {errors}")
        assert errors == [], f"LLM produced invalid plan: {errors}"
        print("  PASS")

    print("\nAll tests completed.")
