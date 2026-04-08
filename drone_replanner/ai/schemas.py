"""
schemas.py -- Pydantic v2 output schemas for the AI replanner.

Defines the strict structured output the LLM must produce, plus
three validator rules enforced before any plan is applied:

  1. new_waypoint_order must not contain blocked waypoint IDs.
  2. new_waypoint_order + skipped_waypoints must cover ALL remaining waypoints.
  3. If abort_mission is True, new_waypoint_order must be empty.
"""

import os
import sys
from typing import Literal, Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# SkippedWaypoint
# ---------------------------------------------------------------------------

class SkippedWaypoint(BaseModel):
    """
    Record of a single waypoint the LLM has decided to skip.

    Attributes:
        waypoint_id:           ID of the waypoint being skipped.
        reason:                Plain-English explanation.
        priority_acknowledged: True if the LLM explicitly acknowledged that
                               this waypoint has HIGH or CRITICAL priority
                               (forces the LLM to justify skipping important tasks).
    """

    waypoint_id:           int  = Field(..., description="ID of the skipped waypoint.")
    reason:                str  = Field(..., description="Plain-English reason for skipping.")
    priority_acknowledged: bool = Field(
        ...,
        description=(
            "Set True if this waypoint is HIGH or CRITICAL priority and you "
            "are explicitly acknowledging the tradeoff of skipping it."
        ),
    )

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# ReplanDecision
# ---------------------------------------------------------------------------

class ReplanDecision(BaseModel):
    """
    Complete replanning decision returned by the LLM.

    Attributes:
        reasoning:                  LLM explanation of its decision in plain English.
        new_waypoint_order:         Waypoint IDs in new execution order (PENDING only).
        skipped_waypoints:          Waypoints explicitly skipped with reasons.
        confidence:                 LLM's self-assessed confidence level.
        estimated_battery_remaining: LLM's estimate of battery % after mission.
        abort_mission:              True if drone should RTB immediately.
        abort_reason:               Required explanation when abort_mission is True.
    """

    reasoning: str = Field(
        ...,
        description="Step-by-step explanation of the replanning decision.",
    )
    new_waypoint_order: list[int] = Field(
        default_factory=list,
        description="Ordered list of PENDING waypoint IDs to visit.",
    )
    skipped_waypoints: list[SkippedWaypoint] = Field(
        default_factory=list,
        description="Waypoints intentionally skipped with justification.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Self-assessed confidence in this plan.",
    )
    estimated_battery_remaining: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Estimated battery % remaining after completing the plan.",
    )
    abort_mission: bool = Field(
        default=False,
        description="Set True to abort the mission and return to base immediately.",
    )
    abort_reason: Optional[str] = Field(
        default=None,
        description="Required explanation if abort_mission is True.",
    )

    # ------------------------------------------------------------------
    # Cross-field validators
    # ------------------------------------------------------------------

    @model_validator(mode="after")
    def abort_requires_empty_order(self) -> "ReplanDecision":
        """If aborting, new_waypoint_order must be empty."""
        if self.abort_mission and self.new_waypoint_order:
            raise ValueError(
                "abort_mission is True but new_waypoint_order is not empty. "
                "Set new_waypoint_order to [] when aborting."
            )
        return self

    @model_validator(mode="after")
    def abort_requires_reason(self) -> "ReplanDecision":
        """If aborting, abort_reason must be provided."""
        if self.abort_mission and not self.abort_reason:
            raise ValueError(
                "abort_mission is True but abort_reason is missing. "
                "Provide a reason for aborting the mission."
            )
        return self

    # ------------------------------------------------------------------
    # Contextual validators (called externally with mission state)
    # ------------------------------------------------------------------

    def validate_against_mission(
        self,
        remaining_ids: list[int],
        blocked_ids:   list[int],
    ) -> list[str]:
        """
        Validate this decision against live mission state.

        Checks:
          1. new_waypoint_order contains no blocked IDs.
          2. Every remaining (non-blocked) waypoint appears in either
             new_waypoint_order or skipped_waypoints.
          3. No ID appears in both new_waypoint_order and skipped_waypoints.

        Args:
            remaining_ids: IDs of all PENDING (non-blocked) waypoints.
            blocked_ids:   IDs of all BLOCKED waypoints.

        Returns:
            List of validation error strings (empty = valid).
        """
        errors: list[str] = []

        order_set   = set(self.new_waypoint_order)
        skipped_set = {sw.waypoint_id for sw in self.skipped_waypoints}

        # 1. No blocked IDs in order
        blocked_in_order = order_set & set(blocked_ids)
        if blocked_in_order:
            errors.append(
                f"new_waypoint_order contains blocked waypoint IDs: {sorted(blocked_in_order)}. "
                "Remove them — blocked waypoints cannot be visited."
            )

        # 2. All remaining waypoints accounted for (unless aborting)
        if not self.abort_mission:
            accounted = order_set | skipped_set
            unaccounted = set(remaining_ids) - accounted
            if unaccounted:
                errors.append(
                    f"These remaining waypoints are unaccounted for: {sorted(unaccounted)}. "
                    "Each must appear in new_waypoint_order or skipped_waypoints."
                )

        # 3. No ID in both lists
        overlap = order_set & skipped_set
        if overlap:
            errors.append(
                f"Waypoint IDs appear in both new_waypoint_order and skipped_waypoints: "
                f"{sorted(overlap)}. Each ID must appear in exactly one list."
            )

        return errors

    def is_valid_for_mission(
        self,
        remaining_ids: list[int],
        blocked_ids:   list[int],
    ) -> bool:
        """Convenience bool wrapper around validate_against_mission."""
        return len(self.validate_against_mission(remaining_ids, blocked_ids)) == 0

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Replanner metadata (wraps ReplanDecision with execution info)
# ---------------------------------------------------------------------------

class ReplanResult(BaseModel):
    """
    Full output of one replanning cycle, including LLM metadata.

    Attributes:
        decision:      The validated ReplanDecision from the LLM.
        model_used:    Name of the model that produced the decision.
        latency_ms:    Round-trip LLM call latency in milliseconds.
        retry_count:   Number of retries needed (0 = first attempt succeeded).
        used_fallback: True if rule-based fallback was used instead of LLM.
        raw_response:  Raw JSON string from the LLM (for debugging).
    """

    decision:      ReplanDecision
    model_used:    str
    latency_ms:    float
    retry_count:   int   = 0
    used_fallback: bool  = False
    raw_response:  str   = ""

    model_config = {"frozen": True}


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("COMPONENT 5 -- Pydantic Schemas Test")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Test 1: Valid decision — reorder, skip one low-priority WP
    # ------------------------------------------------------------------
    print("\n[Test 1] Valid ReplanDecision")
    d1 = ReplanDecision(
        reasoning="Battery is at 45%. WP5 is low priority and far away. "
                  "Reordering to visit CRITICAL WPs first.",
        new_waypoint_order=[3, 1, 4, 6],
        skipped_waypoints=[
            SkippedWaypoint(waypoint_id=5, reason="Low priority, too far", priority_acknowledged=False),
        ],
        confidence="high",
        estimated_battery_remaining=12.5,
        abort_mission=False,
        abort_reason=None,
    )
    errors = d1.validate_against_mission(
        remaining_ids=[1, 3, 4, 5, 6],
        blocked_ids=[2],
    )
    print(f"  Validation errors: {errors}")
    assert errors == [], f"Expected no errors, got: {errors}"
    assert d1.is_valid_for_mission([1, 3, 4, 5, 6], [2])
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 2: Abort mission
    # ------------------------------------------------------------------
    print("\n[Test 2] Abort decision")
    d2 = ReplanDecision(
        reasoning="Battery critically low, cannot reach any remaining waypoint safely.",
        new_waypoint_order=[],
        skipped_waypoints=[],
        confidence="high",
        estimated_battery_remaining=8.0,
        abort_mission=True,
        abort_reason="Battery at 12%, insufficient to reach nearest waypoint and RTB.",
    )
    errors2 = d2.validate_against_mission(remaining_ids=[3, 4, 5], blocked_ids=[])
    print(f"  Validation errors: {errors2}")
    assert errors2 == []
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 3: Invalid — blocked WP in order
    # ------------------------------------------------------------------
    print("\n[Test 3] Invalid: blocked WP in new_waypoint_order")
    d3 = ReplanDecision(
        reasoning="Attempting to include a blocked waypoint.",
        new_waypoint_order=[3, 2, 4],   # WP2 is blocked
        skipped_waypoints=[
            SkippedWaypoint(waypoint_id=5, reason="Low priority", priority_acknowledged=False),
        ],
        confidence="low",
        estimated_battery_remaining=10.0,
        abort_mission=False,
        abort_reason=None,
    )
    errors3 = d3.validate_against_mission(
        remaining_ids=[3, 4, 5],
        blocked_ids=[2],
    )
    print(f"  Errors: {errors3}")
    assert any("blocked" in e for e in errors3)
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 4: Invalid — unaccounted waypoints
    # ------------------------------------------------------------------
    print("\n[Test 4] Invalid: waypoint missing from order and skipped list")
    d4 = ReplanDecision(
        reasoning="Forgot to include WP4.",
        new_waypoint_order=[3, 5],
        skipped_waypoints=[],           # WP4 is neither ordered nor skipped
        confidence="low",
        estimated_battery_remaining=15.0,
        abort_mission=False,
        abort_reason=None,
    )
    errors4 = d4.validate_against_mission(
        remaining_ids=[3, 4, 5],
        blocked_ids=[],
    )
    print(f"  Errors: {errors4}")
    assert any("unaccounted" in e for e in errors4)
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 5: Invalid — abort with non-empty order
    # ------------------------------------------------------------------
    print("\n[Test 5] Invalid: abort_mission=True but order not empty")
    try:
        ReplanDecision(
            reasoning="Aborting but forgot to clear order.",
            new_waypoint_order=[3, 4],
            skipped_waypoints=[],
            confidence="low",
            estimated_battery_remaining=5.0,
            abort_mission=True,
            abort_reason="Emergency RTB",
        )
        assert False, "Should have raised ValidationError"
    except Exception as e:
        print(f"  Caught expected error: {type(e).__name__}")
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 6: Invalid — abort with no reason
    # ------------------------------------------------------------------
    print("\n[Test 6] Invalid: abort_mission=True but no abort_reason")
    try:
        ReplanDecision(
            reasoning="Aborting.",
            new_waypoint_order=[],
            skipped_waypoints=[],
            confidence="low",
            estimated_battery_remaining=5.0,
            abort_mission=True,
            abort_reason=None,
        )
        assert False, "Should have raised ValidationError"
    except Exception as e:
        print(f"  Caught expected error: {type(e).__name__}")
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 7: JSON round-trip
    # ------------------------------------------------------------------
    print("\n[Test 7] JSON serialisation round-trip")
    json_str = d1.model_dump_json(indent=2)
    restored = ReplanDecision.model_validate_json(json_str)
    assert restored.new_waypoint_order == d1.new_waypoint_order
    assert restored.confidence == d1.confidence
    print(f"  Serialised keys: {list(json.loads(json_str).keys())}")
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 8: ReplanResult wrapper
    # ------------------------------------------------------------------
    print("\n[Test 8] ReplanResult wrapper")
    rr = ReplanResult(
        decision     = d1,
        model_used   = "llama-3.3-70b-versatile",
        latency_ms   = 842.3,
        retry_count  = 0,
        used_fallback= False,
        raw_response = json_str,
    )
    print(f"  Model:    {rr.model_used}")
    print(f"  Latency:  {rr.latency_ms} ms")
    print(f"  Fallback: {rr.used_fallback}")
    assert rr.decision.confidence == "high"
    print("  PASS")

    print("\nAll tests passed.")
