"""
visualizer.py -- matplotlib visualisation for the drone mission replanner.

Renders a two-panel figure:
  Left panel  -- 2D mission map with:
    - Original planned route       (dashed grey line)
    - Actual flown path            (solid blue line)
    - Completed waypoints          (green circles)
    - Skipped waypoints            (grey X marks)
    - Blocked waypoints            (red X marks)
    - No-fly zones                 (red dashed circles)
    - Replanning decision points   (yellow stars)
    - Base position                (black square)
    - Waypoint ID + priority labels

  Right panel -- LLM reasoning trace for each replanning event.

Can be called after run_simulation() or standalone with test data.
"""

import os
import sys
import textwrap
from typing import Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (safe for all environments)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from drone_replanner.sim.drone import Position
from drone_replanner.sim.mission import Mission, WaypointStatus, WaypointPriority
from drone_replanner.sim.anomaly import NoFlyZone

# Import only for type hints (avoid circular import at runtime)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from main import ReplanEvent


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_ORIGINAL_ROUTE = "#AAAAAA"   # dashed grey
C_FLOWN_PATH     = "#2196F3"   # solid blue
C_COMPLETED      = "#4CAF50"   # green circle
C_SKIPPED        = "#9E9E9E"   # grey X
C_BLOCKED        = "#F44336"   # red X
C_NFZ            = "#FF5722"   # red dashed circle
C_REPLAN_STAR    = "#FFC107"   # yellow star
C_BASE           = "#212121"   # black square
C_LABEL_BOX      = "#FFFFFFCC" # semi-transparent white


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_mission(
    mission:        Mission,
    drone_path:     list[Position],
    base_position:  Position,
    replan_events:  list,           # list[ReplanEvent] -- imported lazily
    no_fly_zones:   list[NoFlyZone],
    original_order: list[int],      # WP IDs in original planned order
    save_path:      Optional[str]   = None,
    show:           bool            = True,
    title:          str             = "Drone Mission Replay",
) -> plt.Figure:
    """
    Render the full mission visualisation and reasoning trace.

    Args:
        mission:        Completed Mission object (for waypoint positions/status).
        drone_path:     List of Position objects from drone.path_history.
        base_position:  Home base position.
        replan_events:  List of ReplanEvent dataclasses from run_simulation().
        no_fly_zones:   List of NoFlyZone objects from AnomalyInjector.
        original_order: Waypoint IDs in the original planned order.
        save_path:      If given, save figure to this path (PNG/PDF/SVG).
        show:           If True, call plt.show() after rendering.
        title:          Figure title.

    Returns:
        The matplotlib Figure object.
    """
    # Determine layout: map on left, reasoning trace on right (if events exist)
    has_reasoning = bool(replan_events)
    fig_width = 16 if has_reasoning else 9
    fig, axes = plt.subplots(
        1, 2 if has_reasoning else 1,
        figsize=(fig_width, 8),
        gridspec_kw={"width_ratios": [1.2, 1]} if has_reasoning else None,
    )
    ax_map = axes[0] if has_reasoning else axes
    ax_text = axes[1] if has_reasoning else None

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    _render_map(
        ax          = ax_map,
        mission     = mission,
        drone_path  = drone_path,
        base_pos    = base_position,
        replan_events = replan_events,
        no_fly_zones = no_fly_zones,
        original_order = original_order,
    )

    if ax_text is not None:
        _render_reasoning(ax_text, replan_events)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[VIZ] Saved to: {save_path}")

    if show:
        plt.show()

    return fig


# ---------------------------------------------------------------------------
# Map panel
# ---------------------------------------------------------------------------

def _render_map(
    ax:             plt.Axes,
    mission:        Mission,
    drone_path:     list[Position],
    base_pos:       Position,
    replan_events:  list,
    no_fly_zones:   list[NoFlyZone],
    original_order: list[int],
) -> None:
    """Draw the 2D mission map onto ax."""

    ax.set_title("Mission Map", fontsize=11, pad=8)
    ax.set_xlabel("X (metres)")
    ax.set_ylabel("Y (metres)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25, linestyle="--")

    all_wps = {wp.id: wp for wp in mission.get_all_waypoints()}

    # ------------------------------------------------------------------
    # 1. Original planned route (dashed grey)
    # ------------------------------------------------------------------
    orig_positions = [base_pos] + [
        all_wps[wid].position for wid in original_order if wid in all_wps
    ]
    if len(orig_positions) > 1:
        xs = [p.x for p in orig_positions]
        ys = [p.y for p in orig_positions]
        ax.plot(xs, ys, color=C_ORIGINAL_ROUTE, linestyle="--",
                linewidth=1.2, alpha=0.6, zorder=1, label="Original route")

    # ------------------------------------------------------------------
    # 2. Actual flown path (solid blue)
    # ------------------------------------------------------------------
    if len(drone_path) > 1:
        pxs = [p.x for p in drone_path]
        pys = [p.y for p in drone_path]
        ax.plot(pxs, pys, color=C_FLOWN_PATH, linewidth=2.0,
                zorder=2, label="Actual path", alpha=0.85)

    # ------------------------------------------------------------------
    # 3. No-fly zones (red dashed circles)
    # ------------------------------------------------------------------
    for nfz in no_fly_zones:
        circle = plt.Circle(
            (nfz.center.x, nfz.center.y), nfz.radius,
            color=C_NFZ, fill=True, alpha=0.10, zorder=3,
        )
        border = plt.Circle(
            (nfz.center.x, nfz.center.y), nfz.radius,
            color=C_NFZ, fill=False, linewidth=1.5,
            linestyle="--", alpha=0.8, zorder=3,
        )
        ax.add_patch(circle)
        ax.add_patch(border)
        ax.annotate(
            f"NFZ\nr={nfz.radius:.0f}m",
            (nfz.center.x, nfz.center.y),
            fontsize=7, color=C_NFZ, ha="center", va="center",
            zorder=4,
        )

    # ------------------------------------------------------------------
    # 4. Waypoints
    # ------------------------------------------------------------------
    priority_markers = {
        WaypointPriority.CRITICAL: ("*", 220),
        WaypointPriority.HIGH:     ("o", 120),
        WaypointPriority.LOW:      ("o",  80),
    }

    for wid, wp in all_wps.items():
        x, y = wp.position.x, wp.position.y
        marker, size = priority_markers[wp.priority]

        if wp.status == WaypointStatus.COMPLETED:
            ax.scatter(x, y, s=size, c=C_COMPLETED, marker=marker,
                       zorder=5, edgecolors="white", linewidths=0.8)

        elif wp.status == WaypointStatus.SKIPPED:
            ax.scatter(x, y, s=100, c=C_SKIPPED, marker="x",
                       zorder=5, linewidths=2.0)

        elif wp.status == WaypointStatus.BLOCKED:
            ax.scatter(x, y, s=120, c=C_BLOCKED, marker="x",
                       zorder=5, linewidths=2.5)

        else:  # PENDING (shouldn't happen post-sim, but handle it)
            ax.scatter(x, y, s=size, c="#FF9800", marker=marker,
                       zorder=5, edgecolors="white", linewidths=0.8)

        # Label: WP id + priority initial
        pri_char = wp.priority.value[0]   # C / H / L
        label = f"WP{wid}\n({pri_char})"
        ax.annotate(
            label, (x, y),
            xytext=(6, 6), textcoords="offset points",
            fontsize=7.5, fontweight="bold" if wp.priority == WaypointPriority.CRITICAL else "normal",
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.15", fc=C_LABEL_BOX, ec="none", alpha=0.7),
            zorder=6,
        )

    # ------------------------------------------------------------------
    # 5. Replanning decision points (yellow stars)
    # ------------------------------------------------------------------
    for ev in replan_events:
        # Find drone position at that tick from path_history
        tick = ev.tick
        path_idx = min(tick, len(drone_path) - 1)
        px, py = drone_path[path_idx].x, drone_path[path_idx].y
        ax.scatter(px, py, s=280, c=C_REPLAN_STAR, marker="*",
                   zorder=7, edgecolors="#795548", linewidths=0.8,
                   label="Replanning point" if ev == replan_events[0] else "")
        ax.annotate(
            f"Replan\ntick {tick}",
            (px, py),
            xytext=(8, -14), textcoords="offset points",
            fontsize=7, color="#795548",
            zorder=8,
        )

    # ------------------------------------------------------------------
    # 6. Base position (black square)
    # ------------------------------------------------------------------
    ax.scatter(
        base_pos.x, base_pos.y, s=180, c=C_BASE, marker="s",
        zorder=7, label="Base", edgecolors="white", linewidths=1.0,
    )
    ax.annotate(
        "BASE", (base_pos.x, base_pos.y),
        xytext=(6, -14), textcoords="offset points",
        fontsize=8, fontweight="bold", color=C_BASE,
        zorder=8,
    )

    # ------------------------------------------------------------------
    # 7. Legend
    # ------------------------------------------------------------------
    legend_elements = [
        Line2D([0], [0], color=C_ORIGINAL_ROUTE, linestyle="--", lw=1.5, label="Original route"),
        Line2D([0], [0], color=C_FLOWN_PATH,     linestyle="-",  lw=2.0, label="Actual path"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_COMPLETED,
               markersize=9, label="Completed"),
        Line2D([0], [0], marker="x", color=C_SKIPPED,  lw=0, markersize=9,
               markeredgewidth=2, label="Skipped"),
        Line2D([0], [0], marker="x", color=C_BLOCKED,  lw=0, markersize=9,
               markeredgewidth=2, label="Blocked"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=C_REPLAN_STAR,
               markersize=12, label="Replan point"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_BASE,
               markersize=9, label="Base"),
    ]
    if no_fly_zones:
        legend_elements.append(
            mpatches.Patch(facecolor=C_NFZ, alpha=0.3, edgecolor=C_NFZ,
                           label="No-fly zone")
        )
    ax.legend(handles=legend_elements, loc="upper left",
              fontsize=8, framealpha=0.85)

    # Auto-scale with padding
    _autoscale(ax, all_wps, drone_path, base_pos, no_fly_zones)


def _autoscale(
    ax:          plt.Axes,
    all_wps:     dict,
    drone_path:  list[Position],
    base_pos:    Position,
    no_fly_zones: list[NoFlyZone],
) -> None:
    """Set axis limits with comfortable padding."""
    xs = [base_pos.x] + [wp.position.x for wp in all_wps.values()]
    ys = [base_pos.y] + [wp.position.y for wp in all_wps.values()]
    if drone_path:
        xs += [p.x for p in drone_path]
        ys += [p.y for p in drone_path]
    for nfz in no_fly_zones:
        xs += [nfz.center.x - nfz.radius, nfz.center.x + nfz.radius]
        ys += [nfz.center.y - nfz.radius, nfz.center.y + nfz.radius]

    pad_x = max((max(xs) - min(xs)) * 0.15, 5)
    pad_y = max((max(ys) - min(ys)) * 0.15, 5)
    ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
    ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)


# ---------------------------------------------------------------------------
# Reasoning trace panel
# ---------------------------------------------------------------------------

def _render_reasoning(ax: plt.Axes, replan_events: list) -> None:
    """Render the LLM reasoning trace as formatted text in ax."""
    ax.set_title("LLM Reasoning Trace", fontsize=11, pad=8)
    ax.axis("off")

    if not replan_events:
        ax.text(0.5, 0.5, "No replanning events.", ha="center", va="center",
                fontsize=10, color="#666666", transform=ax.transAxes)
        return

    lines: list[str] = []
    for i, ev in enumerate(replan_events, 1):
        d = ev.result.decision
        lines.append(f"{'='*48}")
        lines.append(f"Event {i} | Tick {ev.tick} | {ev.trigger.upper()}")
        lines.append(f"{'='*48}")
        lines.append(f"Anomaly:  {ev.anomaly_desc[:60]}")
        lines.append(f"Before:   {ev.feasibility_before}")
        lines.append(f"Model:    {ev.result.model_used}")
        lines.append(f"Latency:  {ev.result.latency_ms:.0f} ms  "
                     f"| Retries: {ev.result.retry_count}  "
                     f"| Fallback: {ev.result.used_fallback}")
        lines.append(f"Confidence: {d.confidence.upper()}")
        lines.append(f"New order:  {d.new_waypoint_order}")
        lines.append(f"Skipped:    {[s.waypoint_id for s in d.skipped_waypoints]}")
        lines.append(f"Abort:      {d.abort_mission}")
        if d.abort_mission and d.abort_reason:
            lines.append(f"Abort reason: {d.abort_reason[:60]}")
        lines.append("")
        lines.append("Reasoning:")
        # Wrap long reasoning text
        wrapped = textwrap.fill(d.reasoning, width=52)
        for wline in wrapped.split("\n"):
            lines.append(f"  {wline}")
        lines.append("")

    full_text = "\n".join(lines)
    ax.text(
        0.02, 0.98, full_text,
        transform=ax.transAxes,
        fontsize=7.5,
        verticalalignment="top",
        horizontalalignment="left",
        fontfamily="monospace",
        color="#212121",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#F5F5F5",
            edgecolor="#BDBDBD",
            alpha=0.9,
        ),
        wrap=False,
    )


# ---------------------------------------------------------------------------
# Convenience: build from run_simulation() outputs
# ---------------------------------------------------------------------------

def visualize_simulation(
    mission:        Mission,
    drone,                          # Drone instance
    replan_events:  list,
    no_fly_zones:   list[NoFlyZone],
    original_order: list[int],
    save_path:      Optional[str]   = None,
    show:           bool            = True,
) -> plt.Figure:
    """
    Convenience wrapper — takes raw simulation outputs.

    Args:
        mission:        Completed Mission object.
        drone:          Drone object after simulation (has path_history).
        replan_events:  List of ReplanEvent from run_simulation().
        no_fly_zones:   List of NoFlyZone from AnomalyInjector.
        original_order: Original planned WP order (before any replanning).
        save_path:      Optional file path to save the figure.
        show:           Whether to call plt.show().

    Returns:
        matplotlib Figure.
    """
    return render_mission(
        mission        = mission,
        drone_path     = drone.path_history,
        base_position  = Position(0.0, 0.0),
        replan_events  = replan_events,
        no_fly_zones   = no_fly_zones,
        original_order = original_order,
        save_path      = save_path,
        show           = show,
        title          = f"Drone Mission: {mission.name}",
    )


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from drone_replanner.sim.drone import Drone
    from drone_replanner.sim.mission import (
        make_mission_medium, WaypointPriority, WaypointType,
        Waypoint, WaypointStatus,
    )
    from drone_replanner.sim.anomaly import AnomalyInjector, AnomalyType, NoFlyZone
    from drone_replanner.sim.feasibility import check_feasibility
    from drone_replanner.ai.replanner import _fallback_replan, DEFAULT_MODEL
    from drone_replanner.ai.schemas import ReplanResult

    # We need to recreate ReplanEvent without importing main.py to avoid
    # circular concerns -- define it locally for test purposes.
    from dataclasses import dataclass

    @dataclass
    class _TestReplanEvent:
        tick:               int
        trigger:            str
        anomaly_desc:       str
        feasibility_before: str
        result:             ReplanResult
        new_order:          list[int]
        skipped:            list[int]

    print("=" * 60)
    print("COMPONENT 9 -- Visualizer Test")
    print("=" * 60)

    BASE   = Position(0.0, 0.0)
    drone  = Drone(Position(0.0, 0.0), battery=100.0, speed=5.0)
    mission = make_mission_medium()
    original_order = [wp.id for wp in mission.get_remaining_waypoints()]

    injector = AnomalyInjector(drone, mission, BASE)

    # Simulate partial flight by manually advancing drone
    from drone_replanner.sim.mission import WaypointStatus
    waypoints = mission.get_remaining_waypoints()

    # Fly to WP1 and WP2 manually
    for _ in range(20):
        wps = mission.get_remaining_waypoints()
        if not wps:
            break
        targets = [wp.position for wp in wps]
        reached = drone.move_to_next_waypoint(targets)
        if reached:
            mission.mark_completed(wps[0].id)
            if wps[0].id == 2:
                break

    # Inject anomalies
    injector.inject_specific_anomaly(AnomalyType.BATTERY_DROP, {"drop_amount": 30.0})
    inj2 = injector.inject_specific_anomaly(
        AnomalyType.NO_FLY_ZONE,
        {"center_x": 45.0, "center_y": 10.0, "radius": 10.0},
    )

    # Build a fake replanning event using the fallback planner
    remaining  = mission.get_remaining_waypoints()
    feasibility = check_feasibility(
        current_position    = drone.position,
        current_battery     = drone.battery,
        remaining_waypoints = remaining,
        base_position       = BASE,
        speed               = drone.speed,
    )
    snapshot  = drone.get_state_snapshot()
    fb_result = _fallback_replan(snapshot, mission, feasibility, DEFAULT_MODEL, 50.0)

    replan_event = _TestReplanEvent(
        tick               = drone.tick,
        trigger            = "anomaly",
        anomaly_desc       = injector.get_active_anomalies()[0].description,
        feasibility_before = feasibility.summary_str(),
        result             = fb_result,
        new_order          = fb_result.decision.new_waypoint_order,
        skipped            = [s.waypoint_id for s in fb_result.decision.skipped_waypoints],
    )

    # Apply the fallback plan to mission so waypoints show correct status
    mission.update_order(fb_result.decision.new_waypoint_order)
    for sw in fb_result.decision.skipped_waypoints:
        wp = mission.get_waypoint(sw.waypoint_id)
        if wp.status == WaypointStatus.PENDING:
            mission.mark_skipped(sw.waypoint_id, sw.reason)

    # Fly remaining waypoints
    for _ in range(150):
        wps = mission.get_remaining_waypoints()
        if not wps:
            drone.set_complete()
            break
        targets = [wp.position for wp in wps]
        reached = drone.move_to_next_waypoint(targets)
        if reached:
            mission.mark_completed(wps[0].id)

    # Render
    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "mission_plot.png")
    out_path = os.path.abspath(out_path)

    fig = visualize_simulation(
        mission        = mission,
        drone          = drone,
        replan_events  = [replan_event],
        no_fly_zones   = injector.get_no_fly_zones(),
        original_order = original_order,
        save_path      = out_path,
        show           = False,   # headless -- no display needed
    )

    print(f"\n  Figure size: {fig.get_size_inches()} inches")
    print(f"  Axes count:  {len(fig.axes)}")
    print(f"  Saved to:    {out_path}")

    # Verify file was written
    assert os.path.exists(out_path), "Plot file not created!"
    size_kb = os.path.getsize(out_path) / 1024
    print(f"  File size:   {size_kb:.1f} KB")
    assert size_kb > 20, f"Plot file suspiciously small: {size_kb:.1f} KB"

    plt.close(fig)

    # Summary
    summary = mission.get_mission_summary()
    print(f"\n  Mission summary: completed={summary.completed} "
          f"skipped={summary.skipped} blocked={summary.blocked}")
    print("\nAll checks passed.")
