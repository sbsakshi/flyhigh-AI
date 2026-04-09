"""
animated_visualizer.py -- Real-time animated drone mission replay.

Features:
  - Smooth drone movement (linear interpolation between path samples)
  - Waypoints change colour as their status updates
  - Live battery bar on the side panel that depletes in real time
  - No-fly zones fade in when their anomaly triggers
  - Replanning pauses the drone, shows "Replanning..." and types
    the LLM reasoning out character by character
  - LLM call (live mode) runs in a background thread so the animation
    never freezes
  - After replan: the new planned route redraws in a fresh colour and
    the drone resumes flight
  - Interactive speed slider: 1x / 2x / 5x
  - Saves the final animation as mission_replay.gif (PillowWriter)

Two entry points:
  play_live(...)  -- interactive window with slider (TkAgg)
  save_gif(...)   -- headless render to mission_replay.gif (PillowWriter)
"""

from __future__ import annotations

import math
import os
import sys
import threading
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Slider

from drone_replanner.sim.drone import Position
from drone_replanner.sim.mission import Mission, WaypointStatus, WaypointPriority
from drone_replanner.sim.anomaly import NoFlyZone


# ---------------------------------------------------------------------------
# Colour palette (kept in sync with visualizer.py)
# ---------------------------------------------------------------------------
C_ORIGINAL_ROUTE = "#AAAAAA"
C_NEW_ROUTE      = "#9C27B0"   # purple -- after replanning
C_FLOWN_PATH     = "#2196F3"
C_DRONE          = "#1565C0"
C_COMPLETED      = "#4CAF50"
C_SKIPPED        = "#9E9E9E"
C_BLOCKED        = "#F44336"
C_PENDING        = "#FF9800"
C_NFZ            = "#FF5722"
C_BASE           = "#212121"
C_BATTERY_OK     = "#4CAF50"
C_BATTERY_WARN   = "#FFC107"
C_BATTERY_CRIT   = "#F44336"


# ---------------------------------------------------------------------------
# Timeline events
# ---------------------------------------------------------------------------
@dataclass
class TimelineEvent:
    """A scheduled event on the playback timeline."""
    tick:        float
    kind:        str             # "anomaly" | "replan" | "status"
    payload:     dict = field(default_factory=dict)


@dataclass
class PlayerConfig:
    interp_steps:     int   = 12        # frames between two path samples
    base_interval_ms: int   = 40        # ms per frame at 1x speed
    typewriter_cps:   int   = 35        # chars per second during replan pause
    replan_min_pause_s: float = 1.5     # minimum pause even if reasoning short


# ===========================================================================
# CORE PLAYER
# ===========================================================================
class AnimatedMissionPlayer:
    """
    Replays a completed drone mission with smooth animation.

    Inputs are the same artefacts produced by run_simulation():
      - mission, drone (with path_history), base_position
      - replan_events (with .tick, .anomaly_desc, .result.decision.reasoning,
        .new_order, .skipped)
      - no_fly_zones (with the tick at which each was injected if available;
        otherwise zones fade in at the first replan tick)
      - original_order: list of waypoint IDs in the original plan

    Optional:
      - llm_callable: in live mode, the callable invoked in a background
        thread when a replan event is reached. Signature:
            llm_callable(event_payload: dict) -> str  (returns reasoning)
        If None (default), the player uses the reasoning already stored
        in the replan event.
    """

    def __init__(
        self,
        mission:        Mission,
        drone_path:     list[Position],
        battery_history: list[float],
        base_position:  Position,
        replan_events:  list,
        no_fly_zones:   list[NoFlyZone],
        original_order: list[int],
        llm_callable:   Optional[Callable[[dict], str]] = None,
        config:         PlayerConfig = PlayerConfig(),
        title:          str = "Drone Mission Replay",
    ):
        self.mission        = mission
        self.drone_path     = drone_path
        self.battery_history = battery_history
        self.base_position  = base_position
        self.replan_events  = sorted(replan_events, key=lambda e: e.tick)
        self.no_fly_zones   = no_fly_zones
        self.original_order = original_order
        self.llm_callable   = llm_callable
        self.cfg            = config
        self.title          = title

        # ----- Build interpolated frame timeline -----
        self.frames: list[dict] = self._build_frames()
        self.total_frames = len(self.frames)

        # ----- Replan state -----
        self.replan_ticks = {int(round(e.tick)) for e in self.replan_events}
        self.replan_lookup = {int(round(e.tick)): e for e in self.replan_events}
        self.in_replan        = False
        self.replan_reasoning = ""
        self.replan_typed     = ""
        self.replan_typed_idx = 0
        self.replan_started_frame = 0
        self.replan_ready     = False    # set by background thread
        self.replan_thread: Optional[threading.Thread] = None
        self.handled_replans: set = set()
        self.applied_new_routes: dict[int, list[int]] = {}

        # ----- Active no-fly zones (faded-in) -----
        # Map nfz index -> alpha (0..1)
        self.nfz_alpha = [0.0] * len(self.no_fly_zones)
        # If we don't know individual injection ticks, fade them all in over
        # the first replan that mentioned a no-fly zone
        self.nfz_trigger_frame = self._guess_nfz_trigger_frame()

        # ----- Speed multiplier -----
        self.speed_mult = 1.0

        # ----- Figure handles (set in _build_figure) -----
        self.fig = None
        self.ax_map = None
        self.ax_side = None
        self.drone_marker = None
        self.flown_line = None
        self.battery_bar = None
        self.battery_text = None
        self.replan_text = None
        self.reasoning_text = None
        self.wp_artists: dict[int, any] = {}
        self.nfz_circles: list = []
        self.nfz_borders: list = []
        self.new_route_line = None
        self.tick_text = None
        self.slider = None

    # ---------------- Frame timeline ----------------
    def _build_frames(self) -> list[dict]:
        """
        Pre-compute every frame's drone (x, y, battery, tick).
        Each segment between two path samples is split into interp_steps.
        """
        frames = []
        n = len(self.drone_path)
        if n == 0:
            return frames
        steps = max(self.cfg.interp_steps, 1)

        # battery_history may be shorter than drone_path if not provided;
        # fill with linear interp from 100 -> last known
        bh = list(self.battery_history) if self.battery_history else []
        if len(bh) < n:
            last = bh[-1] if bh else 0.0
            bh += [last] * (n - len(bh))

        for i in range(n - 1):
            p0, p1 = self.drone_path[i], self.drone_path[i + 1]
            b0, b1 = bh[i], bh[i + 1]
            for s in range(steps):
                t = s / steps
                frames.append({
                    "tick":    i + t,
                    "x":       p0.x + (p1.x - p0.x) * t,
                    "y":       p0.y + (p1.y - p0.y) * t,
                    "battery": b0 + (b1 - b0) * t,
                })
        # Final resting frame
        frames.append({
            "tick":    n - 1,
            "x":       self.drone_path[-1].x,
            "y":       self.drone_path[-1].y,
            "battery": bh[-1],
        })
        return frames

    def _guess_nfz_trigger_frame(self) -> int:
        """Frame index at which no-fly zones should start fading in."""
        if not self.no_fly_zones or not self.replan_events:
            return 0
        first_tick = self.replan_events[0].tick
        return self._tick_to_frame(first_tick)

    def _tick_to_frame(self, tick: float) -> int:
        steps = max(self.cfg.interp_steps, 1)
        return min(int(round(tick * steps)), self.total_frames - 1)

    # ---------------- Figure construction ----------------
    def _build_figure(self):
        self.fig = plt.figure(figsize=(14, 8))
        gs = self.fig.add_gridspec(
            2, 2,
            width_ratios=[2.3, 1],
            height_ratios=[20, 1],
            hspace=0.08, wspace=0.18,
        )
        self.ax_map  = self.fig.add_subplot(gs[0, 0])
        self.ax_side = self.fig.add_subplot(gs[0, 1])
        ax_slider    = self.fig.add_subplot(gs[1, :])

        self.fig.suptitle(self.title, fontsize=14, fontweight="bold", y=0.98)

        self._setup_map_axes()
        self._setup_side_panel()
        self._setup_slider(ax_slider)

    def _setup_map_axes(self):
        ax = self.ax_map
        ax.set_title("Mission Map", fontsize=11, pad=6)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25, linestyle="--")

        # Auto scale based on all known points
        all_wps = list(self.mission.get_all_waypoints())
        xs = [self.base_position.x] + [w.position.x for w in all_wps] + [p.x for p in self.drone_path]
        ys = [self.base_position.y] + [w.position.y for w in all_wps] + [p.y for p in self.drone_path]
        for nfz in self.no_fly_zones:
            xs += [nfz.center.x - nfz.radius, nfz.center.x + nfz.radius]
            ys += [nfz.center.y - nfz.radius, nfz.center.y + nfz.radius]
        pad_x = max((max(xs) - min(xs)) * 0.15, 5)
        pad_y = max((max(ys) - min(ys)) * 0.15, 5)
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)

        # Original planned route (dashed grey)
        all_wps_map = {w.id: w for w in all_wps}
        orig = [self.base_position] + [
            all_wps_map[i].position for i in self.original_order if i in all_wps_map
        ]
        if len(orig) > 1:
            ax.plot([p.x for p in orig], [p.y for p in orig],
                    color=C_ORIGINAL_ROUTE, linestyle="--",
                    linewidth=1.2, alpha=0.55, zorder=1, label="Original route")

        # Base
        ax.scatter(self.base_position.x, self.base_position.y,
                   s=180, c=C_BASE, marker="s", zorder=8,
                   edgecolors="white", linewidths=1.0, label="Base")
        ax.annotate("BASE", (self.base_position.x, self.base_position.y),
                    xytext=(6, -14), textcoords="offset points",
                    fontsize=8, fontweight="bold")

        # Waypoints (start as PENDING; status updates over time)
        for wp in all_wps:
            x, y = wp.position.x, wp.position.y
            size = {WaypointPriority.CRITICAL: 220,
                    WaypointPriority.HIGH:     130,
                    WaypointPriority.LOW:      90}[wp.priority]
            marker = "*" if wp.priority == WaypointPriority.CRITICAL else "o"
            artist = ax.scatter(x, y, s=size, c=C_PENDING, marker=marker,
                                zorder=5, edgecolors="white", linewidths=0.8)
            self.wp_artists[wp.id] = artist
            pri = wp.priority.value[0]
            ax.annotate(f"WP{wp.id}\n({pri})", (x, y),
                        xytext=(6, 6), textcoords="offset points",
                        fontsize=7.5,
                        fontweight="bold" if wp.priority == WaypointPriority.CRITICAL else "normal",
                        color="#333333",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  fc="#FFFFFFCC", ec="none", alpha=0.7),
                        zorder=6)

        # No-fly zones (start invisible, fade in)
        for nfz in self.no_fly_zones:
            fill = plt.Circle((nfz.center.x, nfz.center.y), nfz.radius,
                              color=C_NFZ, fill=True, alpha=0.0, zorder=3)
            border = plt.Circle((nfz.center.x, nfz.center.y), nfz.radius,
                                color=C_NFZ, fill=False, linewidth=1.5,
                                linestyle="--", alpha=0.0, zorder=3)
            ax.add_patch(fill)
            ax.add_patch(border)
            self.nfz_circles.append(fill)
            self.nfz_borders.append(border)

        # Flown path (grows over time)
        self.flown_line, = ax.plot([], [], color=C_FLOWN_PATH, linewidth=2.0,
                                   alpha=0.85, zorder=4, label="Flown path")

        # New route after replanning (drawn lazily)
        self.new_route_line, = ax.plot([], [], color=C_NEW_ROUTE, linestyle=":",
                                       linewidth=1.6, alpha=0.0, zorder=2,
                                       label="Replanned route")

        # Drone marker
        self.drone_marker, = ax.plot([], [], marker="o", markersize=12,
                                     color=C_DRONE, markeredgecolor="white",
                                     markeredgewidth=1.5, zorder=10)

        # Tick label (top-right of map)
        self.tick_text = ax.text(0.98, 0.97, "", transform=ax.transAxes,
                                 ha="right", va="top", fontsize=10,
                                 fontfamily="monospace",
                                 bbox=dict(boxstyle="round,pad=0.3",
                                           fc="#FFFFFFCC", ec="#999"))

        ax.legend(loc="upper left", fontsize=8, framealpha=0.85)

    def _setup_side_panel(self):
        ax = self.ax_side
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_title("Telemetry", fontsize=11, pad=6)

        # Battery bar background
        ax.add_patch(mpatches.Rectangle((0.08, 0.84), 0.84, 0.06,
                                        facecolor="#E0E0E0",
                                        edgecolor="#666", linewidth=1.0))
        self.battery_bar = mpatches.Rectangle((0.08, 0.84), 0.84, 0.06,
                                              facecolor=C_BATTERY_OK)
        ax.add_patch(self.battery_bar)
        self.battery_text = ax.text(0.5, 0.93, "Battery: 100%",
                                    ha="center", va="bottom", fontsize=10,
                                    fontweight="bold", color="#333")

        # Replanning banner (hidden by default)
        self.replan_text = ax.text(0.5, 0.74, "",
                                   ha="center", va="center", fontsize=13,
                                   fontweight="bold", color="#FF6F00")

        # Reasoning typewriter area
        self.reasoning_text = ax.text(
            0.04, 0.66, "",
            ha="left", va="top", fontsize=8.5,
            fontfamily="monospace", color="#212121",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                      edgecolor="#BDBDBD", alpha=0.0),
            wrap=True,
        )

    def _setup_slider(self, ax_slider):
        # Slider only works in interactive mode; harmless in headless
        try:
            self.slider = Slider(
                ax_slider, "Speed",
                valmin=1.0, valmax=5.0, valinit=1.0, valstep=[1.0, 2.0, 5.0],
            )
            self.slider.on_changed(self._on_speed_changed)
        except Exception:
            self.slider = None

    def _on_speed_changed(self, val):
        self.speed_mult = float(val)
        if hasattr(self, "_anim") and self._anim is not None:
            try:
                self._anim.event_source.interval = self.cfg.base_interval_ms / self.speed_mult
            except Exception:
                pass

    # ---------------- Replan handling ----------------
    def _start_replan(self, frame_idx: int, event):
        self.in_replan = True
        self.replan_started_frame = frame_idx
        self.replan_typed = ""
        self.replan_typed_idx = 0
        self.replan_ready = False
        self.replan_text.set_text("Replanning...")
        self.reasoning_text.get_bbox_patch().set_alpha(0.95)

        def _worker():
            # Either call live LLM or use pre-recorded reasoning
            if self.llm_callable is not None:
                try:
                    text = self.llm_callable({"event": event})
                except Exception as exc:
                    text = f"[LLM error] {exc}\nUsing recorded reasoning..."
                    text += "\n\n" + self._get_recorded_reasoning(event)
            else:
                text = self._get_recorded_reasoning(event)
            self.replan_reasoning = textwrap.fill(text, width=38)
            self.replan_ready = True

        self.replan_thread = threading.Thread(target=_worker, daemon=True)
        self.replan_thread.start()

    def _get_recorded_reasoning(self, event) -> str:
        try:
            return event.result.decision.reasoning
        except AttributeError:
            return getattr(event, "reasoning", "(no reasoning available)")

    def _finish_replan(self, event):
        self.in_replan = False
        self.replan_text.set_text("")
        # Draw the new route line
        try:
            new_order = event.result.decision.new_waypoint_order
        except AttributeError:
            new_order = getattr(event, "new_order", [])
        wp_map = {w.id: w for w in self.mission.get_all_waypoints()}
        pts = [wp_map[i].position for i in new_order if i in wp_map]
        if pts:
            xs = [p.x for p in pts]
            ys = [p.y for p in pts]
            self.new_route_line.set_data(xs, ys)
            self.new_route_line.set_alpha(0.85)

    # ---------------- Per-frame update ----------------
    def _update_waypoint_colours(self, current_tick: float):
        """
        Approximate per-tick waypoint status: walk drone_path samples up to
        current_tick, mark waypoints as COMPLETED when the drone reaches them.
        For SKIPPED/BLOCKED status, use the final mission state once that
        waypoint's recorded position is passed.
        """
        wp_map = {w.id: w for w in self.mission.get_all_waypoints()}
        idx = min(int(current_tick), len(self.drone_path) - 1)
        drone_pos = self.drone_path[idx]
        for wid, wp in wp_map.items():
            artist = self.wp_artists[wid]
            final_status = wp.status
            # If drone has been near this waypoint already, treat as final status
            reached = any(
                math.hypot(self.drone_path[k].x - wp.position.x,
                           self.drone_path[k].y - wp.position.y) < 0.6
                for k in range(0, idx + 1)
            )
            if reached and final_status == WaypointStatus.COMPLETED:
                artist.set_color(C_COMPLETED)
            elif final_status == WaypointStatus.BLOCKED and idx >= self.nfz_trigger_frame // max(self.cfg.interp_steps, 1):
                artist.set_color(C_BLOCKED)
            elif final_status == WaypointStatus.SKIPPED and reached:
                artist.set_color(C_SKIPPED)
            # else stay PENDING/orange

    def _update_nfz(self, frame_idx: int):
        if not self.no_fly_zones:
            return
        # Begin fade-in at trigger frame, complete over 20 frames
        elapsed = frame_idx - self.nfz_trigger_frame
        if elapsed < 0:
            target = 0.0
        else:
            target = min(1.0, elapsed / 20.0)
        for i in range(len(self.no_fly_zones)):
            self.nfz_alpha[i] = target
            self.nfz_circles[i].set_alpha(0.18 * target)
            self.nfz_borders[i].set_alpha(0.85 * target)

    def _update_battery(self, battery: float):
        frac = max(0.0, min(1.0, battery / 100.0))
        self.battery_bar.set_width(0.84 * frac)
        if battery > 50:
            color = C_BATTERY_OK
        elif battery > 20:
            color = C_BATTERY_WARN
        else:
            color = C_BATTERY_CRIT
        self.battery_bar.set_facecolor(color)
        self.battery_text.set_text(f"Battery: {battery:5.1f}%")

    def _frame_update(self, frame_idx: int):
        if frame_idx >= self.total_frames:
            frame_idx = self.total_frames - 1

        # ---- Replan pause logic ----
        if self.in_replan:
            # Type out reasoning character by character
            if self.replan_ready:
                steps_per_frame = max(1, int(self.cfg.typewriter_cps *
                                             self.cfg.base_interval_ms / 1000))
                self.replan_typed_idx = min(
                    len(self.replan_reasoning),
                    self.replan_typed_idx + steps_per_frame,
                )
                self.replan_typed = self.replan_reasoning[:self.replan_typed_idx]
                self.reasoning_text.set_text(self.replan_typed)

                fully_typed = self.replan_typed_idx >= len(self.replan_reasoning)
                paused_frames = frame_idx - self.replan_started_frame
                min_pause = (self.cfg.replan_min_pause_s * 1000
                             / self.cfg.base_interval_ms)
                if fully_typed and paused_frames >= min_pause:
                    # End the pause; this tick is now "handled"
                    handled_tick = int(round(self.frames[frame_idx]["tick"]))
                    if handled_tick in self.replan_lookup:
                        self._finish_replan(self.replan_lookup[handled_tick])
                    self.handled_replans.add(handled_tick)
            else:
                # waiting on background thread
                dots = "." * (1 + (frame_idx // 5) % 3)
                self.replan_text.set_text(f"Replanning{dots}")

            return self._artists_for_blit()

        # ---- Check if we hit a replan tick ----
        cur = self.frames[frame_idx]
        cur_int_tick = int(round(cur["tick"]))
        if (cur_int_tick in self.replan_ticks
                and cur_int_tick not in self.handled_replans):
            self._start_replan(frame_idx, self.replan_lookup[cur_int_tick])
            return self._artists_for_blit()

        # ---- Normal update ----
        self.drone_marker.set_data([cur["x"]], [cur["y"]])

        # Grow flown line up to current frame
        path_idx = min(int(cur["tick"]) + 1, len(self.drone_path))
        xs = [p.x for p in self.drone_path[:path_idx]] + [cur["x"]]
        ys = [p.y for p in self.drone_path[:path_idx]] + [cur["y"]]
        self.flown_line.set_data(xs, ys)

        self._update_battery(cur["battery"])
        self._update_waypoint_colours(cur["tick"])
        self._update_nfz(frame_idx)
        self.tick_text.set_text(f"tick {cur['tick']:5.1f}")

        return self._artists_for_blit()

    def _artists_for_blit(self):
        return (self.drone_marker, self.flown_line, self.battery_bar,
                self.battery_text, self.tick_text, self.replan_text,
                self.reasoning_text, self.new_route_line)

    # ---------------- Public entry points ----------------
    def play_live(self):
        """Open an interactive window with the speed slider."""
        try:
            matplotlib.use("TkAgg", force=True)
        except Exception:
            pass
        # Re-import pyplot after backend switch
        import matplotlib.pyplot as _plt  # noqa
        self._build_figure()
        self._anim = FuncAnimation(
            self.fig, self._frame_update,
            frames=self.total_frames,
            interval=self.cfg.base_interval_ms,
            blit=False, repeat=False,
        )
        plt.show()

    def save_gif(self, out_path: str = "mission_replay.gif", fps: int = 20):
        """Render the animation to a GIF using PillowWriter (headless safe)."""
        try:
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        self._build_figure()
        anim = FuncAnimation(
            self.fig, self._frame_update,
            frames=self.total_frames,
            interval=self.cfg.base_interval_ms,
            blit=False, repeat=False,
        )
        writer = PillowWriter(fps=fps)
        print(f"[ANIM] Rendering {self.total_frames} frames to {out_path}...")
        anim.save(out_path, writer=writer, dpi=100)
        print(f"[ANIM] Saved {out_path}")
        plt.close(self.fig)
        return out_path


# ===========================================================================
# Convenience builder
# ===========================================================================
def play_simulation(
    mission, drone, replan_events, no_fly_zones, original_order,
    base_position: Position = Position(0.0, 0.0),
    save_gif_path: Optional[str] = None,
    live: bool = False,
    llm_callable: Optional[Callable[[dict], str]] = None,
):
    """
    Convenience wrapper. Pass the same artefacts produced by run_simulation().

    If save_gif_path is given, render to GIF (headless).
    If live=True, open an interactive window with speed slider.
    """
    battery_history = getattr(drone, "battery_history", None)
    if not battery_history:
        # fall back to constant final battery
        battery_history = [drone.battery] * len(drone.path_history)

    player = AnimatedMissionPlayer(
        mission         = mission,
        drone_path      = drone.path_history,
        battery_history = battery_history,
        base_position   = base_position,
        replan_events   = replan_events,
        no_fly_zones    = no_fly_zones,
        original_order  = original_order,
        llm_callable    = llm_callable,
        title           = f"Drone Mission Replay: {mission.name}",
    )
    if save_gif_path:
        player.save_gif(save_gif_path)
    if live:
        player.play_live()
    return player


# ===========================================================================
# Standalone smoke test
# ===========================================================================
if __name__ == "__main__":
    from dataclasses import dataclass
    from drone_replanner.sim.drone import Drone
    from drone_replanner.sim.mission import make_mission_medium
    from drone_replanner.sim.anomaly import AnomalyInjector, AnomalyType
    from drone_replanner.sim.feasibility import check_feasibility
    from drone_replanner.ai.replanner import _fallback_replan, DEFAULT_MODEL

    print("=" * 60)
    print("ANIMATED VISUALIZER -- smoke test (renders mission_replay.gif)")
    print("=" * 60)

    BASE = Position(0.0, 0.0)
    drone = Drone(BASE, battery=100.0, speed=5.0)
    mission = make_mission_medium()
    original_order = [w.id for w in mission.get_remaining_waypoints()]
    injector = AnomalyInjector(drone, mission, BASE)

    # Track battery for each path step
    battery_history = [drone.battery]

    # Fly until WP2 reached
    for _ in range(40):
        wps = mission.get_remaining_waypoints()
        if not wps:
            break
        reached = drone.move_to_next_waypoint([w.position for w in wps])
        battery_history.append(drone.battery)
        if reached:
            mission.mark_completed(wps[0].id)
            if wps[0].id == 2:
                break

    replan_tick = drone.tick

    # Inject anomalies
    injector.inject_specific_anomaly(AnomalyType.BATTERY_DROP, {"drop_amount": 30.0})
    injector.inject_specific_anomaly(
        AnomalyType.NO_FLY_ZONE,
        {"center_x": 45.0, "center_y": 10.0, "radius": 10.0},
    )
    battery_history[-1] = drone.battery

    # Run fallback replanner
    feasibility = check_feasibility(
        current_position=drone.position, current_battery=drone.battery,
        remaining_waypoints=mission.get_remaining_waypoints(),
        base_position=BASE, speed=drone.speed,
    )
    snapshot = drone.get_state_snapshot()
    fb = _fallback_replan(snapshot, mission, feasibility, DEFAULT_MODEL, 50.0)

    @dataclass
    class _Ev:
        tick: int
        trigger: str
        anomaly_desc: str
        feasibility_before: str
        result: object
        new_order: list
        skipped: list

    ev = _Ev(
        tick=replan_tick, trigger="anomaly",
        anomaly_desc=injector.get_active_anomalies()[0].description,
        feasibility_before=feasibility.summary_str(),
        result=fb,
        new_order=fb.decision.new_waypoint_order,
        skipped=[s.waypoint_id for s in fb.decision.skipped_waypoints],
    )

    mission.update_order(fb.decision.new_waypoint_order)
    for sw in fb.decision.skipped_waypoints:
        wp = mission.get_waypoint(sw.waypoint_id)
        if wp.status == WaypointStatus.PENDING:
            mission.mark_skipped(sw.waypoint_id, sw.reason)

    for _ in range(200):
        wps = mission.get_remaining_waypoints()
        if not wps:
            drone.set_complete()
            break
        reached = drone.move_to_next_waypoint([w.position for w in wps])
        battery_history.append(drone.battery)
        if reached:
            mission.mark_completed(wps[0].id)

    out_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "mission_replay.gif"))

    player = AnimatedMissionPlayer(
        mission=mission,
        drone_path=drone.path_history,
        battery_history=battery_history,
        base_position=BASE,
        replan_events=[ev],
        no_fly_zones=injector.get_no_fly_zones(),
        original_order=original_order,
        title=f"Drone Mission Replay: {mission.name}",
    )
    player.save_gif(out_path, fps=20)
    print(f"\nDone. {os.path.getsize(out_path)/1024:.1f} KB written.")
