"""
get_human_preferences_v2.py

Matplotlib-based preference UI — opens as an external popup window.
Has clickable buttons for replay and voting.

Called from reward_model_pebble_human_v2.py via:

    import get_human_preferences_v2 as get_human_preferences
    get_human_preferences.get_single_human_label(segment1, segment2, trajectory_id, size_segment)

Returns: 0 (prefer A) | 1 (prefer B) | 0.5 (equal) | -1 (can't decide)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.widgets import Button
import threading
import time 
plt.rcParams['toolbar'] = 'None'

# ─── Colours ──────────────────────────────────────────────────────────────────
BG       = "#0c0e14"
BG_CARD  = "#10121a"
BORDER   = "#1e2130"
ACCENT_A = "#f5a623"
ACCENT_B = "#38bdf8"
ACCENT_V = "#a78bfa"
GREEN    = "#4ade80"
GREY     = "#94a3b8"
WHITE    = "#dde1ec"


def _styled_ax(ax, title, accent):
    ax.set_facecolor(BG_CARD)
    ax.set_title(title, color=accent, fontweight="bold", fontsize=13, pad=10,
                 fontfamily="monospace")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(accent)
        spine.set_linewidth(2)


def _make_button(ax, label, text_color, border_color):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.5)
    btn = Button(ax, label, color=BG, hovercolor="#1a1d2a")
    btn.label.set_color(text_color)
    btn.label.set_fontsize(12)
    btn.label.set_fontweight("bold")
    return btn


def create_preference_window():
    """Call once at the start of your session."""
    fig = plt.figure(figsize=(14, 9), facecolor=BG)
    plt.show(block=False)
    return fig


def get_single_human_label(
    segment1: np.ndarray,
    segment2: np.ndarray,
    trajectory_id: int,
    size_segment: int,
    fig: plt.figure = None,
    frame_dim=(200, 350)
):
    """
    Open a popup window showing both trajectories side by side with
    replay and vote buttons. Blocks until the user clicks a vote button.

    Returns a dictionary containing the vote value, times, and replay counts.
    """
    seg1 = np.clip(segment1.reshape(size_segment, frame_dim[0], frame_dim[1], 3), 0, 255).astype(np.uint8)
    seg2 = np.clip(segment2.reshape(size_segment, frame_dim[0], frame_dim[1], 3), 0, 255).astype(np.uint8)
    n_frames = size_segment

    # if fig is None:
    #     fig = plt.figure(figsize=(14, 9), facecolor=BG)
    # else:
    #     fig.canvas.mouse_grabber = None
    #     if hasattr(fig, 'buttons'):
    #         for btn in fig.buttons:
    #             btn.disconnect_events()
    #     fig.clf()
        
    #     # --- MODIFIED: Only show/raise if hidden ---
    #     # This prevents the window from "flashing" between every comparison
    #     if hasattr(fig.canvas.manager, 'window'):
    #          fig.canvas.manager.window.show()
    #          # Only raise to top on the very first comparison of a session
    #          if trajectory_id == 0:
    #              fig.canvas.manager.window.raise_()

    fig = plt.figure(figsize=(14, 9), facecolor=BG)


    # ── Tracking Metrics ──────────────────────────────────────────────────────
    start_time = time.time()
    
    result = {
        "value": None,
        "rewatch_a": 0,
        "rewatch_b": 0,
        "rewatch_both": 0,
        "time_taken": 0.0
    }
    ani_ref = {"ani": None}

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.set_facecolor(BG)
    fig.suptitle(
        f"Preference Comparison  #{trajectory_id+1}",
        color=WHITE, fontsize=15, fontweight="bold",
        fontfamily="monospace", y=0.97
    )

    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        height_ratios=[6, 0.6, 0.6],
        hspace=0.35,
        wspace=0.5,    
        left=0.05, right=0.95,
        top=0.92, bottom=0.05,
    )

    # Video panels
    ax_a = fig.add_subplot(gs[0, :2])
    ax_b = fig.add_subplot(gs[0, 2:])
    _styled_ax(ax_a, "▶  Trajectory A", ACCENT_A)
    _styled_ax(ax_b, "▶  Trajectory B", ACCENT_B)

    img_a = ax_a.imshow(seg1[0])
    img_b = ax_b.imshow(seg2[0])

    # Frame counter
    frame_txt = fig.text(
        0.5, 0.60, f"Frame  1 / {n_frames}",
        ha="center", color=GREY, fontsize=10, fontfamily="monospace"
    )

    # ── Replay buttons (row 1) ────────────────────────────────────────────────
    ax_ra    = fig.add_subplot(gs[1, 0])  
    ax_rb    = fig.add_subplot(gs[1, 1])  
    ax_rboth = fig.add_subplot(gs[1, 2])  
    ax_hint  = fig.add_subplot(gs[1, 3])
    
    ax_hint.axis("off")
    ax_hint.text(0.02, 0.5, "← Replay Controls",
                 color=GREY, fontsize=10, fontweight="bold", 
                 va="center", transform=ax_hint.transAxes)

    btn_ra   = _make_button(ax_ra,    "⟳  Replay A",    ACCENT_A, ACCENT_A)
    btn_rb   = _make_button(ax_rb,    "⟳  Replay B",    ACCENT_B, ACCENT_B)
    btn_both = _make_button(ax_rboth, "⟳  Replay Both", ACCENT_V, ACCENT_V)

    # ── Vote buttons (row 2) ──────────────────────────────────────────────────
    ax_va  = fig.add_subplot(gs[2, 0])
    ax_vb  = fig.add_subplot(gs[2, 1])
    ax_veq = fig.add_subplot(gs[2, 2])
    ax_vnd = fig.add_subplot(gs[2, 3])

    btn_va  = _make_button(ax_va,  "Prefer A",     ACCENT_A, ACCENT_A)
    btn_vb  = _make_button(ax_vb,  "Prefer B",     ACCENT_B, ACCENT_B)
    btn_veq = _make_button(ax_veq, "Equal",        GREEN,    GREEN)
    btn_vnd = _make_button(ax_vnd, "Can't Decide", WHITE,    GREY)

    # Status text
    status_txt = fig.text(
        0.5, 0.02,
        "Watch both clips, then cast your vote.",
        ha="center", color=GREY, fontsize=10, fontfamily="monospace"
    )

    # ── Animation helpers ─────────────────────────────────────────────────────

    def _run_animation(frames_a, frames_b, label=""):
        if ani_ref.get("ani") is not None:
            if ani_ref["ani"].event_source is not None:
                ani_ref["ani"].event_source.stop()

        def update(i):
            if frames_a is not None:
                img_a.set_data(frames_a[i])
            if frames_b is not None:
                img_b.set_data(frames_b[i])
            
            prefix = f"{label} " if label else ""
            frame_txt.set_text(f"{prefix}Frame  {i + 1} / {n_frames}")
            
            if i < n_frames - 1:
                frame_txt.set_color(GREY)
            else:
                frame_txt.set_color(ACCENT_V)
                
            fig.canvas.draw_idle()
            return img_a, img_b, frame_txt

        ani_ref["ani"] = animation.FuncAnimation(
            fig, update,
            frames=n_frames,
            interval=50,
            repeat=False,
            blit=False,
        )
        fig.canvas.draw_idle()

    # Initial playback
    _run_animation(seg1, seg2)

    # ── Replay callbacks ──────────────────────────────────────────────────────

    def track_replay(btn_type, frames_a, frames_b, label=""):
        result[btn_type] += 1
        _run_animation(frames_a, frames_b, label)

    btn_both.on_clicked(lambda _: track_replay("rewatch_both", seg1, seg2))
    btn_ra.on_clicked(  lambda _: track_replay("rewatch_a", seg1, None, label="A —"))
    btn_rb.on_clicked(  lambda _: track_replay("rewatch_b", None, seg2, label="B —"))

    # ── Vote callbacks ────────────────────────────────────────────────────────

    def _vote(value, name, color):
        result["value"] = value
        result["time_taken"] = time.time() - start_time
        
        status_txt.set_text(f"✓  Recorded: {name}")
        status_txt.set_color(color)
        status_txt.set_fontweight("bold")
        for b in [btn_va, btn_vb, btn_veq, btn_vnd]:
            b.label.set_alpha(0.3)
        fig.canvas.draw_idle()

    btn_va.on_clicked( lambda _: _vote(0,   "Prefer A",     ACCENT_A))
    btn_vb.on_clicked( lambda _: _vote(1,   "Prefer B",     ACCENT_B))
    btn_veq.on_clicked(lambda _: _vote(-1, "Equal",        GREEN))
    btn_vnd.on_clicked(lambda _: _vote(-2, "Can't Decide", GREY))

    fig.buttons = [btn_both, btn_ra, btn_rb, btn_va, btn_vb, btn_veq, btn_vnd]

    # # ── Block until vote is cast ──────────────────────────────────────────────
    # plt.show(block=False)

    # while plt.fignum_exists(fig.number) and result.get("value") is None:
    #     plt.pause(0.1)

    plt.show(block=False)
    while plt.fignum_exists(fig.number) and result.get("value") is None:
        plt.pause(0.1)
    
    # close immediately after vote
    plt.close(fig)

    if result["value"] is None:
        result["value"] = -1
        result["time_taken"] = time.time() - start_time

    return result


# ─── Legacy stubs ─────────────────────────────────────────────────────────────

def render_trajectory(frames, trajectory_id, trajectory_comp):
    pass

def render_trajectory_again(segment1, segment2, trajectory_id):
    pass