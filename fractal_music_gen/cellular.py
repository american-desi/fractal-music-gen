"""1D Cellular automata (Rule 30, 90, 110) mapped to musical notes."""

from __future__ import annotations

import numpy as np
from fractal_music_gen.music_theory import quantize_to_scale, get_scale_notes, value_to_velocity


def elementary_ca(rule_number: int, width: int = 64,
                  steps: int = 64, initial: np.ndarray | None = None) -> np.ndarray:
    """Run a 1D elementary cellular automaton.

    Args:
        rule_number: Wolfram rule number (0-255).
        width: Width of the grid.
        steps: Number of time steps.
        initial: Optional initial state. If None, starts with single center cell.

    Returns:
        2D numpy array of shape (steps, width) with values 0 or 1.
    """
    # Decode rule into lookup table
    rule_bin = format(rule_number, "08b")
    lookup = {}
    for i in range(8):
        pattern = format(7 - i, "03b")
        lookup[tuple(int(b) for b in pattern)] = int(rule_bin[i])

    # Initialize grid
    grid = np.zeros((steps, width), dtype=np.int8)
    if initial is not None:
        grid[0, :len(initial)] = initial[:width]
    else:
        grid[0, width // 2] = 1

    # Evolve
    for t in range(1, steps):
        for x in range(width):
            left = grid[t - 1, (x - 1) % width]
            center = grid[t - 1, x]
            right = grid[t - 1, (x + 1) % width]
            grid[t, x] = lookup[(left, center, right)]

    return grid


def ca_to_melody(grid: np.ndarray, root: int = 60, scale: str = "blues",
                 max_notes: int = 200) -> list[dict]:
    """Convert cellular automaton grid to a melody.

    Each row becomes one beat. Active cells in a row determine which notes
    are played. The density and position of active cells map to pitch and velocity.

    Args:
        grid: 2D array from elementary_ca.
        root: Root MIDI note.
        scale: Scale name.
        max_notes: Maximum notes to produce.

    Returns:
        List of note events.
    """
    scale_notes = get_scale_notes(root, scale, octaves=3)
    steps, width = grid.shape
    notes = []
    current_time = 0.0

    for t in range(steps):
        if len(notes) >= max_notes:
            break

        row = grid[t]
        active = np.where(row == 1)[0]

        if len(active) == 0:
            current_time += 0.5
            continue

        # Map center of mass of active cells to pitch
        center = np.mean(active) / width  # 0 to 1
        density = len(active) / width  # 0 to 1

        pitch = quantize_to_scale(center, scale_notes)
        velocity = value_to_velocity(density, 40, 110)
        duration = 0.25 + 0.25 * density

        notes.append({
            "pitch": pitch,
            "start": current_time,
            "duration": duration,
            "velocity": velocity,
        })

        # Add harmony notes from spread of active cells if dense enough
        if density > 0.3 and len(active) >= 3:
            low_note = quantize_to_scale(active[0] / width, scale_notes)
            high_note = quantize_to_scale(active[-1] / width, scale_notes)
            if low_note != pitch:
                notes.append({
                    "pitch": low_note,
                    "start": current_time,
                    "duration": duration * 0.8,
                    "velocity": max(1, velocity - 20),
                })
            if high_note != pitch and high_note != low_note:
                notes.append({
                    "pitch": high_note,
                    "start": current_time,
                    "duration": duration * 0.8,
                    "velocity": max(1, velocity - 20),
                })

        current_time += 0.5

    return notes


def ca_to_rhythm(grid: np.ndarray) -> list[tuple[float, bool]]:
    """Convert CA grid to a rhythm pattern.

    Args:
        grid: 2D array from elementary_ca.

    Returns:
        List of (duration_beats, is_rest) tuples.
    """
    steps, width = grid.shape
    rhythm = []
    center_col = width // 2

    for t in range(steps):
        is_active = grid[t, center_col] == 1
        density = np.sum(grid[t]) / width
        duration = 0.25 if density > 0.5 else 0.5
        rhythm.append((duration, not is_active))

    return rhythm


def generate_ca_piece(rule: int = 30, root: int = 60, scale: str = "blues",
                      width: int = 64, steps: int = 100,
                      max_notes: int = 200) -> list[dict]:
    """Generate a complete piece from a cellular automaton rule.

    Args:
        rule: Wolfram rule number.
        root: Root MIDI note.
        scale: Scale name.
        width: CA grid width.
        steps: CA time steps.
        max_notes: Maximum notes.

    Returns:
        List of note events.
    """
    grid = elementary_ca(rule, width=width, steps=steps)
    return ca_to_melody(grid, root=root, scale=scale, max_notes=max_notes)
