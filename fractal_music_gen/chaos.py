"""Chaos theory generators: Lorenz attractor, logistic map, Henon map."""

from __future__ import annotations

import numpy as np
from fractal_music_gen.music_theory import quantize_to_scale, get_scale_notes, value_to_velocity


def lorenz_attractor(num_steps: int = 5000, dt: float = 0.01,
                     sigma: float = 10.0, rho: float = 28.0,
                     beta: float = 8.0 / 3.0) -> np.ndarray:
    """Compute Lorenz attractor trajectory.

    Args:
        num_steps: Number of integration steps.
        dt: Time step.
        sigma, rho, beta: Lorenz system parameters.

    Returns:
        Array of shape (num_steps, 3) with x, y, z coordinates.
    """
    trajectory = np.zeros((num_steps, 3))
    trajectory[0] = [1.0, 1.0, 1.0]

    for i in range(1, num_steps):
        x, y, z = trajectory[i - 1]
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        trajectory[i] = [x + dx * dt, y + dy * dt, z + dz * dt]

    return trajectory


def logistic_map(r: float = 3.99, x0: float = 0.1,
                 num_steps: int = 500) -> np.ndarray:
    """Iterate the logistic map x_{n+1} = r * x_n * (1 - x_n).

    Args:
        r: Growth rate parameter (interesting range: 3.5-4.0).
        x0: Initial value (0 < x0 < 1).
        num_steps: Number of iterations.

    Returns:
        1D array of values.
    """
    values = np.zeros(num_steps)
    values[0] = x0
    for i in range(1, num_steps):
        values[i] = r * values[i - 1] * (1 - values[i - 1])
    return values


def henon_map(a: float = 1.4, b: float = 0.3, x0: float = 0.0,
              y0: float = 0.0, num_steps: int = 500) -> np.ndarray:
    """Iterate the Henon map.

    x_{n+1} = 1 - a*x_n^2 + y_n
    y_{n+1} = b * x_n

    Args:
        a, b: Henon map parameters.
        x0, y0: Initial values.
        num_steps: Number of iterations.

    Returns:
        Array of shape (num_steps, 2) with x, y values.
    """
    trajectory = np.zeros((num_steps, 2))
    trajectory[0] = [x0, y0]

    for i in range(1, num_steps):
        x, y = trajectory[i - 1]
        new_x = 1 - a * x * x + y
        new_y = b * x
        trajectory[i] = [new_x, new_y]

        # Check for divergence
        if abs(new_x) > 1e6 or abs(new_y) > 1e6:
            trajectory = trajectory[:i + 1]
            break

    return trajectory


def lorenz_melody(num_steps: int = 2000, root: int = 60, scale: str = "minor",
                  sample_rate: int = 10, max_notes: int = 200) -> list[dict]:
    """Generate melody from Lorenz attractor trajectory.

    The x coordinate maps to pitch, y to velocity, and z to note duration.

    Args:
        num_steps: Lorenz integration steps.
        root: Root MIDI note.
        scale: Scale name.
        sample_rate: Sample every N-th point from trajectory.
        max_notes: Maximum notes.

    Returns:
        List of note events.
    """
    scale_notes = get_scale_notes(root, scale, octaves=3)
    traj = lorenz_attractor(num_steps)

    # Normalize each axis to 0-1
    for axis in range(3):
        col = traj[:, axis]
        mn, mx = col.min(), col.max()
        if mx > mn:
            traj[:, axis] = (col - mn) / (mx - mn)
        else:
            traj[:, axis] = 0.5

    notes = []
    current_time = 0.0

    for i in range(0, len(traj), sample_rate):
        if len(notes) >= max_notes:
            break

        x, y, z = traj[i]
        pitch = quantize_to_scale(x, scale_notes)
        velocity = value_to_velocity(y, 40, 110)
        duration = 0.25 + z * 0.5

        notes.append({
            "pitch": pitch,
            "start": current_time,
            "duration": duration,
            "velocity": velocity,
        })
        current_time += 0.25

    return notes


def logistic_melody(r: float = 3.99, root: int = 60, scale: str = "harmonic_minor",
                    max_notes: int = 200) -> list[dict]:
    """Generate melody from logistic map iterations.

    Args:
        r: Logistic map parameter.
        root: Root MIDI note.
        scale: Scale name.
        max_notes: Maximum notes.

    Returns:
        List of note events.
    """
    scale_notes = get_scale_notes(root, scale, octaves=3)
    values = logistic_map(r=r, num_steps=max_notes + 50)
    notes = []
    current_time = 0.0

    for i in range(min(len(values), max_notes)):
        v = values[i]
        pitch = quantize_to_scale(v, scale_notes)

        # Use difference between consecutive values for velocity
        if i > 0:
            delta = abs(values[i] - values[i - 1])
        else:
            delta = 0.5
        velocity = value_to_velocity(delta * 2, 40, 110)

        # Vary duration based on value
        if v > 0.8 or v < 0.2:
            duration = 0.5  # longer at extremes
        else:
            duration = 0.25

        notes.append({
            "pitch": pitch,
            "start": current_time,
            "duration": duration,
            "velocity": velocity,
        })
        current_time += 0.25

    return notes


def henon_melody(root: int = 60, scale: str = "dorian",
                 max_notes: int = 200) -> list[dict]:
    """Generate melody from Henon map trajectory.

    Args:
        root: Root MIDI note.
        scale: Scale name.
        max_notes: Maximum notes.

    Returns:
        List of note events.
    """
    scale_notes = get_scale_notes(root, scale, octaves=3)
    traj = henon_map(num_steps=max_notes + 50)

    # Normalize
    for axis in range(2):
        col = traj[:, axis]
        mn, mx = col.min(), col.max()
        if mx > mn:
            traj[:, axis] = (col - mn) / (mx - mn)
        else:
            traj[:, axis] = 0.5

    notes = []
    current_time = 0.0

    for i in range(min(len(traj), max_notes)):
        x, y = traj[i]
        pitch = quantize_to_scale(x, scale_notes)
        velocity = value_to_velocity(y, 45, 105)
        duration = 0.25

        notes.append({
            "pitch": pitch,
            "start": current_time,
            "duration": duration,
            "velocity": velocity,
        })
        current_time += 0.25

    return notes
