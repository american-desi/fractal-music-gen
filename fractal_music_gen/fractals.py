"""Fractal generators: Mandelbrot orbits, Julia set paths, Sierpinski walks, Koch curve."""

from __future__ import annotations

import numpy as np
from fractal_music_gen.music_theory import quantize_to_scale, get_scale_notes, value_to_velocity


def mandelbrot_orbit(c: complex, max_iter: int = 100) -> list[complex]:
    """Compute the orbit of a point under Mandelbrot iteration z = z^2 + c.

    Args:
        c: Complex parameter.
        max_iter: Maximum iterations before stopping.

    Returns:
        List of complex orbit points (before escape or max_iter).
    """
    z = 0 + 0j
    orbit = [z]
    for _ in range(max_iter):
        z = z * z + c
        orbit.append(z)
        if abs(z) > 2.0:
            break
    return orbit


def julia_orbit(z0: complex, c: complex, max_iter: int = 100) -> list[complex]:
    """Compute orbit of z0 under Julia set iteration z = z^2 + c.

    Args:
        z0: Starting point.
        c: Julia set parameter.
        max_iter: Maximum iterations.

    Returns:
        List of orbit points.
    """
    z = z0
    orbit = [z]
    for _ in range(max_iter):
        z = z * z + c
        orbit.append(z)
        if abs(z) > 2.0:
            break
    return orbit


def sierpinski_walk(iterations: int = 6) -> list[tuple[float, float]]:
    """Generate points along a Sierpinski triangle using chaos game.

    Args:
        iterations: Number of points to generate.

    Returns:
        List of (x, y) coordinate tuples.
    """
    # Triangle vertices
    vertices = [(0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3) / 2)]
    point = (np.random.random(), np.random.random())
    points = []

    for _ in range(iterations):
        vertex = vertices[np.random.randint(3)]
        point = ((point[0] + vertex[0]) / 2, (point[1] + vertex[1]) / 2)
        points.append(point)

    return points


def koch_curve_points(iterations: int = 4) -> list[tuple[float, float]]:
    """Generate points along a Koch curve.

    Args:
        iterations: Subdivision iterations.

    Returns:
        List of (x, y) coordinate tuples along the curve.
    """
    def subdivide(p1, p2):
        """Subdivide a line segment into Koch curve segments."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        a = (p1[0] + dx / 3, p1[1] + dy / 3)
        b = (p1[0] + 2 * dx / 3, p1[1] + 2 * dy / 3)

        # Peak point (equilateral triangle)
        cos60 = np.cos(np.pi / 3)
        sin60 = np.sin(np.pi / 3)
        mx = a[0] + (b[0] - a[0]) * cos60 - (b[1] - a[1]) * sin60
        my = a[1] + (b[0] - a[0]) * sin60 + (b[1] - a[1]) * cos60
        peak = (mx, my)

        return [p1, a, peak, b, p2]

    points = [(0.0, 0.0), (1.0, 0.0)]

    for _ in range(iterations):
        new_points = []
        for i in range(len(points) - 1):
            seg = subdivide(points[i], points[i + 1])
            if new_points:
                new_points.extend(seg[1:])  # Skip duplicate start
            else:
                new_points.extend(seg)
        points = new_points

    return points


def mandelbrot_melody(num_points: int = 20, root: int = 60,
                      scale: str = "minor", max_notes: int = 200) -> list[dict]:
    """Generate melody by sampling Mandelbrot set boundary orbits.

    Scans along the boundary of the Mandelbrot set, using orbit properties
    to determine pitch, duration, and velocity.

    Args:
        num_points: Number of c-values to sample along the boundary.
        root: Root MIDI note.
        scale: Scale name.
        max_notes: Maximum notes.

    Returns:
        List of note events.
    """
    scale_notes = get_scale_notes(root, scale, octaves=3)
    notes = []
    current_time = 0.0

    # Sample along the main cardioid boundary
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    for angle in angles:
        if len(notes) >= max_notes:
            break

        # Points near the cardioid boundary produce interesting orbits
        r = 0.5 * (1 - np.cos(angle))
        c = complex(r * np.cos(angle) - 0.25, r * np.sin(angle))

        # Slightly perturb outward to get escaping orbits with interesting behavior
        c *= 1.02

        orbit = mandelbrot_orbit(c, max_iter=50)

        # Use orbit properties for music
        for i, z in enumerate(orbit[1:]):
            if len(notes) >= max_notes:
                break

            magnitude = min(abs(z), 2.0) / 2.0  # Normalize to 0-1
            phase = (np.angle(z) + np.pi) / (2 * np.pi)  # Normalize to 0-1

            pitch = quantize_to_scale(phase, scale_notes)
            velocity = value_to_velocity(magnitude, 50, 110)
            duration = 0.25 + magnitude * 0.5  # Longer notes for larger magnitudes

            notes.append({
                "pitch": pitch,
                "start": current_time,
                "duration": duration,
                "velocity": velocity,
            })
            current_time += 0.25

    return notes


def julia_melody(c: complex = complex(-0.7, 0.27015), num_points: int = 30,
                 root: int = 60, scale: str = "dorian",
                 max_notes: int = 200) -> list[dict]:
    """Generate melody from Julia set orbit paths.

    Args:
        c: Julia set parameter.
        num_points: Number of starting z0 values to sample.
        root: Root MIDI note.
        scale: Scale name.
        max_notes: Maximum notes.

    Returns:
        List of note events.
    """
    scale_notes = get_scale_notes(root, scale, octaves=3)
    notes = []
    current_time = 0.0

    # Sample starting points along a circle
    for i in range(num_points):
        if len(notes) >= max_notes:
            break

        angle = 2 * np.pi * i / num_points
        z0 = complex(0.5 * np.cos(angle), 0.5 * np.sin(angle))
        orbit = julia_orbit(z0, c, max_iter=30)

        for z in orbit[1:]:
            if len(notes) >= max_notes:
                break

            magnitude = min(abs(z), 2.0) / 2.0
            phase = (np.angle(z) + np.pi) / (2 * np.pi)

            pitch = quantize_to_scale(phase, scale_notes)
            velocity = value_to_velocity(1.0 - magnitude, 40, 100)
            duration = 0.25

            notes.append({
                "pitch": pitch,
                "start": current_time,
                "duration": duration,
                "velocity": velocity,
            })
            current_time += 0.25

    return notes


def sierpinski_melody(num_points: int = 200, root: int = 60,
                      scale: str = "pentatonic",
                      max_notes: int = 200) -> list[dict]:
    """Generate melody from Sierpinski triangle chaos game.

    Args:
        num_points: Number of chaos game points.
        root: Root MIDI note.
        scale: Scale name.
        max_notes: Maximum notes.

    Returns:
        List of note events.
    """
    scale_notes = get_scale_notes(root, scale, octaves=3)
    points = sierpinski_walk(num_points)
    notes = []
    current_time = 0.0

    for x, y in points[:max_notes]:
        pitch = quantize_to_scale(y, scale_notes)
        velocity = value_to_velocity(x, 50, 100)
        duration = 0.25 + 0.25 * x

        notes.append({
            "pitch": pitch,
            "start": current_time,
            "duration": duration,
            "velocity": velocity,
        })
        current_time += 0.25

    return notes


def koch_melody(iterations: int = 4, root: int = 60, scale: str = "major",
                max_notes: int = 200) -> list[dict]:
    """Generate melody from Koch curve geometry.

    Args:
        iterations: Koch curve subdivision depth.
        root: Root MIDI note.
        scale: Scale name.
        max_notes: Maximum notes.

    Returns:
        List of note events.
    """
    scale_notes = get_scale_notes(root, scale, octaves=3)
    points = koch_curve_points(iterations)
    notes = []
    current_time = 0.0

    for i, (x, y) in enumerate(points[:max_notes]):
        pitch = quantize_to_scale(y * 0.8 + 0.1, scale_notes)
        velocity = value_to_velocity(0.3 + 0.4 * abs(np.sin(i * 0.1)))
        duration = 0.25

        notes.append({
            "pitch": pitch,
            "start": current_time,
            "duration": duration,
            "velocity": velocity,
        })
        current_time += 0.125  # Eighth notes for denser texture

    return notes
