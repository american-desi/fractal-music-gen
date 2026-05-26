"""Music theory utilities: scales, chords, rhythm patterns, quantization."""

from __future__ import annotations

import numpy as np


# Scale definitions as semitone intervals from root
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "pentatonic": [0, 2, 4, 7, 9],
    "blues": [0, 3, 5, 6, 7, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "chromatic": list(range(12)),
    "whole_tone": [0, 2, 4, 6, 8, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
}

# Common chord progressions as scale degree offsets (0-indexed)
CHORD_PROGRESSIONS = {
    "I-IV-V-I": [0, 3, 4, 0],
    "I-V-vi-IV": [0, 4, 5, 3],
    "ii-V-I": [1, 4, 0],
    "I-vi-IV-V": [0, 5, 3, 4],
    "i-iv-v-i": [0, 3, 4, 0],
    "I-ii-V-I": [0, 1, 4, 0],
    "vi-IV-I-V": [5, 3, 0, 4],
}

# Rhythm patterns as lists of (duration_in_beats, is_rest) tuples
RHYTHM_PATTERNS = {
    "steady_quarter": [(1.0, False)] * 4,
    "dotted": [(1.5, False), (0.5, False), (1.5, False), (0.5, False)],
    "syncopated": [(0.5, False), (1.0, False), (0.5, False), (1.0, False), (1.0, False)],
    "waltz": [(1.0, False), (0.5, False), (0.5, False), (0.5, False), (0.5, False)],
    "march": [(1.0, False), (0.5, False), (0.5, False), (1.0, False), (1.0, False)],
    "triplet": [(1.0 / 3, False)] * 12,
    "swing": [(0.67, False), (0.33, False)] * 4,
    "rest_heavy": [(1.0, False), (0.5, True), (0.5, False), (1.0, False), (1.0, True)],
}


def get_scale_notes(root_midi: int, scale_name: str, octaves: int = 3) -> list[int]:
    """Get MIDI note numbers for a scale across multiple octaves.

    Args:
        root_midi: MIDI note number of the root (e.g., 60 for middle C).
        scale_name: Name of the scale from SCALES dict.
        octaves: Number of octaves to span.

    Returns:
        List of MIDI note numbers in the scale.
    """
    intervals = SCALES.get(scale_name, SCALES["major"])
    notes = []
    for octave in range(octaves):
        for interval in intervals:
            note = root_midi + octave * 12 + interval
            if 0 <= note <= 127:
                notes.append(note)
    return sorted(set(notes))


def quantize_to_scale(value: float, scale_notes: list[int]) -> int:
    """Map a continuous value (0.0 - 1.0) to the nearest note in a scale.

    Args:
        value: Normalized value between 0 and 1.
        scale_notes: List of valid MIDI note numbers.

    Returns:
        Closest MIDI note number from the scale.
    """
    value = max(0.0, min(1.0, value))
    index = int(value * (len(scale_notes) - 1))
    return scale_notes[index]


def quantize_time(time_value: float, grid: float = 0.25) -> float:
    """Quantize a time value to a rhythmic grid.

    Args:
        time_value: Time in beats.
        grid: Grid resolution in beats (0.25 = sixteenth note).

    Returns:
        Quantized time value.
    """
    return round(time_value / grid) * grid


def value_to_velocity(value: float, min_vel: int = 40, max_vel: int = 110) -> int:
    """Map a normalized value to MIDI velocity.

    Args:
        value: Normalized value between 0 and 1.
        min_vel: Minimum velocity.
        max_vel: Maximum velocity.

    Returns:
        MIDI velocity (0-127).
    """
    value = max(0.0, min(1.0, value))
    vel = int(min_vel + value * (max_vel - min_vel))
    return max(0, min(127, vel))


def build_chord(root_midi: int, scale_intervals: list[int], degree: int,
                num_notes: int = 3) -> list[int]:
    """Build a chord on a given scale degree.

    Args:
        root_midi: MIDI note of the scale root.
        scale_intervals: Scale intervals (from SCALES dict).
        degree: Scale degree (0-indexed).
        num_notes: Number of chord tones (3 = triad, 4 = seventh).

    Returns:
        List of MIDI note numbers forming the chord.
    """
    all_notes = []
    for octave in range(4):
        for interval in scale_intervals:
            all_notes.append(root_midi + octave * 12 + interval)

    all_notes = sorted(set(n for n in all_notes if 0 <= n <= 127))

    if degree >= len(all_notes):
        degree = degree % len(all_notes)

    chord = []
    idx = degree
    for i in range(num_notes):
        step = idx + i * 2  # stack thirds (every other scale degree)
        if step < len(all_notes):
            chord.append(all_notes[step])

    return chord


def get_rhythm_durations(pattern_name: str, num_beats: int) -> list[tuple[float, bool]]:
    """Get rhythm pattern extended to fill a given number of beats.

    Args:
        pattern_name: Name from RHYTHM_PATTERNS.
        num_beats: Total beats to fill.

    Returns:
        List of (duration, is_rest) tuples.
    """
    pattern = RHYTHM_PATTERNS.get(pattern_name, RHYTHM_PATTERNS["steady_quarter"])
    pattern_length = sum(d for d, _ in pattern)

    result = []
    total = 0.0
    while total < num_beats:
        for dur, is_rest in pattern:
            if total + dur > num_beats:
                remaining = num_beats - total
                if remaining > 0.01:
                    result.append((remaining, is_rest))
                    total = num_beats
                break
            result.append((dur, is_rest))
            total += dur

    return result


def midi_to_freq(midi_note: int) -> float:
    """Convert MIDI note number to frequency in Hz.

    Args:
        midi_note: MIDI note number (0-127).

    Returns:
        Frequency in Hz.
    """
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


def note_name(midi_note: int) -> str:
    """Get the name of a MIDI note.

    Args:
        midi_note: MIDI note number.

    Returns:
        Note name string like 'C4', 'F#3'.
    """
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = (midi_note // 12) - 1
    name = names[midi_note % 12]
    return f"{name}{octave}"
