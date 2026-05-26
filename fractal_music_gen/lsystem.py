"""L-system engine with parametric rules and turtle interpretation for music."""

from __future__ import annotations

import numpy as np
from fractal_music_gen.music_theory import quantize_to_scale, get_scale_notes, value_to_velocity


class LSystem:
    """Lindenmayer system string rewriter.

    Supports deterministic and stochastic rules with parametric extensions.
    """

    def __init__(self, axiom: str, rules: dict[str, str | list[tuple[str, float]]]):
        """Initialize L-system.

        Args:
            axiom: Starting string.
            rules: Dict mapping symbols to replacement strings. Values can be
                   a single string or a list of (string, probability) tuples
                   for stochastic rules.
        """
        self.axiom = axiom
        self.rules = rules

    def generate(self, iterations: int) -> str:
        """Apply production rules iteratively.

        Args:
            iterations: Number of rewriting iterations.

        Returns:
            Resulting string after all iterations.
        """
        current = self.axiom
        for _ in range(iterations):
            next_str = []
            for char in current:
                if char in self.rules:
                    rule = self.rules[char]
                    if isinstance(rule, list):
                        # Stochastic: pick based on probability
                        r = np.random.random()
                        cumulative = 0.0
                        chosen = rule[0][0]
                        for replacement, prob in rule:
                            cumulative += prob
                            if r <= cumulative:
                                chosen = replacement
                                break
                        next_str.append(chosen)
                    else:
                        next_str.append(rule)
                else:
                    next_str.append(char)
            current = "".join(next_str)
        return current


class TurtleMusicInterpreter:
    """Interpret L-system strings as musical events using turtle graphics metaphor.

    Symbols:
        F, A, B: Play note (move forward = advance in time)
        +: Increase pitch by step
        -: Decrease pitch by step
        >: Increase velocity
        <: Decrease velocity
        [: Push state (pitch, velocity)
        ]: Pop state
        .: Rest (advance time without playing)
    """

    def __init__(self, scale_notes: list[int], base_duration: float = 0.25,
                 pitch_step: int = 1, velocity_step: int = 10):
        """Initialize turtle interpreter.

        Args:
            scale_notes: List of valid MIDI notes to use.
            base_duration: Duration of each note in beats.
            pitch_step: How many scale steps to move per +/- symbol.
            velocity_step: How much to change velocity per >/<.
        """
        self.scale_notes = scale_notes
        self.base_duration = base_duration
        self.pitch_step = pitch_step
        self.velocity_step = velocity_step

    def interpret(self, lstring: str, max_notes: int = 500) -> list[dict]:
        """Convert L-system string to musical note events.

        Args:
            lstring: L-system output string.
            max_notes: Maximum number of notes to generate.

        Returns:
            List of note event dicts with keys:
            pitch (MIDI), start (beats), duration (beats), velocity (0-127).
        """
        notes = []
        pitch_index = len(self.scale_notes) // 2  # Start in middle of range
        velocity = 80
        current_time = 0.0
        stack = []
        note_count = 0

        for char in lstring:
            if note_count >= max_notes:
                break

            if char in ("F", "A", "B"):
                # Clamp pitch index
                pitch_index = max(0, min(len(self.scale_notes) - 1, pitch_index))
                midi_note = self.scale_notes[pitch_index]
                vel = max(1, min(127, velocity))
                notes.append({
                    "pitch": midi_note,
                    "start": current_time,
                    "duration": self.base_duration,
                    "velocity": vel,
                })
                current_time += self.base_duration
                note_count += 1

            elif char == "+":
                pitch_index += self.pitch_step
            elif char == "-":
                pitch_index -= self.pitch_step
            elif char == ">":
                velocity = min(127, velocity + self.velocity_step)
            elif char == "<":
                velocity = max(1, velocity - self.velocity_step)
            elif char == "[":
                stack.append((pitch_index, velocity))
            elif char == "]":
                if stack:
                    pitch_index, velocity = stack.pop()
            elif char == ".":
                current_time += self.base_duration

        return notes


# Predefined L-systems that produce interesting music
MUSICAL_LSYSTEMS = {
    "algae": LSystem(
        axiom="A",
        rules={"A": "AB", "B": "A"}
    ),
    "fibonacci": LSystem(
        axiom="A",
        rules={"A": "B+A", "B": "A-B"}
    ),
    "dragon_curve": LSystem(
        axiom="F",
        rules={"F": "F+G", "G": "F-G"}
    ),
    "koch_melody": LSystem(
        axiom="F",
        rules={"F": "F+F-F-F+F"}
    ),
    "plant": LSystem(
        axiom="A",
        rules={"A": "A[+B]A[-B]+B", "B": "BA"}
    ),
    "pentatonic_walk": LSystem(
        axiom="F",
        rules={"F": "F+F-F.F+F-F"}
    ),
    "stochastic_melody": LSystem(
        axiom="A",
        rules={
            "A": [("A+B-A", 0.5), ("A-B+A", 0.3), ("B+A+B", 0.2)],
            "B": [("B-A+B", 0.6), ("A-B-A", 0.4)],
        }
    ),
}


def generate_lsystem_melody(system_name: str = "fibonacci",
                            iterations: int = 6,
                            root: int = 60,
                            scale: str = "pentatonic",
                            bpm: float = 120.0,
                            max_notes: int = 200) -> list[dict]:
    """Generate a melody from a named L-system.

    Args:
        system_name: Name from MUSICAL_LSYSTEMS.
        iterations: L-system rewriting iterations.
        root: Root MIDI note.
        scale: Scale name.
        bpm: Tempo (used for duration scaling).
        max_notes: Maximum notes to produce.

    Returns:
        List of note events.
    """
    lsystem = MUSICAL_LSYSTEMS.get(system_name, MUSICAL_LSYSTEMS["fibonacci"])
    lstring = lsystem.generate(iterations)

    scale_notes = get_scale_notes(root, scale, octaves=3)
    beat_dur = 0.25  # sixteenth notes
    interpreter = TurtleMusicInterpreter(
        scale_notes=scale_notes,
        base_duration=beat_dur,
        pitch_step=1,
        velocity_step=8,
    )
    return interpreter.interpret(lstring, max_notes=max_notes)
