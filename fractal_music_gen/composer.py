"""High-level composition engine combining generators with musical constraints."""

from __future__ import annotations

import numpy as np
from fractal_music_gen.music_theory import (
    get_scale_notes, build_chord, SCALES, CHORD_PROGRESSIONS,
    get_rhythm_durations, quantize_time,
)
from fractal_music_gen.midi_writer import MidiFile, multi_track_to_midi
from fractal_music_gen.wav_writer import WavSynthesizer, ADSREnvelope, write_wav


class Composition:
    """A musical composition built from multiple voice/track layers."""

    def __init__(self, bpm: float = 120.0, root: int = 60, scale: str = "minor",
                 time_signature: tuple[int, int] = (4, 4)):
        """Initialize composition.

        Args:
            bpm: Tempo in beats per minute.
            root: Root MIDI note number.
            scale: Scale name.
            time_signature: (beats_per_measure, beat_unit).
        """
        self.bpm = bpm
        self.root = root
        self.scale = scale
        self.time_signature = time_signature
        self.tracks: list[dict] = []

    def add_melody_track(self, notes: list[dict], name: str = "Melody",
                         program: int = 0, waveform: str = "sine",
                         volume: float = 0.6):
        """Add a melody track from pre-generated notes.

        Args:
            notes: List of note event dicts.
            name: Track name.
            program: MIDI program number.
            waveform: Waveform for WAV synthesis.
            volume: Track volume (0-1).
        """
        self.tracks.append({
            "notes": notes,
            "name": name,
            "program": program,
            "waveform": waveform,
            "volume": volume,
        })

    def add_bass_line(self, melody_notes: list[dict], name: str = "Bass",
                      program: int = 33, waveform: str = "sawtooth",
                      volume: float = 0.4):
        """Generate and add a bass line that follows the melody's harmonic rhythm.

        Creates a simple bass line by tracking the lowest notes in the melody
        and transposing them down.

        Args:
            melody_notes: Melody notes to derive bass from.
            name: Track name.
            program: MIDI program.
            waveform: Waveform for WAV.
            volume: Track volume.
        """
        if not melody_notes:
            return

        scale_notes = get_scale_notes(self.root - 24, self.scale, octaves=2)
        bass_notes = []

        # Group melody notes into measures
        beats_per_measure = self.time_signature[0]
        max_time = max(n["start"] for n in melody_notes)

        current_beat = 0
        while current_beat <= max_time:
            # Find melody notes in this measure
            measure_notes = [
                n for n in melody_notes
                if current_beat <= n["start"] < current_beat + beats_per_measure
            ]

            if measure_notes:
                # Use the pitch class of the first note as the bass root
                melody_pitch = measure_notes[0]["pitch"]
                # Find closest bass note
                pitch_class = melody_pitch % 12
                bass_pitch = self.root - 24 + pitch_class
                while bass_pitch < 36:
                    bass_pitch += 12
                while bass_pitch > 55:
                    bass_pitch -= 12

                # Quantize to scale
                if scale_notes:
                    bass_pitch = min(scale_notes, key=lambda n: abs(n - bass_pitch))

                # Simple bass pattern: root on 1, root on 3
                bass_notes.append({
                    "pitch": bass_pitch,
                    "start": current_beat,
                    "duration": 1.5,
                    "velocity": 70,
                })
                if beats_per_measure >= 4:
                    bass_notes.append({
                        "pitch": bass_pitch,
                        "start": current_beat + 2,
                        "duration": 1.5,
                        "velocity": 60,
                    })

            current_beat += beats_per_measure

        self.tracks.append({
            "notes": bass_notes,
            "name": name,
            "program": program,
            "waveform": waveform,
            "volume": volume,
        })

    def add_chord_pad(self, melody_notes: list[dict], name: str = "Pad",
                      program: int = 48, waveform: str = "triangle",
                      volume: float = 0.25):
        """Generate chord pads based on melody harmony.

        Args:
            melody_notes: Melody to harmonize.
            name: Track name.
            program: MIDI program (48 = strings).
            waveform: Waveform for WAV.
            volume: Track volume.
        """
        if not melody_notes:
            return

        scale_intervals = SCALES.get(self.scale, SCALES["minor"])
        pad_notes = []
        beats_per_measure = self.time_signature[0]
        max_time = max(n["start"] for n in melody_notes)

        current_beat = 0
        while current_beat <= max_time:
            measure_notes = [
                n for n in melody_notes
                if current_beat <= n["start"] < current_beat + beats_per_measure
            ]

            if measure_notes:
                # Determine chord root from average pitch of melody notes
                avg_pitch = int(np.mean([n["pitch"] for n in measure_notes]))
                degree = ((avg_pitch - self.root) % 12)
                # Find nearest scale degree
                if scale_intervals:
                    degree_idx = min(
                        range(len(scale_intervals)),
                        key=lambda i: abs(scale_intervals[i] - degree)
                    )
                else:
                    degree_idx = 0

                chord = build_chord(self.root, scale_intervals, degree_idx, num_notes=3)

                # Transpose chord to comfortable range
                chord = [max(48, min(72, p)) for p in chord]

                for pitch in chord:
                    pad_notes.append({
                        "pitch": pitch,
                        "start": current_beat,
                        "duration": beats_per_measure * 0.9,
                        "velocity": 50,
                    })

            current_beat += beats_per_measure

        self.tracks.append({
            "notes": pad_notes,
            "name": name,
            "program": program,
            "waveform": waveform,
            "volume": volume,
        })

    def trim_to_duration(self, target_beats: float):
        """Trim all tracks to a maximum duration in beats.

        Args:
            target_beats: Maximum duration in beats.
        """
        for track in self.tracks:
            track["notes"] = [
                n for n in track["notes"]
                if n["start"] < target_beats
            ]
            # Trim notes that extend past the end
            for note in track["notes"]:
                if note["start"] + note["duration"] > target_beats:
                    note["duration"] = target_beats - note["start"]

    def get_duration_beats(self) -> float:
        """Get the total duration of the composition in beats."""
        max_end = 0
        for track in self.tracks:
            for note in track.get("notes", []):
                end = note["start"] + note["duration"]
                if end > max_end:
                    max_end = end
        return max_end

    def get_duration_seconds(self) -> float:
        """Get total duration in seconds."""
        return self.get_duration_beats() * 60.0 / self.bpm

    def save_midi(self, filepath: str):
        """Save composition as MIDI file.

        Args:
            filepath: Output MIDI file path.
        """
        tracks_data = []
        for i, track in enumerate(self.tracks):
            tracks_data.append({
                "notes": track["notes"],
                "name": track["name"],
                "program": track.get("program", 0),
                "channel": i % 16,
            })

        multi_track_to_midi(tracks_data, filepath, bpm=self.bpm)

    def save_wav(self, filepath: str, sample_rate: int = 44100):
        """Save composition as WAV file.

        Args:
            filepath: Output WAV file path.
            sample_rate: Audio sample rate.
        """
        adsr = ADSREnvelope(attack=0.02, decay=0.08, sustain_level=0.6, release=0.15)
        synth = WavSynthesizer(sample_rate=sample_rate, adsr=adsr)
        audio = synth.render_multi_track(self.tracks, bpm=self.bpm, master_volume=0.4)
        write_wav(filepath, audio, sample_rate=sample_rate)


def compose_piece(melody_notes: list[dict], bpm: float = 120.0,
                  root: int = 60, scale: str = "minor",
                  target_duration_sec: float = 45.0,
                  add_bass: bool = True,
                  add_pad: bool = True) -> Composition:
    """Create a complete multi-track composition from a melody.

    Args:
        melody_notes: Generated melody note events.
        bpm: Tempo.
        root: Root MIDI note.
        scale: Scale name.
        target_duration_sec: Desired duration in seconds.
        add_bass: Whether to add a bass track.
        add_pad: Whether to add chord pads.

    Returns:
        Composition object ready to save.
    """
    comp = Composition(bpm=bpm, root=root, scale=scale)

    # Calculate target beats
    target_beats = target_duration_sec * bpm / 60.0

    comp.add_melody_track(melody_notes, waveform="square", volume=0.5)

    if add_bass:
        comp.add_bass_line(melody_notes, waveform="sawtooth", volume=0.35)

    if add_pad:
        comp.add_chord_pad(melody_notes, waveform="triangle", volume=0.2)

    comp.trim_to_duration(target_beats)

    return comp
