"""Pure Python MIDI file writer - writes raw MIDI binary format directly."""


from __future__ import annotations
import struct


def _write_variable_length(value: int) -> bytes:
    """Encode an integer as MIDI variable-length quantity.

    Args:
        value: Non-negative integer to encode.

    Returns:
        Bytes representing the variable-length quantity.
    """
    if value < 0:
        raise ValueError("Variable-length value must be non-negative")

    result = []
    result.append(value & 0x7F)
    value >>= 7
    while value > 0:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.reverse()
    return bytes(result)


class MidiTrack:
    """Represents a single MIDI track with note and control events."""

    def __init__(self, name: str = "", channel: int = 0):
        """Initialize a MIDI track.

        Args:
            name: Track name (optional).
            channel: MIDI channel (0-15).
        """
        self.name = name
        self.channel = channel & 0x0F
        self.events: list[tuple[int, bytes]] = []  # (absolute_tick, raw_event_bytes)

    def add_note(self, pitch: int, start_tick: int, duration_ticks: int,
                 velocity: int = 80):
        """Add a note event (note on + note off pair).

        Args:
            pitch: MIDI note number (0-127).
            start_tick: Start time in ticks.
            duration_ticks: Duration in ticks.
            velocity: Note velocity (0-127).
        """
        pitch = max(0, min(127, pitch))
        velocity = max(1, min(127, velocity))

        # Note On: 0x90 | channel
        note_on = bytes([0x90 | self.channel, pitch, velocity])
        self.events.append((start_tick, note_on))

        # Note Off: 0x80 | channel
        note_off = bytes([0x80 | self.channel, pitch, 0])
        self.events.append((start_tick + duration_ticks, note_off))

    def add_program_change(self, program: int, tick: int = 0):
        """Add a program (instrument) change event.

        Args:
            program: MIDI program number (0-127).
            tick: Time in ticks.
        """
        program = max(0, min(127, program))
        event = bytes([0xC0 | self.channel, program])
        self.events.append((tick, event))

    def add_control_change(self, controller: int, value: int, tick: int = 0):
        """Add a control change event.

        Args:
            controller: Controller number (0-127).
            value: Controller value (0-127).
            tick: Time in ticks.
        """
        event = bytes([0xB0 | self.channel, controller & 0x7F, value & 0x7F])
        self.events.append((tick, event))

    def add_tempo(self, bpm: float, tick: int = 0):
        """Add a tempo meta event.

        Args:
            bpm: Tempo in beats per minute.
            tick: Time in ticks.
        """
        microseconds_per_beat = int(60_000_000 / bpm)
        tempo_bytes = struct.pack(">I", microseconds_per_beat)[1:]  # 3 bytes
        event = bytes([0xFF, 0x51, 0x03]) + tempo_bytes
        self.events.append((tick, event))

    def _build_track_data(self) -> bytes:
        """Convert events to MIDI track chunk data with delta times.

        Returns:
            Raw bytes for the track data (excluding chunk header).
        """
        # Sort events by tick time, with note-offs before note-ons at same tick
        def sort_key(ev):
            tick, data = ev
            # Note offs (0x80) before note ons (0x90) at same tick
            event_type = data[0] & 0xF0 if data[0] != 0xFF else 0x00
            return (tick, 0 if event_type == 0x80 else 1)

        sorted_events = sorted(self.events, key=sort_key)

        data = bytearray()

        # Track name meta event
        if self.name:
            name_bytes = self.name.encode("ascii", errors="replace")
            data.extend(_write_variable_length(0))  # Delta time 0
            data.extend(bytes([0xFF, 0x03]))
            data.extend(_write_variable_length(len(name_bytes)))
            data.extend(name_bytes)

        prev_tick = 0
        for tick, event_data in sorted_events:
            delta = max(0, tick - prev_tick)
            data.extend(_write_variable_length(delta))
            data.extend(event_data)
            prev_tick = tick

        # End of track meta event
        data.extend(_write_variable_length(0))
        data.extend(bytes([0xFF, 0x2F, 0x00]))

        return bytes(data)


class MidiFile:
    """Represents a complete MIDI file with multiple tracks."""

    def __init__(self, ticks_per_beat: int = 480):
        """Initialize MIDI file.

        Args:
            ticks_per_beat: Resolution in ticks per quarter note.
        """
        self.ticks_per_beat = ticks_per_beat
        self.tracks: list[MidiTrack] = []

    def add_track(self, name: str = "", channel: int = 0) -> MidiTrack:
        """Create and add a new track.

        Args:
            name: Track name.
            channel: MIDI channel (0-15).

        Returns:
            The new MidiTrack instance.
        """
        track = MidiTrack(name=name, channel=channel)
        self.tracks.append(track)
        return track

    def beats_to_ticks(self, beats: float) -> int:
        """Convert beats to ticks.

        Args:
            beats: Time in beats (quarter notes).

        Returns:
            Time in ticks.
        """
        return int(beats * self.ticks_per_beat)

    def save(self, filepath: str):
        """Write MIDI file to disk.

        Args:
            filepath: Output file path.
        """
        with open(filepath, "wb") as f:
            # MIDI header chunk
            num_tracks = len(self.tracks)
            fmt = 1 if num_tracks > 1 else 0

            # "MThd" + length (6) + format + num_tracks + ticks_per_beat
            header = b"MThd"
            header += struct.pack(">I", 6)
            header += struct.pack(">HHH", fmt, num_tracks, self.ticks_per_beat)
            f.write(header)

            # Track chunks
            for track in self.tracks:
                track_data = track._build_track_data()
                f.write(b"MTrk")
                f.write(struct.pack(">I", len(track_data)))
                f.write(track_data)


def notes_to_midi(notes: list[dict], filepath: str, bpm: float = 120.0,
                  program: int = 0, channel: int = 0,
                  ticks_per_beat: int = 480):
    """Convenience function to write note events directly to a MIDI file.

    Args:
        notes: List of note dicts with keys: pitch, start, duration, velocity.
        filepath: Output MIDI file path.
        bpm: Tempo in beats per minute.
        program: MIDI program (instrument) number.
        channel: MIDI channel.
        ticks_per_beat: Resolution.
    """
    midi = MidiFile(ticks_per_beat=ticks_per_beat)
    track = midi.add_track(name="Track 1", channel=channel)
    track.add_tempo(bpm, tick=0)
    track.add_program_change(program, tick=0)

    for note in notes:
        start_tick = midi.beats_to_ticks(note["start"])
        dur_ticks = max(1, midi.beats_to_ticks(note["duration"]))
        track.add_note(
            pitch=note["pitch"],
            start_tick=start_tick,
            duration_ticks=dur_ticks,
            velocity=note.get("velocity", 80),
        )

    midi.save(filepath)


def multi_track_to_midi(tracks_data: list[dict], filepath: str,
                        bpm: float = 120.0, ticks_per_beat: int = 480):
    """Write multiple tracks of notes to a single MIDI file.

    Args:
        tracks_data: List of dicts, each with keys:
            - notes: list of note event dicts
            - name: track name (optional)
            - program: MIDI program number (optional, default 0)
            - channel: MIDI channel (optional, default auto-assigned)
        filepath: Output MIDI file path.
        bpm: Tempo.
        ticks_per_beat: Resolution.
    """
    midi = MidiFile(ticks_per_beat=ticks_per_beat)

    for i, track_info in enumerate(tracks_data):
        channel = track_info.get("channel", i % 16)
        # Skip percussion channel 9 unless explicitly requested
        if channel == 9 and "channel" not in track_info:
            channel = (i + 1) % 16
            if channel == 9:
                channel = 10

        name = track_info.get("name", f"Track {i + 1}")
        program = track_info.get("program", 0)

        track = midi.add_track(name=name, channel=channel)

        # Only first track needs tempo
        if i == 0:
            track.add_tempo(bpm, tick=0)

        track.add_program_change(program, tick=0)

        for note in track_info.get("notes", []):
            start_tick = midi.beats_to_ticks(note["start"])
            dur_ticks = max(1, midi.beats_to_ticks(note["duration"]))
            track.add_note(
                pitch=note["pitch"],
                start_tick=start_tick,
                duration_ticks=dur_ticks,
                velocity=note.get("velocity", 80),
            )

    midi.save(filepath)
