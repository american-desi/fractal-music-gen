"""Pure Python WAV synthesizer with sine/square/sawtooth waves and ADSR envelopes."""

from __future__ import annotations

import struct
import numpy as np
from fractal_music_gen.music_theory import midi_to_freq


class ADSREnvelope:
    """Attack-Decay-Sustain-Release envelope generator."""

    def __init__(self, attack: float = 0.02, decay: float = 0.05,
                 sustain_level: float = 0.7, release: float = 0.1):
        """Initialize ADSR envelope.

        Args:
            attack: Attack time in seconds.
            decay: Decay time in seconds.
            sustain_level: Sustain amplitude (0-1).
            release: Release time in seconds.
        """
        self.attack = attack
        self.decay = decay
        self.sustain_level = sustain_level
        self.release = release

    def generate(self, duration: float, sample_rate: int = 44100) -> np.ndarray:
        """Generate envelope samples.

        Args:
            duration: Total note duration in seconds.
            sample_rate: Audio sample rate.

        Returns:
            1D numpy array of envelope values (0-1).
        """
        total_samples = int(duration * sample_rate)
        if total_samples == 0:
            return np.array([], dtype=np.float64)

        envelope = np.zeros(total_samples, dtype=np.float64)

        attack_samples = int(self.attack * sample_rate)
        decay_samples = int(self.decay * sample_rate)
        release_samples = int(self.release * sample_rate)

        # Ensure we don't exceed total duration
        sustain_samples = max(0, total_samples - attack_samples - decay_samples - release_samples)
        if sustain_samples == 0:
            # Short note: compress envelope
            attack_samples = min(attack_samples, total_samples // 3)
            decay_samples = min(decay_samples, total_samples // 3)
            release_samples = total_samples - attack_samples - decay_samples
            sustain_samples = 0

        pos = 0

        # Attack
        if attack_samples > 0:
            envelope[pos:pos + attack_samples] = np.linspace(0, 1, attack_samples)
            pos += attack_samples

        # Decay
        if decay_samples > 0:
            envelope[pos:pos + decay_samples] = np.linspace(1, self.sustain_level, decay_samples)
            pos += decay_samples

        # Sustain
        if sustain_samples > 0:
            envelope[pos:pos + sustain_samples] = self.sustain_level
            pos += sustain_samples

        # Release
        remaining = total_samples - pos
        if remaining > 0:
            envelope[pos:] = np.linspace(self.sustain_level, 0, remaining)

        return envelope


def sine_wave(freq: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """Generate a sine wave.

    Args:
        freq: Frequency in Hz.
        duration: Duration in seconds.
        sample_rate: Audio sample rate.

    Returns:
        1D numpy array of samples.
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return np.sin(2 * np.pi * freq * t)


def square_wave(freq: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """Generate a square wave (band-limited approximation).

    Args:
        freq: Frequency in Hz.
        duration: Duration in seconds.
        sample_rate: Audio sample rate.

    Returns:
        1D numpy array of samples.
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    # Band-limited: sum of odd harmonics
    signal = np.zeros_like(t)
    max_harmonic = int(sample_rate / (2 * freq))
    for k in range(1, min(max_harmonic, 20) + 1, 2):
        signal += np.sin(2 * np.pi * k * freq * t) / k
    return signal * (4 / np.pi)


def sawtooth_wave(freq: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """Generate a sawtooth wave (band-limited approximation).

    Args:
        freq: Frequency in Hz.
        duration: Duration in seconds.
        sample_rate: Audio sample rate.

    Returns:
        1D numpy array of samples.
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    signal = np.zeros_like(t)
    max_harmonic = int(sample_rate / (2 * freq))
    for k in range(1, min(max_harmonic, 20) + 1):
        signal += np.sin(2 * np.pi * k * freq * t) / k
    return signal * (2 / np.pi)


def triangle_wave(freq: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """Generate a triangle wave (band-limited approximation).

    Args:
        freq: Frequency in Hz.
        duration: Duration in seconds.
        sample_rate: Audio sample rate.

    Returns:
        1D numpy array of samples.
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    signal = np.zeros_like(t)
    max_harmonic = int(sample_rate / (2 * freq))
    for k in range(0, min(max_harmonic, 15) + 1):
        n = 2 * k + 1
        signal += ((-1) ** k) * np.sin(2 * np.pi * n * freq * t) / (n * n)
    return signal * (8 / (np.pi ** 2))


WAVEFORMS = {
    "sine": sine_wave,
    "square": square_wave,
    "sawtooth": sawtooth_wave,
    "triangle": triangle_wave,
}


class WavSynthesizer:
    """Synthesize audio from note events and write WAV files."""

    def __init__(self, sample_rate: int = 44100, waveform: str = "sine",
                 adsr: ADSREnvelope | None = None):
        """Initialize synthesizer.

        Args:
            sample_rate: Audio sample rate in Hz.
            waveform: Waveform type ('sine', 'square', 'sawtooth', 'triangle').
            adsr: ADSR envelope. If None, uses default.
        """
        self.sample_rate = sample_rate
        self.waveform_func = WAVEFORMS.get(waveform, sine_wave)
        self.adsr = adsr or ADSREnvelope()

    def render_notes(self, notes: list[dict], bpm: float = 120.0,
                     master_volume: float = 0.5) -> np.ndarray:
        """Render note events to audio samples.

        Args:
            notes: List of note dicts with pitch, start, duration, velocity.
            bpm: Tempo in beats per minute.
            master_volume: Overall volume scaling (0-1).

        Returns:
            1D numpy array of audio samples (float64, -1 to 1).
        """
        if not notes:
            return np.zeros(self.sample_rate, dtype=np.float64)

        seconds_per_beat = 60.0 / bpm

        # Calculate total duration
        max_end = 0
        for note in notes:
            end = (note["start"] + note["duration"]) * seconds_per_beat
            if end > max_end:
                max_end = end

        total_samples = int((max_end + 0.5) * self.sample_rate)  # Extra padding
        audio = np.zeros(total_samples, dtype=np.float64)

        for note in notes:
            freq = midi_to_freq(note["pitch"])
            start_sec = note["start"] * seconds_per_beat
            dur_sec = note["duration"] * seconds_per_beat
            velocity_scale = note.get("velocity", 80) / 127.0

            # Generate waveform
            wave = self.waveform_func(freq, dur_sec, self.sample_rate)

            # Apply ADSR envelope
            envelope = self.adsr.generate(dur_sec, self.sample_rate)

            # Match lengths
            min_len = min(len(wave), len(envelope))
            wave = wave[:min_len] * envelope[:min_len]

            # Apply velocity
            wave *= velocity_scale * master_volume

            # Mix into output buffer
            start_sample = int(start_sec * self.sample_rate)
            end_sample = start_sample + len(wave)

            if end_sample > len(audio):
                audio = np.pad(audio, (0, end_sample - len(audio)))

            audio[start_sample:end_sample] += wave

        # Normalize to prevent clipping
        peak = np.max(np.abs(audio))
        if peak > 0.95:
            audio = audio * 0.95 / peak

        return audio

    def render_multi_track(self, tracks: list[dict], bpm: float = 120.0,
                           master_volume: float = 0.4) -> np.ndarray:
        """Render multiple tracks, each with potentially different waveforms.

        Args:
            tracks: List of dicts with keys:
                - notes: list of note events
                - waveform: waveform name (optional)
                - volume: track volume 0-1 (optional)
            bpm: Tempo.
            master_volume: Overall volume.

        Returns:
            Mixed audio as 1D numpy array.
        """
        if not tracks:
            return np.zeros(self.sample_rate, dtype=np.float64)

        rendered = []

        for track_info in tracks:
            wave_name = track_info.get("waveform", "sine")
            volume = track_info.get("volume", 0.5)
            synth = WavSynthesizer(
                sample_rate=self.sample_rate,
                waveform=wave_name,
                adsr=self.adsr,
            )
            audio = synth.render_notes(
                track_info.get("notes", []),
                bpm=bpm,
                master_volume=volume * master_volume,
            )
            rendered.append(audio)

        max_len = max(len(a) for a in rendered) if rendered else 0

        # Mix all tracks
        mixed = np.zeros(max_len, dtype=np.float64)
        for audio in rendered:
            mixed[:len(audio)] += audio

        # Normalize
        peak = np.max(np.abs(mixed))
        if peak > 0.95:
            mixed = mixed * 0.95 / peak

        return mixed


def write_wav(filepath: str, audio: np.ndarray, sample_rate: int = 44100,
              bits_per_sample: int = 16):
    """Write audio data to a WAV file (raw binary, no external library).

    Args:
        filepath: Output file path.
        audio: 1D numpy array of float samples (-1 to 1).
        sample_rate: Audio sample rate.
        bits_per_sample: Bit depth (16 or 24).
    """
    num_channels = 1
    num_samples = len(audio)

    # Clamp to -1..1
    audio = np.clip(audio, -1.0, 1.0)

    # Convert to integer samples
    if bits_per_sample == 16:
        max_val = 32767
        int_audio = (audio * max_val).astype(np.int16)
        sample_data = int_audio.tobytes()
    elif bits_per_sample == 24:
        max_val = 8388607
        int_audio = (audio * max_val).astype(np.int32)
        sample_data = bytearray()
        for sample in int_audio:
            # Pack as 3 bytes, little-endian
            sample_data.extend(struct.pack("<i", sample)[:3])
        sample_data = bytes(sample_data)
    else:
        raise ValueError(f"Unsupported bits_per_sample: {bits_per_sample}")

    bytes_per_sample = bits_per_sample // 8
    byte_rate = sample_rate * num_channels * bytes_per_sample
    block_align = num_channels * bytes_per_sample
    data_size = num_samples * bytes_per_sample * num_channels

    with open(filepath, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))  # File size - 8
        f.write(b"WAVE")

        # fmt subchunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))  # Subchunk size
        f.write(struct.pack("<H", 1))   # PCM format
        f.write(struct.pack("<H", num_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))

        # data subchunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(sample_data)


def notes_to_wav(notes: list[dict], filepath: str, bpm: float = 120.0,
                 waveform: str = "sine", sample_rate: int = 44100):
    """Convenience: render notes and write to WAV in one call.

    Args:
        notes: List of note event dicts.
        filepath: Output WAV file path.
        bpm: Tempo.
        waveform: Waveform type.
        sample_rate: Audio sample rate.
    """
    synth = WavSynthesizer(sample_rate=sample_rate, waveform=waveform)
    audio = synth.render_notes(notes, bpm=bpm)
    write_wav(filepath, audio, sample_rate=sample_rate)
