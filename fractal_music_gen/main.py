"""Generate multiple pieces from different mathematical sources."""

import os
import time
import numpy as np

from fractal_music_gen.lsystem import generate_lsystem_melody
from fractal_music_gen.fractals import mandelbrot_melody, julia_melody, sierpinski_melody
from fractal_music_gen.cellular import generate_ca_piece
from fractal_music_gen.chaos import lorenz_melody, logistic_melody, henon_melody
from fractal_music_gen.composer import compose_piece
from fractal_music_gen.midi_writer import notes_to_midi
from fractal_music_gen.wav_writer import notes_to_wav


def ensure_output_dir() -> str:
    """Create and return the output directory path."""
    # Find the project root (parent of the fractal_music_gen package)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def generate_lsystem_piece(output_dir: str):
    """Piece 1: L-System Fibonacci melody with bass and pad."""
    print("\n--- Piece 1: L-System Fibonacci ---")
    print("  Generating L-system string and interpreting as melody...")

    np.random.seed(42)
    melody = generate_lsystem_melody(
        system_name="fibonacci",
        iterations=8,
        root=62,  # D
        scale="dorian",
        max_notes=250,
    )
    print(f"  Generated {len(melody)} melody notes")

    comp = compose_piece(
        melody_notes=melody,
        bpm=130.0,
        root=62,
        scale="dorian",
        target_duration_sec=45.0,
        add_bass=True,
        add_pad=True,
    )

    midi_path = os.path.join(output_dir, "01_lsystem_fibonacci.mid")
    wav_path = os.path.join(output_dir, "01_lsystem_fibonacci.wav")

    comp.save_midi(midi_path)
    print(f"  Saved MIDI: {midi_path}")

    comp.save_wav(wav_path)
    print(f"  Saved WAV: {wav_path} ({comp.get_duration_seconds():.1f}s)")


def generate_mandelbrot_piece(output_dir: str):
    """Piece 2: Mandelbrot orbit melody."""
    print("\n--- Piece 2: Mandelbrot Orbits ---")
    print("  Sampling orbits along Mandelbrot set boundary...")

    melody = mandelbrot_melody(
        num_points=40,
        root=57,  # A
        scale="harmonic_minor",
        max_notes=250,
    )
    print(f"  Generated {len(melody)} melody notes")

    comp = compose_piece(
        melody_notes=melody,
        bpm=100.0,
        root=57,
        scale="harmonic_minor",
        target_duration_sec=50.0,
        add_bass=True,
        add_pad=True,
    )

    midi_path = os.path.join(output_dir, "02_mandelbrot_orbits.mid")
    wav_path = os.path.join(output_dir, "02_mandelbrot_orbits.wav")

    comp.save_midi(midi_path)
    print(f"  Saved MIDI: {midi_path}")

    comp.save_wav(wav_path)
    print(f"  Saved WAV: {wav_path} ({comp.get_duration_seconds():.1f}s)")


def generate_cellular_piece(output_dir: str):
    """Piece 3: Rule 30 cellular automaton."""
    print("\n--- Piece 3: Cellular Automaton Rule 30 ---")
    print("  Running Rule 30 cellular automaton...")

    melody = generate_ca_piece(
        rule=30,
        root=64,  # E
        scale="blues",
        width=64,
        steps=120,
        max_notes=250,
    )
    print(f"  Generated {len(melody)} melody notes")

    comp = compose_piece(
        melody_notes=melody,
        bpm=110.0,
        root=64,
        scale="blues",
        target_duration_sec=45.0,
        add_bass=True,
        add_pad=True,
    )

    midi_path = os.path.join(output_dir, "03_rule30_cellular.mid")
    wav_path = os.path.join(output_dir, "03_rule30_cellular.wav")

    comp.save_midi(midi_path)
    print(f"  Saved MIDI: {midi_path}")

    comp.save_wav(wav_path)
    print(f"  Saved WAV: {wav_path} ({comp.get_duration_seconds():.1f}s)")


def generate_lorenz_piece(output_dir: str):
    """Piece 4: Lorenz attractor trajectory."""
    print("\n--- Piece 4: Lorenz Attractor ---")
    print("  Computing Lorenz attractor trajectory...")

    melody = lorenz_melody(
        num_steps=3000,
        root=60,  # C
        scale="minor",
        sample_rate=12,
        max_notes=250,
    )
    print(f"  Generated {len(melody)} melody notes")

    comp = compose_piece(
        melody_notes=melody,
        bpm=120.0,
        root=60,
        scale="minor",
        target_duration_sec=50.0,
        add_bass=True,
        add_pad=True,
    )

    midi_path = os.path.join(output_dir, "04_lorenz_attractor.mid")
    wav_path = os.path.join(output_dir, "04_lorenz_attractor.wav")

    comp.save_midi(midi_path)
    print(f"  Saved MIDI: {midi_path}")

    comp.save_wav(wav_path)
    print(f"  Saved WAV: {wav_path} ({comp.get_duration_seconds():.1f}s)")


def generate_logistic_piece(output_dir: str):
    """Piece 5 (bonus): Logistic map at the edge of chaos."""
    print("\n--- Piece 5: Logistic Map ---")
    print("  Iterating logistic map at edge of chaos (r=3.82)...")

    melody = logistic_melody(
        r=3.82,
        root=67,  # G
        scale="mixolydian",
        max_notes=200,
    )
    print(f"  Generated {len(melody)} melody notes")

    comp = compose_piece(
        melody_notes=melody,
        bpm=135.0,
        root=67,
        scale="mixolydian",
        target_duration_sec=40.0,
        add_bass=True,
        add_pad=False,
    )

    midi_path = os.path.join(output_dir, "05_logistic_map.mid")
    wav_path = os.path.join(output_dir, "05_logistic_map.wav")

    comp.save_midi(midi_path)
    print(f"  Saved MIDI: {midi_path}")

    comp.save_wav(wav_path)
    print(f"  Saved WAV: {wav_path} ({comp.get_duration_seconds():.1f}s)")


def main():
    """Generate all pieces."""
    print("=" * 60)
    print("  FRACTAL MUSIC GENERATOR")
    print("  Generating music from mathematical structures")
    print("=" * 60)

    start_time = time.time()
    output_dir = ensure_output_dir()
    print(f"\nOutput directory: {output_dir}")

    generate_lsystem_piece(output_dir)
    generate_mandelbrot_piece(output_dir)
    generate_cellular_piece(output_dir)
    generate_lorenz_piece(output_dir)
    generate_logistic_piece(output_dir)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"  Generation complete! ({elapsed:.1f}s)")
    print(f"  Output files in: {output_dir}")

    # List all generated files
    files = sorted(os.listdir(output_dir))
    print(f"\n  Generated {len(files)} files:")
    for f in files:
        size = os.path.getsize(os.path.join(output_dir, f))
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} bytes"
        print(f"    {f} ({size_str})")

    print("=" * 60)


if __name__ == "__main__":
    main()
