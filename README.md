# Fractal Music Generator

Generate music from mathematical structures including fractals, L-systems, cellular automata, and chaos theory. Outputs MIDI files and WAV audio.

## Requirements

- Python 3.10+
- numpy

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

This generates 5 pieces in the `output/` directory, each as both MIDI and WAV:

1. **L-System Fibonacci** - Melody derived from Fibonacci L-system string rewriting, Dorian mode
2. **Mandelbrot Orbits** - Orbits along the Mandelbrot set boundary mapped to harmonic minor
3. **Cellular Automaton Rule 30** - Wolfram Rule 30 evolution mapped to blues scale
4. **Lorenz Attractor** - Lorenz system trajectory mapped to C minor
5. **Logistic Map** - Logistic map iterations at edge of chaos in G Mixolydian

## Project Structure

```
fractal_music_gen/
  __init__.py
  lsystem.py        - L-system engine with parametric/stochastic rules
  fractals.py        - Mandelbrot, Julia, Sierpinski, Koch curve generators
  cellular.py        - 1D elementary cellular automata (Rules 30, 90, 110)
  chaos.py           - Lorenz attractor, logistic map, Henon map
  music_theory.py    - Scales, chords, rhythm patterns, quantization
  midi_writer.py     - Pure Python MIDI file writer (no external libs)
  wav_writer.py      - Pure Python WAV synthesizer with ADSR envelopes
  composer.py        - Multi-track composition engine
  main.py            - Generates all pieces
```

## Mathematical Sources

- **L-Systems**: String rewriting systems (Fibonacci, dragon curve, Koch, plant) interpreted via turtle graphics as pitch/duration sequences
- **Fractals**: Mandelbrot orbit escape sequences, Julia set paths, Sierpinski chaos game, Koch curve geometry
- **Cellular Automata**: Wolfram elementary CA rules mapped to notes via cell density and spatial distribution
- **Chaos Theory**: Lorenz attractor (3D trajectory to pitch/velocity/duration), logistic map (bifurcation to melody), Henon map

## Features

- Pure Python MIDI writer (raw binary MIDI format, no mido/pretty_midi)
- Pure Python WAV synthesizer (sine, square, sawtooth, triangle waveforms)
- ADSR envelope shaping
- Musical quantization to scales and time grids
- Multi-track compositions with auto-generated bass and chord pads
- 10 scale types (major, minor, pentatonic, blues, dorian, mixolydian, etc.)
- Each piece is 30-60 seconds
