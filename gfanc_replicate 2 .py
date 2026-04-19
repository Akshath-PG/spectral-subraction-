"""
GFANC Stethoscope — Python Replication of MATLAB Pipeline
==========================================================

Signal understanding:
  main.wav  → stethoscope mic: heart + lung sounds + faint ambient bleed
  anc.wav   → reference mic:   ambient noise ONLY (loud, no body sounds)

Goal:
  Find frequency content present in BOTH signals → subtract it (ambient removal)
  Then bandpass filter the clean residual into Heart or Lung mode

Method: Spectral Subtraction in 30ms chunks (matching MATLAB behaviour)
  For each chunk:
    1. Compute FFT of main chunk and anc chunk
    2. Subtract anc magnitude spectrum from main magnitude spectrum
    3. Reconstruct signal using main's phase (body sounds preserve their phase)
    4. Bandpass filter to the mode-specific range

Frequency bands:
  Heart mode : 50  – 200  Hz  (S1/S2 cardiac sounds)
  Lung mode  : 100 – 1500 Hz  (vesicular + bronchial breath sounds)
"""

import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter, lfilter_zi
import os
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# PATHS  ← paste your file paths here
# ─────────────────────────────────────────────────────────────────────────────

MAIN_PATH   = r"C:\Users\aksha\OneDrive\Documents\Vtitan\Eletro seth\motorola motorola edge 50 pro (1b991842ca3f82c3)\motorola motorola edge 50 pro (1b991842ca3f82c3)\1 Testing\Heart_file.wav"
ANC_PATH    = r"C:\Users\aksha\OneDrive\Documents\Vtitan\Eletro seth\motorola motorola edge 50 pro (1b991842ca3f82c3)\motorola motorola edge 50 pro (1b991842ca3f82c3)\1 Testing\Anc_mic.wav"       # reference mic (ambient only)
OUTPUT_DIR  = r"C:\Users\aksha\OneDrive\Documents\Vtitan\Eletro seth\Spectral subtaction\Outputs\Result"        # folder for saved results


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_MS       = 30        # Processing block size in milliseconds (matches MATLAB)
OVER_SUB_ALPHA = 1.2       # Over-subtraction factor (>1 = more aggressive removal)
FLOOR_BETA     = 0.02      # Spectral floor — prevents musical noise artefacts
OVERLAP        = 0.5       # 50% overlap between chunks for smooth reconstruction

# Frequency bands (Hz)
HEART_LOW,  HEART_HIGH  = 50,   200
LUNG_LOW,   LUNG_HIGH   = 100,  1500


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def butter_bandpass(low_hz, high_hz, fs, order=4):
    nyq = fs / 2
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype='band')
    return b, a

def butter_lowpass(cutoff_hz, fs, order=6):
    b, a = butter(order, cutoff_hz / (fs / 2), btype='low')
    return b, a

def apply_filter_stateful(b, a, signal, chunk_size):
    """
    Apply IIR filter in chunks with preserved state between blocks.
    This is the key to matching MATLAB's real-time chunk processing —
    no clicks or discontinuities at chunk boundaries.
    """
    zi = lfilter_zi(b, a) * signal[0]
    output = np.zeros_like(signal)
    for start in range(0, len(signal), chunk_size):
        end = min(start + chunk_size, len(signal))
        chunk = signal[start:end]
        filtered_chunk, zi = lfilter(b, a, chunk, zi=zi)
        output[start:end] = filtered_chunk
    return output


# ─────────────────────────────────────────────────────────────────────────────
# CORE: SPECTRAL SUBTRACTION IN 30ms CHUNKS
# ─────────────────────────────────────────────────────────────────────────────

def spectral_subtraction(main: np.ndarray, anc: np.ndarray,
                          fs: int, chunk_ms: int = 30,
                          alpha: float = 1.2, beta: float = 0.02) -> np.ndarray:
    """
    Remove ambient noise from main by subtracting the spectral magnitude
    of the ANC (reference) signal. Processing happens in chunk_ms blocks
    with 50% overlap-add for smooth output, matching MATLAB behaviour.

    Parameters
    ----------
    main      : stethoscope signal (heart + lungs + faint ambient)
    anc       : reference signal   (ambient noise only)
    fs        : sample rate
    chunk_ms  : block size in milliseconds (default 30ms)
    alpha     : over-subtraction factor (1.0–2.0). Higher = more aggressive.
    beta      : spectral floor factor (0.01–0.05). Prevents musical noise.

    Returns
    -------
    clean     : ambient-reduced signal, same length as main
    """
    chunk_size = int(fs * chunk_ms / 1000)
    hop_size   = chunk_size // 2                   # 50% overlap
    win        = np.hanning(chunk_size)             # Hanning window (matches MATLAB default)

    # Pad signals to fit exact number of chunks
    pad_len = chunk_size
    main_pad = np.concatenate([np.zeros(pad_len), main, np.zeros(pad_len)])
    anc_pad  = np.concatenate([np.zeros(pad_len), anc,  np.zeros(pad_len)])

    output    = np.zeros(len(main_pad))
    win_accum = np.zeros(len(main_pad))            # Track overlap-add normalisation

    n_chunks = (len(main_pad) - chunk_size) // hop_size + 1

    for i in range(n_chunks):
        start = i * hop_size
        end   = start + chunk_size
        if end > len(main_pad):
            break

        main_chunk = main_pad[start:end] * win
        anc_chunk  = anc_pad [start:end] * win

        # FFT of both chunks
        main_fft = np.fft.rfft(main_chunk)
        anc_fft  = np.fft.rfft(anc_chunk)

        main_mag = np.abs(main_fft)
        anc_mag  = np.abs(anc_fft)
        main_phase = np.angle(main_fft)             # Keep body sound phase

        # Spectral subtraction:
        # Clean magnitude = max(main_mag - alpha*anc_mag, beta*main_mag)
        # The beta floor prevents over-subtraction artefacts ("musical noise")
        clean_mag = np.maximum(
            main_mag - alpha * anc_mag,
            beta * main_mag
        )

        # Reconstruct from clean magnitude + original phase
        clean_fft   = clean_mag * np.exp(1j * main_phase)
        clean_chunk = np.fft.irfft(clean_fft, n=chunk_size)

        # Overlap-add
        output[start:end]    += clean_chunk * win
        win_accum[start:end] += win ** 2

    # Normalise the overlap-add
    win_accum = np.where(win_accum < 1e-8, 1.0, win_accum)
    output = output / win_accum

    # Trim padding and match original length
    clean = output[pad_len: pad_len + len(main)]
    return clean


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def process(main_path: str, anc_path: str, output_dir: str = "."):
    """
    Full pipeline:
      1. Load main + anc
      2. Spectral subtraction (ambient removal) in 30ms chunks
      3. Bandpass filter into Heart mode (50–200 Hz)
      4. Bandpass filter into Lung mode  (100–1500 Hz)
      5. Save outputs
    """
    print("=" * 55)
    print("GFANC Stethoscope — Python Replication")
    print("=" * 55)

    # Load
    fs_main, main = wav.read(main_path)
    fs_anc,  anc  = wav.read(anc_path)
    assert fs_main == fs_anc, "Sample rates must match!"
    fs = fs_main

    main = main.astype(np.float64)
    anc  = anc.astype(np.float64)

    # Trim/pad ANC to match main length
    if len(anc) > len(main):
        anc = anc[:len(main)]
    elif len(anc) < len(main):
        anc = np.pad(anc, (0, len(main) - len(anc)))

    chunk_size = int(fs * CHUNK_MS / 1000)
    print(f"  Sample rate  : {fs} Hz")
    print(f"  Duration     : {len(main)/fs:.2f} s")
    print(f"  Chunk size   : {CHUNK_MS} ms = {chunk_size} samples")
    print(f"  Alpha        : {OVER_SUB_ALPHA}  (over-subtraction)")
    print(f"  Beta         : {FLOOR_BETA}  (spectral floor)")
    print("=" * 55)

    # Step 1: Spectral subtraction
    print("\n[STEP 1/3] STARTED  - Spectral Subtraction")
    print("  Subtracting ambient noise (anc) from stethoscope (main)...")
    print("  Processing in 30ms chunks with 50% overlap-add...")
    clean = spectral_subtraction(main, anc, fs,
                                  chunk_ms=CHUNK_MS,
                                  alpha=OVER_SUB_ALPHA,
                                  beta=FLOOR_BETA)
    print("[STEP 1/3] COMPLETE - Spectral Subtraction done")

    # Step 2: Heart mode
    print(f"\n[STEP 2/3] STARTED  - Heart Mode Filter ({HEART_LOW}-{HEART_HIGH} Hz)")
    print("  Isolating S1/S2 cardiac sounds...")
    b_h, a_h = butter_bandpass(HEART_LOW, HEART_HIGH, fs, order=4)
    heart_out = apply_filter_stateful(b_h, a_h, clean, chunk_size)
    print("[STEP 2/3] COMPLETE - Heart Mode Filter done")

    # Step 3: Lung mode
    print(f"\n[STEP 3/3] STARTED  - Lung Mode Filter ({LUNG_LOW}-{LUNG_HIGH} Hz)")
    print("  Isolating vesicular and bronchial breath sounds...")
    b_l, a_l = butter_bandpass(LUNG_LOW, LUNG_HIGH, fs, order=4)
    lung_out  = apply_filter_stateful(b_l, a_l, clean, chunk_size)
    print("[STEP 3/3] COMPLETE - Lung Mode Filter done")

    # Save
    print("\n[SAVING]  STARTED  - Writing output files...")

    def save(filename, data, label):
        path = os.path.join(output_dir, filename)
        clipped = np.clip(data, -32767, 32767).astype(np.int16)
        wav.write(path, fs, clipped)
        rms      = np.sqrt(np.mean(data**2))
        duration = len(data) / fs
        print(f"  Saved  : {label}")
        print(f"  Path   : {path}")
        print(f"  Info   : {duration:.2f} s  |  RMS = {rms:.1f}")
        print()

    save("heart_mode_replicated.wav", heart_out, "heart_mode_replicated.wav")
    save("lung_mode_replicated.wav",  lung_out,  "lung_mode_replicated.wav")
    save("anc_clean_residual.wav",    clean,     "anc_clean_residual.wav  (noise-reduced, pre-filter)")

    print("[SAVING]  COMPLETE - All files written")

    print("\n[SPECTROGRAM] STARTED  - Generating spectrogram plots...")
    plt.style.use('dark_background')
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.ticker as ticker
    
    # Custom Audacity-style colormap
    audacity_colors = [
        (0.00, '#000000'), 
        (0.20, '#000044'), # dark blue
        (0.40, '#550055'), # purple
        (0.60, '#cc0022'), # crimson
        (0.80, '#ff7700'), # orange
        (0.95, '#ffff00'), # yellow
        (1.00, '#ffffff')  # white
    ]
    audacity_cmap = LinearSegmentedColormap.from_list("audacity", audacity_colors)

    fig, axes = plt.subplots(5, 1, figsize=(14, 22), sharex=True)
    fig.patch.set_facecolor('#0f0f0f')
    
    signals = [
        ("Main Mic", main),
        ("ANC Mic", anc),
        ("Clean Residual (pre-filter)", clean),
        ("Heart Mode Output", heart_out),
        ("Lung Mode Output", lung_out)
    ]
    
    for ax, (title, sig) in zip(axes, signals):
        ax.set_facecolor('#0a0a0a')
        # specgram automatically plots in dB
        Pxx, freqs, bins, im = ax.specgram(sig, NFFT=1024, Fs=fs, noverlap=512, cmap=audacity_cmap)
        ax.set_title(f"Spectrogram \u2014 {title}", color='white', fontweight='bold', pad=15)
        ax.set_ylabel('Frequency (Hz)', color='lightgray')
        ax.tick_params(axis='x', colors='lightgray')
        ax.tick_params(axis='y', colors='lightgray')
        
        nyq = fs / 2
        # Use a logarithmic/Mel-style Y-axis to match Audacity
        ax.set_yscale('symlog', linthresh=100)
        ax.set_ylim(50, min(8000, nyq))
        
        # Set explicit ticks to match Audacity
        ticks = [100, 500, 1000, 2000, 4000]
        if nyq >= 7000: ticks.append(7000)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        
        # Add colorbar
        cbar = fig.colorbar(im, ax=ax, pad=0.01, aspect=30)
        cbar.set_label('Power (dB, rel. peak)', color='lightgray')
        cbar.ax.tick_params(colors='lightgray')
        
        # Frequency Bands Markers
        t_max = len(sig) / fs
        x_text = t_max * 0.99
        
        # Heart Band
        ax.axhline(50, color='c', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(200, color='c', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.text(x_text, 125, 'Heart 50-200Hz', color='c', ha='right', va='center', fontsize=9, alpha=0.9)
        
        # Artifact Band
        ax.axhline(580, color='tomato', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(720, color='tomato', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.text(x_text, 650, 'Artifact 580-720Hz', color='tomato', ha='right', va='center', fontsize=9, alpha=0.9)
        
        # Lung Band
        ax.axhline(100, color='lime', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axhline(1500, color='lime', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.text(x_text, 1400, 'Lung 100-1500Hz', color='lime', ha='right', va='center', fontsize=9, alpha=0.9)

    axes[-1].set_xlabel('Time (s)', color='lightgray')
    plt.tight_layout()
    
    spec_path = os.path.join(output_dir, "spectrogram_comparison_output.png")
    plt.savefig(spec_path, dpi=150)
    plt.close()
    print(f"  Saved  : {spec_path}")
    print("[SPECTROGRAM] COMPLETE - Spectrogram image saved")

    print("\n" + "=" * 55)
    print("  PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Output folder : {output_dir}")
    print(f"  |-- heart_mode_replicated.wav")
    print(f"  |-- lung_mode_replicated.wav")
    print(f"  |-- anc_clean_residual.wav")
    print(f"  `-- spectrogram_comparison_output.png")
    print("=" * 55)

    return heart_out, lung_out, clean, fs


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

heart_out, lung_out, clean, fs = process(MAIN_PATH, ANC_PATH, OUTPUT_DIR)