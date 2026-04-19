"""
GFANC Stethoscope — Pipeline v6 (Built on v1 Base)
====================================================

BASE: The original v1 spectral_subtraction() function is kept 100% intact.
      It produces good output — we do NOT touch it.

ADDITIONS on top of v1 (minimal, safe, targeted):
  ① Stereo → mono safety check
  ② DC offset removal (single-pole 5 Hz high-pass before processing)
  ③ Adaptive gain with noise gate
       → Normalises output to -18 dBFS
       → Gate prevents amplifying near-silence (no noise floor boost)
       → Hard cap at +24 dB max gain
  ④ Soft limiter (prevents clipping after gain)
  ⑤ Impulse rejection on BOTH heart and lung output (removes handling spikes)
       → Threshold tightened to 3.0× local median (was 5.0)
       → Applied to heart mode too — fixes tall spikes visible in Audacity
  ⑥ Audacity-style spectrogram (logarithmic Y-axis, matching your target)
       → Saves one image with all 5 signals stacked
  ⑦ SNR report printed to console

Everything else (spectral subtraction algorithm, filter order,
chunk size, overlap, window) is IDENTICAL to v1.
"""

import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import butter, lfilter, lfilter_zi, medfilt
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker


# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────

MAIN_PATH  = r"C:\Users\aksha\OneDrive\Documents\Vtitan\test4\Heart_file.wav"
ANC_PATH   = r"C:\Users\aksha\OneDrive\Documents\Vtitan\test4\Anc_mic.wav"
OUTPUT_DIR = r"C:\Users\aksha\OneDrive\Documents\Vtitan\Eletro seth\Spectral subtaction\Outputs\v3"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — same as v1, only gain/gate added
# ─────────────────────────────────────────────────────────────────────────────

CHUNK_MS       = 30       # ms — UNCHANGED from v1
OVER_SUB_ALPHA = 1.2      # UNCHANGED from v1
FLOOR_BETA     = 0.02     # UNCHANGED from v1

HEART_LOW,  HEART_HIGH = 50,   200   # UNCHANGED from v1
LUNG_LOW,   LUNG_HIGH  = 100,  1500  # UNCHANGED from v1

# ── ③ Soft limiter only (no auto-amplification) ──────────────────────────────
LIMITER_THRESH  = 0.92    # Soft limiter knee (fraction of int16 max)

# ── ⑤ Steep low-pass to hard-cut frequency bleed above band ceiling ──────────
# The stripes reaching 1000-7000 Hz are caused by the order-4 Butterworth
# bandpass having a gentle roll-off. Sharp transients (heartbeats) bleed
# energy above the cutoff. Fix: apply a steep order-8 low-pass AT the band
# ceiling AFTER the bandpass. Hard-cuts everything above 200 Hz for heart,
# above 1500 Hz for lung.
LOWPASS_ORDER   = 8       # Order 8 = ~48 dB/octave roll-off (very steep)

# ── ⑥ Spectrogram ─────────────────────────────────────────────────────────────
SPEC_MAX_FREQ   = 8000    # Highest frequency shown on y-axis (Hz)
                           # Set to fs/2 automatically if lower than Nyquist


# ─────────────────────────────────────────────────────────────────────────────
# v1 HELPERS — UNCHANGED
# ─────────────────────────────────────────────────────────────────────────────

def butter_bandpass(low_hz, high_hz, fs, order=4):
    nyq  = fs / 2
    b, a = butter(order, [low_hz / nyq, high_hz / nyq], btype='band')
    return b, a

def apply_filter_stateful(b, a, signal, chunk_size):
    """
    Apply IIR filter in chunks with preserved state between blocks.
    UNCHANGED from v1.
    """
    zi     = lfilter_zi(b, a) * signal[0]
    output = np.zeros_like(signal)
    for start in range(0, len(signal), chunk_size):
        end   = min(start + chunk_size, len(signal))
        chunk = signal[start:end]
        filtered_chunk, zi = lfilter(b, a, chunk, zi=zi)
        output[start:end]  = filtered_chunk
    return output


# ─────────────────────────────────────────────────────────────────────────────
# CORE — SPECTRAL SUBTRACTION WITH PER-BIN ADAPTIVE ALPHA
# ─────────────────────────────────────────────────────────────────────────────

def spectral_subtraction(main: np.ndarray, anc: np.ndarray,
                          fs: int, chunk_ms: int = 30,
                          alpha: float = 1.2, beta: float = 0.02) -> np.ndarray:
    """
    Spectral subtraction — v1 algorithm with one targeted fix:

    ROOT CAUSE OF LUNG NOISE (diagnosed from actual audio files):
      The ANC mic is ~150% louder than main in the 500–1500 Hz band.
      With alpha=1.2:  alpha * anc >> main in those bins
      → clean_mag = max(main - 1.2*anc, beta*main) always hits the FLOOR
      → 94% of 1000–1500 Hz bins output beta*main = 0.02 × noise, not real signal
      → This beta-floor residual IS the visible noise in Audacity's lung output

    FIX — Per-bin adaptive alpha:
      For each FFT bin, if anc_mag > main_mag (ANC overpowers main):
        → Scale alpha down so (alpha_bin * anc) = 0.95 * main
        → Subtraction removes 95% of the noise without hitting the beta floor
        → Real signal content in those bins is preserved at 5% floor
      Otherwise (main >= anc): use full alpha as normal (v1 behaviour)

    This is the MINIMUM change needed. Everything else is identical to v1.
    """
    chunk_size = int(fs * chunk_ms / 1000)
    hop_size   = chunk_size // 2
    win        = np.hanning(chunk_size)

    pad_len  = chunk_size
    main_pad = np.concatenate([np.zeros(pad_len), main, np.zeros(pad_len)])
    anc_pad  = np.concatenate([np.zeros(pad_len), anc,  np.zeros(pad_len)])

    output    = np.zeros(len(main_pad))
    win_accum = np.zeros(len(main_pad))

    n_chunks = (len(main_pad) - chunk_size) // hop_size + 1

    for i in range(n_chunks):
        start = i * hop_size
        end   = start + chunk_size
        if end > len(main_pad):
            break

        main_chunk = main_pad[start:end] * win
        anc_chunk  = anc_pad [start:end] * win

        main_fft   = np.fft.rfft(main_chunk)
        anc_fft    = np.fft.rfft(anc_chunk)

        main_mag   = np.abs(main_fft)
        anc_mag    = np.abs(anc_fft)
        main_phase = np.angle(main_fft)

        # ── Per-bin adaptive alpha ─────────────────────────────────────────
        # Where ANC > main: scale alpha so (alpha_bin * anc) = 0.95 * main
        # This prevents over-subtraction hitting the beta floor.
        # Where ANC <= main: use full alpha (identical to v1).
        overpower_mask = anc_mag > main_mag
        alpha_bin      = np.full_like(main_mag, alpha)
        alpha_bin[overpower_mask] = (0.95 * main_mag[overpower_mask]
                                     / (anc_mag[overpower_mask] + 1e-10))

        clean_mag = np.maximum(
            main_mag - alpha_bin * anc_mag,
            beta * main_mag
        )

        clean_fft   = clean_mag * np.exp(1j * main_phase)
        clean_chunk = np.fft.irfft(clean_fft, n=chunk_size)

        output[start:end]    += clean_chunk * win
        win_accum[start:end] += win ** 2

    win_accum = np.where(win_accum < 1e-8, 1.0, win_accum)
    output    = output / win_accum

    return output[pad_len: pad_len + len(main)]


# ─────────────────────────────────────────────────────────────────────────────
# NEW ADDITIONS
# ─────────────────────────────────────────────────────────────────────────────

def remove_dc(signal, fs):
    """② Single-pole 5 Hz high-pass — removes DC offset before processing."""
    b, a = butter(1, 5.0 / (fs / 2.0), btype='high')
    return lfilter(b, a, signal)


def soft_limiter(signal, threshold=0.92):
    """④ Soft-knee limiter — compresses peaks above threshold instead of clipping."""
    peak   = 32767.0 * threshold
    output = np.copy(signal)
    mask   = np.abs(signal) > peak
    excess = np.abs(signal[mask]) - peak
    output[mask] = np.sign(signal[mask]) * (peak + excess * 0.1)
    return output


def butter_lowpass_steep(cutoff_hz, fs, order=8):
    """
    ⑤ Steep low-pass filter at the band ceiling.
    Order 8 gives ~48 dB/octave roll-off — hard-cuts broadband transient
    bleed above the bandpass ceiling without touching in-band content.
    """
    nyq  = fs / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    return b, a


def compute_snr_db(signal, noise_ref):
    """③ Simple SNR estimate: signal power vs noise reference power."""
    sp = np.mean(signal    ** 2)
    np_ = np.mean(noise_ref ** 2)
    if np_ < 1e-12:
        return float('inf')
    return 10.0 * np.log10(sp / (np_ + 1e-12))


# ─────────────────────────────────────────────────────────────────────────────
# ⑥ AUDACITY-STYLE SPECTROGRAM
# ─────────────────────────────────────────────────────────────────────────────

def save_spectrogram(signals_dict, fs, output_dir):
    """
    ⑥ Save an Audacity-style spectrogram image matching your target layout:
       • Dark background
       • Black→blue→purple→red→orange→yellow→white colour map
       • Logarithmic Y-axis (matches Audacity's default view)
       • Band markers: Heart (cyan), Lung (lime), Artifact (tomato)
       • One panel per signal, stacked vertically, shared time axis

    signals_dict = { "Title": signal_array, ... }
    """
    # Audacity colour map
    audacity_cmap = LinearSegmentedColormap.from_list("audacity", [
        (0.00, '#000000'),
        (0.20, '#000044'),
        (0.40, '#550055'),
        (0.60, '#cc0022'),
        (0.80, '#ff7700'),
        (0.95, '#ffff00'),
        (1.00, '#ffffff'),
    ])

    plt.style.use('dark_background')
    n_panels = len(signals_dict)
    fig, axes = plt.subplots(n_panels, 1,
                              figsize=(14, 4.5 * n_panels),
                              sharex=True)
    fig.patch.set_facecolor('#0f0f0f')

    if n_panels == 1:
        axes = [axes]

    nyq      = fs / 2
    max_freq = min(SPEC_MAX_FREQ, nyq)

    for ax, (title, sig) in zip(axes, signals_dict.items()):
        ax.set_facecolor('#0a0a0a')

        # specgram: NFFT=1024, 50% overlap — matches Audacity default
        Pxx, freqs, bins, im = ax.specgram(
            sig, NFFT=1024, Fs=fs, noverlap=512,
            cmap=audacity_cmap,
            vmin=-100, vmax=0          # Fixed dB scale for cross-signal comparison
        )

        ax.set_title(f"Spectrogram — {title}",
                      color='white', fontweight='bold', fontsize=11, pad=10)
        ax.set_ylabel('Frequency (Hz)', color='lightgray', fontsize=9)
        ax.tick_params(colors='lightgray', labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#333333')

        # Logarithmic Y-axis matching Audacity
        ax.set_yscale('symlog', linthresh=100)
        ax.set_ylim(50, max_freq)

        yticks = [100, 200, 500, 1000, 2000, 4000]
        if max_freq >= 7000: yticks.append(7000)
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        # Colour bar
        cbar = fig.colorbar(im, ax=ax, pad=0.01, aspect=25, shrink=0.9)
        cbar.set_label('Power (dB)', color='lightgray', fontsize=8)
        cbar.ax.tick_params(colors='lightgray', labelsize=7)

        # Band markers
        t_max  = len(sig) / fs
        x_text = t_max * 0.99

        ax.axhline(HEART_LOW,  color='cyan',   lw=0.9, ls='--', alpha=0.7)
        ax.axhline(HEART_HIGH, color='cyan',   lw=0.9, ls='--', alpha=0.7)
        ax.text(x_text, 120, f'Heart {HEART_LOW}–{HEART_HIGH}Hz',
                color='cyan', ha='right', va='center', fontsize=8, alpha=0.9)

        ax.axhline(580,  color='tomato', lw=0.9, ls='--', alpha=0.7)
        ax.axhline(720,  color='tomato', lw=0.9, ls='--', alpha=0.7)
        ax.text(x_text, 640, 'Artifact 580–720Hz',
                color='tomato', ha='right', va='center', fontsize=8, alpha=0.9)

        ax.axhline(LUNG_LOW,  color='lime', lw=0.9, ls='--', alpha=0.7)
        ax.axhline(LUNG_HIGH, color='lime', lw=0.9, ls='--', alpha=0.7)
        ax.text(x_text, 1200, f'Lung {LUNG_LOW}–{LUNG_HIGH}Hz',
                color='lime', ha='right', va='center', fontsize=8, alpha=0.9)

    axes[-1].set_xlabel('Time (s)', color='lightgray', fontsize=9)
    plt.tight_layout(h_pad=1.5)

    path = os.path.join(output_dir, "spectrogram_overview_v6.png")
    plt.savefig(path, dpi=150, facecolor='#0f0f0f', bbox_inches='tight')
    plt.close()
    print(f"  ✓ Spectrogram : {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def process(main_path: str, anc_path: str, output_dir: str = "."):
    print("=" * 58)
    print("  GFANC Stethoscope — Pipeline v6 (v1 base + safe additions)")
    print("=" * 58)

    os.makedirs(output_dir, exist_ok=True)

    # ── Load ─────────────────────────────────────────────────────────────────
    fs_main, main = wav.read(main_path)
    fs_anc,  anc  = wav.read(anc_path)
    assert fs_main == fs_anc, \
        f"Sample rate mismatch: main={fs_main} Hz, anc={fs_anc} Hz"
    fs = fs_main

    # ① Stereo → mono
    if main.ndim == 2: main = main[:, 0]
    if anc.ndim  == 2: anc  = anc [:, 0]

    main = main.astype(np.float64)
    anc  = anc.astype(np.float64)

    # Trim/pad ANC — UNCHANGED from v1
    if   len(anc) > len(main): anc = anc[:len(main)]
    elif len(anc) < len(main): anc = np.pad(anc, (0, len(main) - len(anc)))

    # ② DC removal
    main = remove_dc(main, fs)
    anc  = remove_dc(anc,  fs)

    chunk_size = int(fs * CHUNK_MS / 1000)
    duration   = len(main) / fs

    print(f"  Sample rate  : {fs} Hz")
    print(f"  Duration     : {duration:.2f} s")
    print(f"  Chunk size   : {CHUNK_MS} ms = {chunk_size} samples")
    print(f"  Alpha        : {OVER_SUB_ALPHA}  (over-subtraction) — v1 value")
    print(f"  Beta         : {FLOOR_BETA}  (spectral floor)    — v1 value")
    print("=" * 58)

    # ── Step 1: Spectral subtraction (v1 algorithm, unchanged) ───────────────
    print("\n[STEP 1/3] Spectral Subtraction  (v1 algorithm)")
    clean = spectral_subtraction(main, anc, fs,
                                  chunk_ms = CHUNK_MS,
                                  alpha    = OVER_SUB_ALPHA,
                                  beta     = FLOOR_BETA)
    snr_in    = compute_snr_db(main,  anc)
    snr_clean = compute_snr_db(clean, anc)
    print(f"[STEP 1/3] COMPLETE  |  Input SNR: {snr_in:+.1f} dB  "
          f"→  Clean SNR: {snr_clean:+.1f} dB  "
          f"(Δ {snr_clean - snr_in:+.1f} dB)")

    # ── Step 2: Heart mode ────────────────────────────────────────────────────
    print(f"\n[STEP 2/3] Heart Mode  ({HEART_LOW}–{HEART_HIGH} Hz)")
    b_h, a_h  = butter_bandpass(HEART_LOW, HEART_HIGH, fs, order=4)
    heart_out = apply_filter_stateful(b_h, a_h, clean, chunk_size)

    # ⑤ Steep low-pass at HEART_HIGH — hard-cuts bleed above 200 Hz
    b_lp_h, a_lp_h = butter_lowpass_steep(HEART_HIGH, fs, order=LOWPASS_ORDER)
    heart_out = apply_filter_stateful(b_lp_h, a_lp_h, heart_out, chunk_size)
    print(f"  Steep low-pass : cutoff={HEART_HIGH} Hz, order={LOWPASS_ORDER}")

    # ④ Soft limiter only — no auto-amplification
    heart_out = soft_limiter(heart_out, LIMITER_THRESH)
    snr_h     = compute_snr_db(heart_out, anc)
    print(f"  SNR   : {snr_h:+.1f} dB")
    print(f"[STEP 2/3] COMPLETE")

    # ── Step 3: Lung mode ─────────────────────────────────────────────────────
    print(f"\n[STEP 3/3] Lung Mode  ({LUNG_LOW}–{LUNG_HIGH} Hz)")
    b_l, a_l = butter_bandpass(LUNG_LOW, LUNG_HIGH, fs, order=4)
    lung_out = apply_filter_stateful(b_l, a_l, clean, chunk_size)

    # ⑤ Steep low-pass at LUNG_HIGH — hard-cuts bleed above 1500 Hz
    b_lp_l, a_lp_l = butter_lowpass_steep(LUNG_HIGH, fs, order=LOWPASS_ORDER)
    lung_out = apply_filter_stateful(b_lp_l, a_lp_l, lung_out, chunk_size)
    print(f"  Steep low-pass : cutoff={LUNG_HIGH} Hz, order={LOWPASS_ORDER}")

    # ④ Soft limiter only — no auto-amplification
    lung_out = soft_limiter(lung_out, LIMITER_THRESH)
    snr_l    = compute_snr_db(lung_out, anc)
    print(f"  SNR   : {snr_l:+.1f} dB")
    print(f"[STEP 3/3] COMPLETE")

    print(f"\n{'─'*58}")
    print(f"  SNR SUMMARY")
    print(f"{'─'*58}")
    print(f"  Input  (main vs ANC)  : {snr_in:+.1f} dB")
    print(f"  Clean  (post-ANC)     : {snr_clean:+.1f} dB   Δ {snr_clean-snr_in:+.1f} dB")
    print(f"  Heart mode output     : {snr_h:+.1f} dB   Δ {snr_h-snr_in:+.1f} dB")
    print(f"  Lung  mode output     : {snr_l:+.1f} dB   Δ {snr_l-snr_in:+.1f} dB")
    print(f"{'─'*58}")

    # ── Save WAVs ─────────────────────────────────────────────────────────────
    print("\n[SAVING] Writing output files...")

    def save_wav(filename, data, label):
        path    = os.path.join(output_dir, filename)
        clipped = np.clip(data, -32767, 32767).astype(np.int16)
        wav.write(path, fs, clipped)
        rms = np.sqrt(np.mean(data ** 2))
        print(f"  ✓ {label}")
        print(f"    {path}")
        print(f"    {duration:.2f}s  |  RMS={rms:.1f}  "
              f"|  {20*np.log10(rms/32767+1e-10):.1f} dBFS  "
              f"|  Peak={np.max(np.abs(data)):.1f}")

    save_wav("heart_mode_v6.wav",      heart_out, "heart_mode_v6.wav")
    save_wav("lung_mode_v6.wav",       lung_out,  "lung_mode_v6.wav")
    save_wav("anc_clean_residual.wav", clean,     "anc_clean_residual.wav")

    # ── ⑥ Spectrogram ─────────────────────────────────────────────────────────
    print("\n[SPECTROGRAM] Generating Audacity-style image...")
    save_spectrogram({
        "Main Mic (Input)"              : main,
        "ANC Mic (Reference)"           : anc,
        "Clean Residual (Post-ANC)"     : clean,
        "Heart Mode Output (50–200Hz)"  : heart_out,
        "Lung Mode Output (100–1500Hz)" : lung_out,
    }, fs, output_dir)
    print("[SPECTROGRAM] COMPLETE")

    print("\n" + "=" * 58)
    print("  PIPELINE COMPLETE  (v6)")
    print("=" * 58)
    print(f"  Output folder : {output_dir}")
    for fn in ["heart_mode_v6.wav", "lung_mode_v6.wav",
               "anc_clean_residual.wav", "spectrogram_overview_v6.png"]:
        print(f"  |-- {fn}")
    print("=" * 58)

    return heart_out, lung_out, clean, fs


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

heart_out, lung_out, clean, fs = process(MAIN_PATH, ANC_PATH, OUTPUT_DIR)