import numpy as np

# === === === === === === === === === === === === === === === === === === === === === === === === === == #Filters And Functions# === === === === === === === === === === === === === === === === === === === === === === === === === ==

#A wave deffination
wave_sine = lambda amp, freq, phase, time: amp * np.sin(2 * np.pi * freq * time + phase)

# complete morlet wavelet
morlet = lambda time, wavelet_frequency_spread, cenfreq: np.exp(2 * 1j * np.pi * time) * \
  np.exp(-np.power(time, 2) / 2 * np.power(
    (wavelet_frequency_spread / 2 * np.pi * cenfreq),
    2)) / cenfreq

# stable characteristic function


def stable_characteristic_function(t, alpha, beta, c, mu):
    if alpha != 1:
        PHI = np.tan(np.pi * alpha / 2)
    else:
        PHI = -2 / np.pi * np.log(abs(t))

    stable_charact = np.exp(1j * t * mu - np.power(abs(c * t), alpha) * (1 - beta * np.sign(t) * PHI))
    return stable_charact

# Stable morlet


def stable_morlet(time, alpha, beta, c, mu):
    heavy_wavelet = np.exp(2 * 1j * np.pi * time) * stable_characteristic_function(time, alpha, beta, c, mu)
    return heavy_wavelet


def generalized_wavelet(signal, filter_sig):
    filtered_signal = np.convolve(signal, filter_sig, 'same')
    return filtered_signal

