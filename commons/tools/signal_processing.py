import numpy as np
import scipy.stats as st
import scipy.signal as sig

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


def gen_fx_gsmooth(data, sigma=15):

    if len(np.shape(data)) > 1:

        m, n = np.shape(data)

        if m > n:
            data = np.transpose(data)

        len_data = np.max([m, n])
        y = st.norm.pdf(np.arange(-1*sigma, 1*(sigma+1)), 0, sigma)
        y1 = np.tile(y, (len_data, 1))
        len_y1 = np.shape(y1)[1]

        for i in range(sigma + 1):
            y1[i, 0:sigma - i] = 0
            y1[(len_data - 1) - i, len_y1 - (sigma - i):len_y1] = 0

        smoothed = sig.convolve2d(data, y[:, None], 'same') / np.sum(y1, axis=1)

    else:
        len_data = len(data)
        y = st.norm.pdf(np.arange(-1 * sigma, 1 * (sigma + 1)), 0, sigma)
        y1 = np.tile(y, (len_data, 1))
        len_y1 = np.shape(y1)[1]

        for i in range(sigma + 1):
            y1[i, 0:sigma - i] = 0
            y1[(len_data - 1) - i, len_y1 - (sigma - i):len_y1] = 0

        smoothed = sig.convolve(data, y, 'same') / np.sum(y1, axis=1)

    return smoothed
