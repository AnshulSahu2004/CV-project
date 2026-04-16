from scipy.signal import butter, lfilter
import numpy as np


def bandpass_filter(signal, low=0.6, high=4.0, fs=30):

    nyquist = 0.5 * fs
    low /= nyquist
    high /= nyquist

    b, a = butter(1, [low, high], btype='band')
    return lfilter(b, a, signal)


def extract_rppg(frames):

    rgb_means = []

    for frame in frames:

        R = np.mean(frame[:,:,2])
        G = np.mean(frame[:,:,1])
        B = np.mean(frame[:,:,0])

        rgb_means.append([R,G,B])

    rgb_means = np.array(rgb_means)

    rgb_means = rgb_means / np.mean(rgb_means, axis=0)

    R = rgb_means[:,0]
    G = rgb_means[:,1]
    B = rgb_means[:,2]
    
    S1 = G - B
    S2 = G + B - 2 * R

    alpha = np.std(S1) / (np.std(S2) + 1e-6)

    pulse = S1 + alpha * S2

    pulse = bandpass_filter(pulse)

    pulse = pulse - np.mean(pulse)
    pulse = pulse / (np.std(pulse) + 1e-6)

    return pulse