import numpy as np
import wave

# Read the input speech signal from a WAV file
filename = 'input.wav'
with wave.open(filename, 'rb') as wavefile:
    sample_rate = wavefile.getframerate()
    num_samples = wavefile.getnframes()
    signal = np.frombuffer(wavefile.readframes(num_samples), dtype=np.int16)

# Pre-emphasis to emphasize the high-frequency components
pre_emphasis = 0.97
emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

# Frame the pre-emphasized signal into overlapping frames
frame_length = 0.025  # 25ms frame length
frame_overlap = 0.01   # 10ms overlap
frame_size = int(round(sample_rate * frame_length))
frame_shift = int(round(sample_rate * (frame_length - frame_overlap)))
num_frames = int(np.floor((len(emphasized_signal) - frame_size) / frame_shift) + 1)

# Initialize an empty matrix to store the frames
frames = np.zeros((num_frames, frame_size))

# Extract overlapping frames from the pre-emphasized signal
for i in range(num_frames):
    start_index = i * frame_shift
    end_index = start_index + frame_size
    frames[i] = emphasized_signal[start_index:end_index]

# Apply windowing to each frame (Hamming window)
window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_size) / (frame_size - 1))
windowed_frames = frames * window

# Compute the magnitude spectrum using the Fourier Transform
mag_spectrum_frames = np.abs(np.fft.fft(windowed_frames))

# Compute the Mel filterbank energies
num_filters = 20

def mel_filter_bank(sample_rate, frame_size, num_filters):
    f_min = 0
    f_max = sample_rate / 2
    mel_points = np.linspace(1125 * np.log(1 + f_min / 700), 1125 * np.log(1 + f_max / 700), num_filters + 2)
    hz_points = 700 * (np.exp(mel_points / 1125) - 1)
    filterbank = np.zeros((num_filters, frame_size // 2 + 1))

    for i in range(num_filters):
        filter_start = int(np.floor(hz_points[i] / sample_rate * (frame_size + 1)))
        filter_mid = int(np.floor(hz_points[i + 1] / sample_rate * (frame_size + 1)))
        filter_end = int(np.floor(hz_points[i + 2] / sample_rate * (frame_size + 1)))

        for j in range(filter_start, filter_mid):
            filterbank[i, j] = (j - filter_start) / (filter_mid - filter_start)

        for j in range(filter_mid, filter_end):
            filterbank[i, j] = 1 - (j - filter_mid) / (filter_end - filter_mid)

    # Normalize the filterbank
    filterbank /= filterbank.sum(axis=1, keepdims=True)
    return filterbank

mel_filterbank = mel_filter_bank(sample_rate, frame_size, num_filters)
mel_energies = np.dot(mag_spectrum_frames, mel_filterbank.T)

# Take the logarithm of the Mel filterbank energies
log_mel_energies = np.log(mel_energies)

# Compute the Discrete Cosine Transform (DCT) of the log Mel energies
num_coefficients = 12
mfcc = np.zeros((num_frames, num_coefficients))
for i in range(num_frames):
    mfcc[i] = dct(log_mel_energies[i, 1:num_filters], type=2, norm='ortho')[:num_coefficients]

# Display the MFCC coefficients
print("MFCC Coefficients:")
print(mfcc)