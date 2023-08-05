% Speech Processing with MFCC using Fourier Transform

% Read the input speech signal from a WAV file
filename = 'input.wav';
[x, Fs] = audioread(filename);

% Pre-emphasis to emphasize the high-frequency components
pre_emphasis = 0.97;
x_preemphasized = filter([1, -pre_emphasis], 1, x);

% Frame the pre-emphasized signal into overlapping frames
frame_length = 0.025; % 25ms frame length
frame_overlap = 0.01; % 10ms overlap
frame_size = round(Fs * frame_length);
frame_shift = round(Fs * (frame_length - frame_overlap));
num_frames = floor((length(x_preemphasized) - frame_size) / frame_shift) + 1;

% Initialize an empty matrix to store the frames
frames = zeros(frame_size, num_frames);

% Extract overlapping frames from the pre-emphasized signal
for i = 1:num_frames
    start_index = (i - 1) * frame_shift + 1;
    end_index = start_index + frame_size - 1;
    frames(:, i) = x_preemphasized(start_index:end_index);
end


% Apply windowing to each frame (Hamming window)
window = hamming(frame_size);
windowed_frames = frames .* window;

% Compute the magnitude spectrum using the Fourier Transform
mag_spectrum_frames = abs(fft(windowed_frames));

% Compute the Mel filterbank energies
num_filters = 20;
mel_filterbank = melFilterBank(Fs, frame_size, num_filters);
mel_energies = mel_filterbank * mag_spectrum_frames;

% Take the logarithm of the Mel filterbank energies
log_mel_energies = log(mel_energies);

% Compute the Discrete Cosine Transform (DCT) of the log Mel energies
num_coefficients = 12;
mfcc = dct(log_mel_energies(2:num_filters, :), num_coefficients);

% Display the MFCC coefficients
disp('MFCC Coefficients:');
disp(mfcc);

% Helper function to create Mel filterbank
function filterbank = melFilterBank(Fs, frame_size, num_filters)
    % Calculate the Mel frequency scale range
    f_min = 0;
    f_max = Fs / 2;
    mel_f_min = 1125 * log(1 + f_min / 700);
    mel_f_max = 1125 * log(1 + f_max / 700);
    mel_points = linspace(mel_f_min, mel_f_max, num_filters + 2);
    hz_points = 700 * (exp(mel_points / 1125) - 1);
    filterbank = zeros(num_filters, frame_size / 2 + 1);

    for i = 1:num_filters
        filter_start = floor(hz_points(i) / Fs * (frame_size + 1));
        filter_mid = floor(hz_points(i + 1) / Fs * (frame_size + 1));
        filter_end = floor(hz_points(i + 2) / Fs * (frame_size + 1));

        for j = filter_start:filter_mid
            filterbank(i, j) = (j - filter_start) / (filter_mid - filter_start);
        end

        for j = filter_mid:filter_end
            filterbank(i, j) = 1 - (j - filter_mid) / (filter_end - filter_mid);
        end
    end

    % Normalize the filterbank
    filterbank = filterbank ./ sum(filterbank, 2);
end