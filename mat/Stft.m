function audio_stft = Stft(audio_signal,window_function,step_length)
% stft Compute the short-time Fourier transform (STFT).
%   audio_stft = zaf.stft(audio_signal,window_function,step_length)
%
%   Inputs:
%       audio_signal: audio signal [number_samples,1]
%       window_function: window function [window_length,1]
%       step_length: step length in samples
%   Output:
%       audio_stft: audio STFT [window_length,number_frames]
%
%   Example: Compute and display the spectrogram from an audio file.
%       % Read the audio signal with its sampling frequency in Hz, and average it over its channels
%       [audio_signal,sampling_frequency] = audioread('audio_file.wav');
%       audio_signal = mean(audio_signal,2);
%
%       % Set the window duration in seconds (audio is stationary around 40 milliseconds)
%       window_duration = 0.04;
%
%       % Derive the window length in samples (use powers of 2 for faster FFT and constant overlap-add (COLA))
%       window_length = 2^nextpow2(window_duration*sampling_frequency);
%
%       % Compute the window function (periodic Hamming window for COLA)
%       window_function = hamming(window_length,'periodic');
%
%       % Set the step length in samples (half of the window length for COLA)
%       step_length = window_length/2;
%
%       % Compute the STFT
%       audio_stft = zaf.stft(audio_signal,window_function,step_length);
%
%       % Derive the magnitude spectrogram (without the DC component and the mirrored frequencies)
%       audio_spectrogram = abs(audio_stft(2:window_length/2+1,:));
%
%       % Display the spectrogram in dB, seconds, and Hz
%		number_samples = length(audio_signal);
%       xtick_step = 1;
%       ytick_step = 1000;
%       figure
%       zaf.specshow(audio_spectrogram, number_samples, sampling_frequency, xtick_step, ytick_step);
%       title('Spectrogram (dB)')

% Get the number of samples and the window length in samples
number_samples = length(audio_signal);
window_length = length(window_function);

% Derive the zero-padding length at the start and at the end of the signal to center the windows
padding_length = floor(window_length/2);

% Compute the number of time frames given the zero-padding at the start and at the end of the signal
number_times = ceil(((number_samples+2*padding_length)-window_length)/step_length)+1;

% Zero-pad the start and the end of the signal to center the windows
audio_signal = [zeros(padding_length,1);audio_signal; ...
    zeros((number_times*step_length+(window_length-step_length)-padding_length)-number_samples,1)];

% Initialize the STFT
audio_stft = zeros(window_length,number_times);

% Loop over the time frames
i = 0;
for j = 1:number_times

    % Window the signal
    audio_stft(:,j) = audio_signal(i+1:i+window_length).*window_function;
    i = i+step_length;

end

% Compute the Fourier transform of the frames using the FFT
audio_stft = fft(audio_stft);

end