
clear;close all;clc

Data_Folder = "Y:\Echolalia_proj_Michael\DATA\New folder";
% cd(Data_Folder);

Autism_data = dir(Data_Folder);

proj ="C:\Users\97254\Documents\GitHub\Project-speech\Project_echolalia_detection\mat" ;
ADOS_table = readtable(proj+"\New folder\675830557_170820_new.xlsx");

[Signal,Fs]=audioread(proj+"\New folder\675830557_170820.wav");

ADOS_mat = table2array(ADOS_table(:,1:2));

% find Therapist occurrences of echolalia from the data
EcholaliaEventTherapistStart = ADOS_mat( strcmp({'Echolalia'},...
    ADOS_table.Var4) & strcmp({'Therapist'}, ADOS_table.Var3) ,1);
EcholaliaEventTherapistEnd =ADOS_mat( strcmp({'Echolalia'},...
    ADOS_table.Var4) & strcmp({'Therapist'}, ADOS_table.Var3) ,2);
% find Child occurrences of echolalia from the data
EcholaliaEventChildStart =ADOS_mat( strcmp({'Echolalia'}, ADOS_table.Var4)...
    & strcmp({'Child'}, ADOS_table.Var3) ,1);
EcholaliaEventChildEnd =ADOS_mat( strcmp({'Echolalia'}, ADOS_table.Var4)...
    & strcmp({'Child'}, ADOS_table.Var3) ,2);

% lets look at the first occurrence of echolalia at the therapist
Signal_therapist = Signal(EcholaliaEventTherapistStart(1)*Fs:EcholaliaEventTherapistEnd(1)*Fs);
Signal_child = Signal(EcholaliaEventChildStart(1)*Fs:EcholaliaEventChildEnd(1)*Fs);

T = 1/Fs;             % Sampling period
Len_thera = length(Signal_therapist);
Len_child = length(Signal_child);

tTherapist = EcholaliaEventTherapistStart(1)+(0:Len_thera-1)*T;% Time vector
tChild = EcholaliaEventChildStart(1)+(0:Len_child-1)*T;

figure,subplot(2,1,1)
plot(tTherapist,Signal_therapist)
title('speech signal therapist');xlabel('time[s]');ylabel('amp')
grid on; axis tight
hold on
subplot(2,1,2)
plot(tChild,Signal_child)
title('speech signal child');xlabel('time[s]');ylabel('amp')
grid on; axis tight

% soundsc(Signal_therapist,Fs);
% soundsc(Signal_child,Fs);

% % Compute the Fourier transform of the signal.
% Y = fft(Signal_therapist);
% % Compute the two-sided spectrum P2. Then compute the single-sided spectrum
% % P1 based on P2 and the even-valued signal length L.
% P2 = abs(Y/Len_thera);P1 = P2(1:Len_thera/2+1);P1(2:end-1) = 2*P1(2:end-1);
% % Define the frequency domain f and plot the single-sided amplitude spectrum P1.
% % The amplitudes are not exactly at 0.7 and 1, as expected, because of the
% % added noise. On average, longer signals produce better frequency approximations.
% f = Fs*(0:(Len_thera/2))/Len_thera;
% figure
% plot(f,P1)
% title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')
% %



% numHops = floor((r-winLength)/hopLength) + 1
alpha=0.98;
WindowLength=30*10^-3;  % 30 [mS] window
WindowLenSamp=WindowLength*Fs;
Overlap=75;             % 50% overlap
fftLength = 2^nextpow2(WindowLenSamp);
noverlap=20*10^(-3)*Fs;

[ProcessedSig_therapist,FramedSig_therapist] = PreProcess(Signal_therapist,Fs,alpha,WindowLength,Overlap);
[ProcessedSig_child,FramedSig_child] = PreProcess(Signal_child,Fs,alpha,WindowLength,Overlap);

% Convert the audio signal to a frequency-domain representation using 30 ms
% windows with 15 ms overlap. Because the input is real and therefore the
% spectrum is symmetric, you can use just one side of the frequency domain
% representation without any loss of information. Convert the complex
% spectrum to the magnitude spectrum: phase information is discarded
% when calculating mel frequency cepstral coefficients (MFCC).
[S,F,t] = stft(Signal_therapist,Fs, ...
    "Window",hamming(WindowLenSamp,"periodic"), ...
    "OverlapLength",noverlap, ...
    "FrequencyRange","onesided");
PowerSpectrum = S.*conj(S);

NumBands = 13;
range = [0,Fs/2];

[Filter_Bank,center_Frequencies,MF,BW] = Mel_Filter_bank(range,WindowLenSamp,Fs,NumBands);

figure,plot(F,Filter_Bank.'),grid on;
title("Mel Filter Bank- my implementation"),xlabel("Frequency (Hz)");




%% add filterbank monipulations here!!!!


% Discrete cosine transform matrix..
[m,k] = meshgrid(0:NumBands-1);
m = m+1;   % m [1...M=numBands]

lamba_m = (2*m-1)/(2*NumBands);

DCT_mat_lam = sqrt(2 / NumBands) * cos(pi * th_p_of_Lamda1(alpha(1),lamba_m).* k );
DCT_mat_lam(1,:) = DCT_mat_lam(1,:) / sqrt(2);
% round(DCT_mat_lam*DCT_mat_lam')

alpha = 0.88: 0.02: 1.22;
% warping the center frequency.
theta_p_of_lamda = zeros(NumBands);
for j =1:NumBands
    for i = 1:NumBands
        theta_p_of_lamda(i,j) = th_p_of_Lamda(alpha(j),center_Frequencies(i),Fs);
    end
end



DCT_mat = sqrt(2 / NumBands) * cos(pi * (2*m - 1) .* k / (2 * NumBands));
DCT_mat(1,:) = DCT_mat(1,:) / sqrt(2);

% round(DCT_mat*DCT_mat') -> unitary!!
inv_DCT_mat = DCT_mat';


% To apply frequency domain filtering, perform a matrix multiplication of
% the filter bank and the power spectrogram.
melSpectrogram = Filter_Bank*PowerSpectrum;

% Visualize the power-per-band in dB.
melSpectrogramdB_cepstrum = 10*log10(melSpectrogram);



figure
surf(t,center_Frequencies,melSpectrogramdB_cepstrum,"EdgeColor","none");
view([0,90])
axis([t(1) t(end) center_Frequencies(1) center_Frequencies(end)])
xlabel('Time (s)')
ylabel('Frequency (Hz)')
c = colorbar;
c.Label.String = 'Power (dB)';

ccc = cepstralCoefficients(melSpectrogram,'NumCoeffs',15);
% by default the equasion is:
MFCC_features = (DCT_mat*log10(melSpectrogram))';

Log_Mel_spectrum = (inv_DCT_mat*MFCC_features')';


% [coeffs1,delta1,deltaDelta1,loc1] = mfcc(ProcessedSig_therapist,Fs,...
%     "OverlapLength",noverlap);
% figure
% mfcc(ProcessedSig_therapist,Fs,"LogEnergy","Ignore",...
%         "OverlapLength",noverlap);
%
% [coeffs,delta,deltaDelta,loc] = mfcc(S,Fs,...
%         "OverlapLength",noverlap);
% figure
% mfcc(S,Fs,"LogEnergy","Ignore",...
%         "OverlapLength",noverlap);

%%



%
%
%
%
%
% % Example: Compute and display the mel filterbank.
% % Compute the mel filterbank using some parameters
% number_mels = 128;
% mel_filterbank = melfilterbank(Fs,WindowLength*Fs,number_mels);
%
% % Display the mel filterbank
% figure
% imagesc(mel_filterbank)
% axis xy
% colormap(jet)
% title('Mel filterbank')
% xlabel('Frequency index')
% ylabel('Mel index')
%
% %    Example: Compute and display the MFCCs, delta MFCCs, and delta-delta MFCCs.
% % Read the audio signal with its sampling frequency in Hz, and average it over its channels
%
% % Set the parameters for the Fourier analysis
% window_length = 2^nextpow2(0.04*Fs);
% window_function = hamming(window_length,'periodic');
% step_length = window_length/2;
%
% % Compute the mel filterbank
% number_mels = 40;
% mel_filterbank = melfilterbank(Fs,window_length,number_mels);
%
% % Compute the MFCCs using the filterbank
% number_coefficients = 20;
% audio_mfcc = MFCC(Signal_therapist,window_function,step_length,mel_filterbank,number_coefficients);
%
% % Compute the delta and delta-delta MFCCs
% audio_dmfcc = diff(audio_mfcc,1,2);
% audio_ddmfcc = diff(audio_dmfcc,1,2);
%
% % Compute the time resolution for the MFCCs in number of time frames per second (~ sampling frequency for the MFCCs)
% time_resolution = Fs*size(audio_mfcc,2)/length(Signal_therapist);
%
% % Display the MFCCs, delta MFCCs, and delta-delta MFCCs in seconds
% xtick_step = 1;
% number_samples = length(Signal_therapist);
% figure
% subplot(3,1,1), mfccshow(audio_mfcc,number_samples,Fs,xtick_step), title('MFCCs')
% subplot(3,1,2), mfccshow(audio_dmfcc,number_samples,Fs,xtick_step), title('Delta MFCCs')
% subplot(3,1,3), mfccshow(audio_ddmfcc,number_samples,Fs,xtick_step), title('Delta-delta MFCCs')
%
%

