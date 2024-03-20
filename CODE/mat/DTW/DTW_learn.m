close all
clc
clear all
% Load two speech waveforms of the same utterance (from TIMIT)
[d1,sr] = audioread('sm1_cln.wav');
[d2,sr] = audioread('sm2_cln.wav');

% Listen to them together:
ml = min(length(d1),length(d2));
soundsc(d1(1:ml)+d2(1:ml),sr)
% or, in stereo
soundsc([d1(1:ml),d2(1:ml)],sr)

% Calculate STFT features for both sounds (25% window overlap)
D1 = specgram(d1,512,sr,512,384);
D2 = specgram(d2,512,sr,512,384);

% Construct the 'local match' scores matrix as the cosine distance
% between the STFT magnitudes
% similar frames have a value close to 1 and dissimilar frames
% have a value that can approach -1

SM = simmx(abs(D1),abs(D2));
% Look at it:
subplot(121)
imagesc(SM)
colormap(1-gray)
% You can see a dark stripe (high similarity values) approximately
% down the leading diagonal.

% Use dynamic programming to find the lowest-cost path between the
% opposite corners of the cost matrix
% Note that we use 1-SM because dp will find the *lowest* total cost
[p,q,C] = dp(1-SM);
% Overlay the path on the local similarity matrix
hold on; plot(q,p,'r'); hold off
% Path visibly follows the dark stripe

% Plot the minimum-cost-to-this point matrix too
subplot(122)
imagesc(C)
hold on; plot(q,p,'r'); hold off
% Bottom right corner of C gives cost of minimum-cost alignment of the two
C(size(C,1),size(C,2))
% ans =128.2873
% This is the value we would compare between different
% templates if we were doing classification.

% Calculate the frames in D2 that are indicated to match each frame
% in D1, so we can resynthesize a warped, aligned version
D2i1 = zeros(1, size(D1,2));
for i = 1:length(D2i1)
    D2i1(i) = q(min(find(p >= i)));
end
% Phase-vocoder interpolate D2's STFT under the time warp
D2x = pvsample(D2, D2i1-1, 128);
% Invert it back to time domain
d2x = istft(D2x, 512, 512, 128);

% Listen to the results
% Warped version alone
soundsc(d2x,sr)
% Warped version added to original target (have to fine-tune length)
d2x = Resize(d2x', length(d1),1);
soundsc(d1+d2x,sr)
% .. and in stereo
soundsc([d1,d2x],sr)
% Compare to unwarped pair:
soundsc([d1(1:ml),d2(1:ml)],sr)



%%
clear all
clc

load mtlb

% To hear, type soundsc(mtlb,Fs)
% Extract the two segments that correspond to the two instances of the /æ/ phoneme. 
% The first one occurs roughly between 150 ms and 250 ms,
% and the second one between 370 ms and 450 ms. 
% Plot the two waveforms.

a1 = mtlb(round(0.15*Fs):round(0.25*Fs));
a2 = mtlb(round(0.37*Fs):round(0.45*Fs));
figure
subplot(2,1,1)
plot((0:numel(a1)-1)/Fs+0.15,a1)
title('a_1')
subplot(2,1,2)
plot((0:numel(a2)-1)/Fs+0.37,a2)
title('a_2')
xlabel('Time (seconds)')
% To hear, type soundsc(a1,Fs), pause(1), soundsc(a2,Fs)

% Warp the time axes so that the Euclidean distance between the signals is minimized.
% Compute the shared "duration" of the warped signals and plot them.

[d,i1,i2] = dtw(a1,a2);

% figure
% imagesc(d)
% hold on; plot(i2,i1,'r'); hold off

a1w = a1(i1);
a2w = a2(i2);

t = (0:numel(i1)-1)/Fs;
duration = t(end)

figure
subplot(2,1,1)
plot(t,a1w)
title('a_1, Warped')
subplot(2,1,2)
plot(t,a2w)
title('a_2, Warped')
xlabel('Time (seconds)')

% To hear, type soundsc(a1w,Fs), pause(1), sound(a2w,Fs)    
% Repeat the experiment with a complete word. Load a file containing the
% word "strong," spoken by a woman and by a man. 
% The signals are sampled at 8 kHz.

load('strong.mat')

% To hear, type soundsc(her,fs), pause(2), soundsc(him,fs)
% Warp the time axes so that the absolute distance between the signals is minimized.
% Plot the original and transformed signals.
% Compute their shared warped "duration."

dtw(her,him,'absolute');
legend('her','him')

[d,iher,ihim] = dtw(her,him,'absolute');
duration = numel(iher)/Fs
% To hear, type soundsc(her(iher),fs), pause(2), soundsc(him(ihim),fs)

%%
clear all
clc


[Signal,Fs]=audioread('sample.wav');
Signal = Signal(:,2);

a1 = Signal(round(0.15*Fs):round(0.45*Fs));
a2 = Signal(round(0.3*Fs):round(0.65*Fs));

% soundsc(s,Fs);
alpha=0.98;
WindowLength=10*10^-3;  % 30 [mS] window
Overlap=50;             % 50% overlap
[ProcessedSig_a1,FramedSig_a1] = PreProcess(a1,Fs,alpha,WindowLength,Overlap);
[ProcessedSig_a2,FramedSig_a2] = PreProcess(a2,Fs,alpha,WindowLength,Overlap);
[d,i1,i2] = dtw(a1,a2,'absolute');

% figure
% imagesc(d)
% hold on; plot(i2,i1,'r'); hold off

a1w = a1(i1);
a2w = a2(i2);

t = (0:numel(i1)-1)/Fs;
duration = t(end)

figure
subplot(2,1,1)
plot(t,a1w)
title('a_1, Warped')
subplot(2,1,2)
plot(t,a2w)
title('a_2, Warped')
xlabel('Time (seconds)')

