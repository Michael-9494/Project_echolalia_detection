function [seg_ind,delta,D1n]=segmentation(signal,winlen,eta,dt,Fs,Idx)
% This function returns the index for the beginning of each segment and the
% spectral error measure delta_1n

Signal = signal(Idx(1):Idx(2)); %relevant signal
WindowLength = winlen*Fs; %in samples
Dt = dt*Fs; %in samples

ref = Signal(1:WindowLength);
test = Signal(1:WindowLength);
delta = zeros(1,length(Signal)-WindowLength);
D1n = zeros(1,length(Signal)-WindowLength);
flag = 0;
seg_ind = zeros(1,length(Signal));
j = 1;

for i = 1:length(Signal)-WindowLength
    delta(i) = 2*pi*sum((abs(fft(xcorr(test)))-abs(fft(xcorr(Signal(1:WindowLength))))).^2)/(sum(abs(fft(xcorr(test))))*sum(abs(fft(xcorr(Signal(1:WindowLength))))));
    D1n(i) = 2*pi*sum((abs(fft(xcorr(test)))-abs(fft(xcorr(ref)))).^2)/(sum(abs(fft(xcorr(test))))*sum(abs(fft(xcorr(ref)))));
    if D1n(i) > eta
        if i-flag > Dt
            ref = test;
            flag = i;
            seg_ind(j) = i;
            j = j+1;
        end
    end
    test = Signal(i+1:i+WindowLength);
end

% seg_ind to be according to original signal indices
seg_ind = nonzeros(seg_ind);
seg_ind = seg_ind+Idx(1);

% make delta the same length as signal - zero padding
before = zeros(1,Idx(1));
after = zeros(1,WindowLength+length(signal)-Idx(2)-1);
delta = cat(2,before,delta);
delta = cat(2,delta,after);
%{
% removing extra parts
extra = 0;
for i=1:length(seg_ind)-1
    len_seg = seg_ind(i+1) - seg_ind(i);
    if len_seg < 550
        seg_ind(i:end-1) = seg_ind(i+1:end);
        extra = extra+1;
    end
end
seg_ind = seg_ind(1:end-extra);
%}
end

