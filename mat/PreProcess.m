function [ProcessedSig,FramedSig]=PreProcess(Signal,Fs,alpha,WindowLength,Overlap)
% PreProcess make all the pre processing wotk we need to do
% the steps are based on DC removal, HP filtering , framing and multiply
% each segment with rectangular filter
% Signal - the raw speech signal; Fs- Sampling frequency
% alpha - pre-emphasis filter parameter;
% WindowLength - window length [seconds];
% percentage of overlap between adjacent frames [0-100]
% FUNCTION OUTPUT:
% ProcessedSig - the preprocessed speech signal;
% a matrix of the framed signal (each row is a frame)

N = WindowLength*Fs; % [sec]*[sample/sec]=[sample]
% Remove DC noise:
Signal = Signal - mean(Signal);
Signal = Signal/norm(Signal);
% Pre-Emphasis filtering  
ProcessedSig = filter([1 -alpha],1,Signal);%Signal;filtfilt
% Apply the de-emphasis filter
% ProcessedSig=filter(1, [1 -alpha], ProcessedSig);
FramedSig = enframe(ProcessedSig ,hamming(N,"periodic"), ((Overlap)*N)/100 );

end

