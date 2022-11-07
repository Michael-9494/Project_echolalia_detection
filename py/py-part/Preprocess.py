# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 12:56:39 2022

@author: 97254
"""
import numpy as np
from matplotlib import patches
from numba import jit
import scipy
import librosa
import matplotlib.pyplot as plt

def enframe(x, overlap, WindowLength, Fs):
        
        
    # d_frames_r = enframe(cluster_frameR, 50, 1, fps)
    # one_d_frames_r = d_frames_r.reshape(1,-1)
    # ZeroCrossingSignal_R_clust = calcZCR(d_frames_r)
    # one_d_frames_b = enframe(frames_b[:, 100, 100], 50, 1, fps)
    # ZeroCrossingSignal_B = calcZCR(one_d_frames_b)
    # one_d_frames_g = enframe(frames_g[:, 100, 100], 50, 1, fps)
    # ZeroCrossingSignal_G = calcZCR(one_d_frames_g)
    
    # split signal up into (overlapping) frames: one per row.
    nx = len(x[:]);  # the signal length
    N = WindowLength * Fs  # [sec]*[sample/sec]=[sample]. samples in each row
    inc = ((overlap) * N) // 100
    nf = int((nx - N + inc) / inc);  # the number of segmants    
    
    f = np.zeros((int(nf), int(N)), 'int32')  # the segmants matrix
    indf = ((inc * (np.arange(nf))).T).astype(int)  # the initial index of the segment
    inds = (np.arange(N)).astype(int);  # numbers between 1-the segment length
    e_row = np.repeat(indf[:,None],int(N),axis = 1) 
    e_col = np.repeat(inds[None,:],int(nf),axis = 0)
    ind_mat = e_row + e_col

    f = x[ind_mat] #*np.hamming(N)
         
    return f


def first_PreProcess(Signal,Fs,alpha,WindowLength,Overlap):
    
    """
     first_PreProcess make all the pre processing wotk we need to do
     the steps are based on DC removal, HP filtering , framing and multiply
     each segment with hanning filter
     
    Args:
        Signal - the raw speech signal; Fs- Sampling frequency
        alpha - pre-emphasis filter parameter;
        WindowLength - window length [seconds];
        Overlap - percentage of overlap between adjacent frames [0-100]
        


    Returns:
        ProcessedSig -the preprocessed speech signal
        FramedSig -a matrix of the framed signal (each row is a frame)
        
    """
    N = WindowLength*Fs; # [sec]*[sample/sec]=[sample]
    # Remove DC noise:
    Signal = Signal - np.mean(Signal);
    Signal = Signal/np.linalg.norm(Signal);
    # Pre-Emphasis filtering  
    ProcessedSig = librosa.effects.preemphasis(Signal, coef=alpha)
    FramedSig = enframe(ProcessedSig ,((Overlap)*N)//100, WindowLength, Fs  )
    return ProcessedSig,FramedSig


    
def stft(Signal, window, Hopsize, only_positive_frequencies=False):
    """
    Compute a  short-time Fourier transform (STFT)

    Args:
        Signal (np.ndarray)- Signal to be transformed
        window (np.ndarray)- window function
        Hopsize (int) - Hopsize 
        only_positive_frequencies (bool)- Return only positive frequency part of spectrum (non-invertible)
            (Default value = False)

    Returns:
        X (np.ndarray): The discrete short-time Fourier transform
    """
    N = len(window)
    L = len(Signal)
    M = int((L - N + Hopsize) / Hopsize)  # the number of segmants 
    X = np.zeros((N, M), dtype='complex')
    for m in range(M):
        x_windowin = Signal[m * Hopsize:m * Hopsize + N] * window
        X_windowin = np.fft.fft(x_windowin)
        X[:, m] = X_windowin

    if only_positive_frequencies:
        K = 1 + N // 2
        X = X[0:K, :]
    return X

def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="time", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")

def rmse(signal, frame_size, hop_length):
    rmse = []
    
    # calculate rmse for each frame
    for i in range(0, len(signal), hop_length): 
        rmse_current_frame = np.sqrt(sum(signal[i:i+frame_size]**2) / frame_size)
        rmse.append(rmse_current_frame)
    return np.array(rmse) 

    
def spread_mel(hz_points,hz_c,hz_size,hz_max):
#  hz_points- is a discrete array of frequencies in Hz, it should be spaced
#             in mel frequencies.
#  hz_c- is defined here as the frequency index for which we are computing
#        the masking (rather than the actual frequency).
#  hz_size- defines the resolution of the output masking array
#  hz_max-  the upper frequency limit to evaluate
#        over (normally set to the Nyquist frequency).
    band=np.zeros(1, hz_size);
    hz1=hz_points(max(1,hz_c-1));                 #start
    hz2=hz_points(hz_c);                          # middle
    hz3=hz_points(min(len(hz_points),hz_c+1)); #end

    return band    

def mfcc_model(seg, N, M, Fs):
 # Do FFT of audio frame seg, map to M MFCCs
 # from 0 Hz to Fs/2 Hz, using N filterbanks
 # typical values N=26,M=12,Fs=8000,seg~20ms
m_low=0; #%mel span lower limit
m_top=f2mel(Fs/2); #mel span upper limit
mdiv=(m_top-m_low)/(N-1); #mel resolution
# %Define an array of centre frequencies

xm=m_low:mdiv:m_top;
%Convert this to Hz frequencies
xf=mel2f(xm);
%Quantise to the FFT resolution
xq = floor((length(seg)/2 + 1)*xf/(Fs/2));
%Take the FFT of the speech...
S=fft(seg);
S=abs(2*(S.*S)/length(S));
S=S(1:length(S)/2);
F=[1:length(S)]*(Fs/2)/length(S);

%Compute the mel filterbanks.m
x1=zeros(1,N);
for xi=1:N
    band=spread_mel(xf,xi,length(S),Fs/2);
    x1(xi)=sum(band.*S');
end
x=log(x1);
%Convert to MFCC using loop (could use matrix)
cc=zeros(1,M);
for xc=1:M
    cc(xc)=sqrt(2/N)*sum(x.*cos(pi*xc*([1:N]-0.5)/N));
end
end
    
