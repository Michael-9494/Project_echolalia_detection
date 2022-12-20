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
import glob
import pandas as pd
import parselmouth 
import statistics


from parselmouth.praat import call
from scipy.stats.mstats import zscore
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


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
#
    f = x[ind_mat] *np.hamming(N)
         
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
    Signal = Signal/max(abs(Signal));
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

    
# def spread_mel(hz_points,hz_c,hz_size,hz_max):
# #  hz_points- is a discrete array of frequencies in Hz, it should be spaced
# #             in mel frequencies.
# #  hz_c- is defined here as the frequency index for which we are computing
# #        the masking (rather than the actual frequency).
# #  hz_size- defines the resolution of the output masking array
# #  hz_max-  the upper frequency limit to evaluate
# #        over (normally set to the Nyquist frequency).
#     band=np.zeros(1, hz_size);
#     hz1=hz_points(max(1,hz_c-1));                 #start
#     hz2=hz_points(hz_c);                          # middle
#     hz3=hz_points(min(len(hz_points),hz_c+1)); #end

#     return band    

# def mfcc_model(seg, N, M, Fs):
#  # Do FFT of audio frame seg, map to M MFCCs
#  # from 0 Hz to Fs/2 Hz, using N filterbanks
#  # typical values N=26,M=12,Fs=8000,seg~20ms
# m_low=0; #%mel span lower limit
# m_top=f2mel(Fs/2); #mel span upper limit
# mdiv=(m_top-m_low)/(N-1); #mel resolution
# # %Define an array of centre frequencies

# xm=m_low:mdiv:m_top;
# %Convert this to Hz frequencies
# xf=mel2f(xm);
# %Quantise to the FFT resolution
# xq = floor((length(seg)/2 + 1)*xf/(Fs/2));
# %Take the FFT of the speech...
# S=fft(seg);
# S=abs(2*(S.*S)/length(S));
# S=S(1:length(S)/2);
# F=[1:length(S)]*(Fs/2)/length(S);

# %Compute the mel filterbanks.m
# x1=zeros(1,N);
# for xi=1:N
#     band=spread_mel(xf,xi,length(S),Fs/2);
#     x1(xi)=sum(band.*S');
# end
# x=log(x1);
# %Convert to MFCC using loop (could use matrix)
# cc=zeros(1,M);
# for xc=1:M
#     cc(xc)=sqrt(2/N)*sum(x.*cos(pi*xc*([1:N]-0.5)/N));
# end
# end
    
# This function measures formants using Formant Position formula
def measureFormants(sound, wave_file, f0min,f0max):
    sound = parselmouth.Sound(sound) # read the sound
    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    
    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")

    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []
    time_f = []
    
    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        time_f.append(t)
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)
        
    
    f1_list_no_nan = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list_no_nan = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list_no_nan = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list_no_nan = [f4 for f4 in f4_list if str(f4) != 'nan']
    
    # calculate mean formants across pulses
    f1_mean = statistics.mean(f1_list_no_nan)
    f2_mean = statistics.mean(f2_list_no_nan)
    f3_mean = statistics.mean(f3_list_no_nan)
    f4_mean = statistics.mean(f4_list_no_nan)
    
    # calculate median formants across pulses, this is what is used in all subsequent calcualtions
    # you can use mean if you want, just edit the code in the boxes below to replace median with mean
    f1_median = statistics.median(f1_list_no_nan)
    f2_median = statistics.median(f2_list_no_nan)
    f3_median = statistics.median(f3_list_no_nan)
    f4_median = statistics.median(f4_list_no_nan)
    
    return f1_list, f2_list, f3_list , f4_list,f1_mean, f2_mean, f3_mean, f4_mean, f1_median, f2_median, f3_median, f4_median,time_f



# This is the function to measure source acoustics using default male parameters.

def measurePitch(voiceID, f0min, f0max, unit):
    sound = parselmouth.Sound(voiceID) # read the sound
    duration = call(sound, "Get total duration") # duration
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    return duration, meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer

# This function runs a 2-factor Principle Components Analysis (PCA) on Jitter and Shimmer
def runPCA(df):
    # z-score the Jitter and Shimmer measurements
    measures = ['localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
                'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
    x = df.loc[:, measures].values
    x = StandardScaler().fit_transform(x)
    # PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['JitterPCA', 'ShimmerPCA'])
    principalDf
    return principalDf


def draw_spectrogram(spectrogram, dynamic_range=50): 
    X, Y = spectrogram.x_grid(),spectrogram.y_grid() 
    sg_db = 10 * np.log10(spectrogram.values) 
    plt.pcolormesh(X, Y, sg_db, vmin=sg_db.max()- dynamic_range)
    plt.ylim([spectrogram.ymin, spectrogram.ymax]) 
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")
    
def draw_intensity(intensity): 
        plt.plot(intensity.xs(), intensity.values.T, linewidth=3, color='w') 
        plt.plot(intensity.xs(), intensity.values.T, linewidth=1)
        plt.grid(False) 
        plt.ylim(0)
        plt.ylabel("intensity [dB]")
        
def draw_pitch(pitch): 
    # Extract selected pitch contour, and
    # replace unvoiced samples by NaN to not plot 
    pitch_values = pitch.selected_array['frequency'] 
    pitch_values[pitch_values==0] = np.nan 
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=5, color='w')
    plt.plot(pitch.xs(), pitch_values, 'o', markersize=2) 
    plt.grid(False) 
    plt.ylim(0, pitch.ceiling) 
    plt.ylabel("fundamental frequency [Hz]")
    
@jit( forceobj=True)    
def process_frames(d,sound):
    
    out_dictionery = {'F1':{},'F2':{},'F3':{},'intensity':{},
                      'F1_mean':{},'F2_mean':{},'F3_mean':{},
                      'F1_median':{},'F2_median':{},'F3_median':{},
                      'time_f':{},'Event':{},'Speaker':{},
                      'speech_data_time':{},'speech_data':{},
                      'pitch':{},'spectrogram':{},'pitch_obj':{}}
    # ,'f':{},'t':{}
    for i in range(0,len(d['Event'])):
        snd_part = sound.extract_part(from_time=d['Start_time'][i],
                                                     to_time =d['End_time'][i], preserve_times=True )
    
        # snd_part = sound.extract_part(from_time=d['Start_time'][0],to_time =d['End_time'][0], preserve_times=True )
        out_dictionery['speech_data_time'][i]=snd_part.xs()
        out_dictionery['speech_data'][i]=snd_part.values.T
        pitch = snd_part.to_pitch()
        pitch_values = pitch.selected_array['frequency'] 
        pitch_values[pitch_values==0] = np.nan
        out_dictionery['pitch'][i] = pitch_values
        out_dictionery['pitch'][i] = pitch
        spectrogram = snd_part.to_spectrogram(window_length=0.02,maximum_frequency=8000)
        # t, f = spectrogram.x_grid(),spectrogram.y_grid()
        # out_dictionery['f'][i] = f
        # out_dictionery['t'][i] = t
        # sg_db = 10 * np.log10(spectrogram.values)
        out_dictionery['spectrogram'][i] = spectrogram
        out_dictionery['Event'][i]=d['Event'][i]
        out_dictionery['Speaker'][i]=d['Speaker'][i]
        intensity = snd_part.to_intensity()
        out_dictionery['intensity'][i]=intensity
        f0min=75
        f0max=800
        pointProcess = call(snd_part, "To PointProcess (periodic, cc)", f0min, f0max)
        numPoints = call(pointProcess, "Get number of points")
        # time step(s), num formants(int), formant ceiling(Hz), window length(s), Pre-emphasis from(Hz)
        formants = call(snd_part, "To Formant (burg)", 0.01, 5, 8000, 0.02, 50)
        
        f1_list = []
        f2_list = []
        f3_list = []
        f4_list = []
        time_f = []
        
            # Measure formants only at glottal pulses
        for point in range(0, numPoints):
            point += 1
            t = call(pointProcess, "Get time from index", point)
            f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
            f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
            f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
            f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
            time_f.append(t)
            f1_list.append(f1)
            f2_list.append(f2)
            f3_list.append(f3)
            f4_list.append(f4)
                
        if f1_list:
            f1_list_no_nan = [f1 for f1 in f1_list if str(f1) != 'nan']
            f2_list_no_nan = [f2 for f2 in f2_list if str(f2) != 'nan']
            f3_list_no_nan = [f3 for f3 in f3_list if str(f3) != 'nan']
            # f4_list_no_nan = [f4 for f4 in f4_list if str(f4) != 'nan']
           
            # calculate mean formants across pulses
            f1_mean = statistics.mean(f1_list_no_nan)
            f2_mean = statistics.mean(f2_list_no_nan)
            f3_mean = statistics.mean(f3_list_no_nan)
            # f4_mean = statistics.mean(f4_list_no_nan)
           
            # # calculate median formants across pulses, this is what is used in all subsequent calcualtions
            # # you can use mean if you want, just edit the code in the boxes below to replace median with mean
            f1_median = statistics.median(f1_list_no_nan)
            f2_median = statistics.median(f2_list_no_nan)
            f3_median = statistics.median(f3_list_no_nan)
            # f4_median = statistics.median(f4_list_no_nan)
           
            out_dictionery['time_f'][i]=time_f
            out_dictionery['F1'][i]=f1_list
            out_dictionery['F2'][i]=f2_list
            out_dictionery['F3'][i]=f3_list
            out_dictionery['F1_mean'][i]=f1_mean
            out_dictionery['F2_mean'][i]=f2_mean
            out_dictionery['F3_mean'][i]=f3_mean
            out_dictionery['F1_median'][i]=f1_median
            out_dictionery['F2_median'][i]=f2_median
            out_dictionery['F3_median'][i]=f3_median
        else:
            out_dictionery['time_f'][i]=[]
            out_dictionery['F1'][i]=[]
            out_dictionery['F2'][i]=[]
            out_dictionery['F3'][i]=[]
            out_dictionery['F1_mean'][i]=[]
            out_dictionery['F2_mean'][i]=[]
            out_dictionery['F3_mean'][i]=[]
            out_dictionery['F1_median'][i]=[]
            out_dictionery['F2_median'][i]=[]
            out_dictionery['F3_median'][i]=[]
        # 
        i += 1
        
    return out_dictionery
    